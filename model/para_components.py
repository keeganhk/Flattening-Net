import os, sys
prj_root = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(prj_root)
from cdbs.pkgs import *
from cdbs.utils import *



class NbrAgg(nn.Module):
    def __init__(self, num_neighbors, out_channels):
        super(NbrAgg, self).__init__()
        self.num_neighbors = num_neighbors
        self.out_channels = out_channels
        self.smlp_1 = nn.Sequential(SMLP(7, 16, True, 'relu'), SMLP(16, out_channels, True, 'relu'))
        self.smlp_2 = SMLP(3, out_channels, True, 'relu')
        self.smlp_3 = SMLP(out_channels*2, out_channels, True, 'relu')
    def forward(self, pts):
        # pts: [B, N, 3]
        assert pts.ndim == 3 
        assert pts.size(2) == 3
        B, N, K, C = pts.size(0), pts.size(1), self.num_neighbors, self.out_channels
        knn_idx = knn_search(pts, pts, K+1)
        knn_pts = index_points(pts, knn_idx)
        abs_pts = knn_pts[:, :, :1, :]
        rel_nbs = knn_pts[:, :, 1:, :] - knn_pts[:, :, :1, :]
        dists = torch.sqrt((rel_nbs ** 2).sum(dim=-1, keepdim=True) + 1e-8)
        concat = torch.cat((abs_pts.repeat(1, 1, K, 1), rel_nbs, dists), dim=-1)
        nbs_pooled = self.smlp_1(concat.view(B*N, K, -1)).view(B, N, K, -1).max(dim=2)[0]
        pts_lifted = self.smlp_2(pts)
        pts_ebd = self.smlp_3(torch.cat((pts_lifted, nbs_pooled), dim=-1))
        return pts_ebd


class AttPool(nn.Module):
    def __init__(self, in_chs):
        super(AttPool, self).__init__()
        self.in_chs = in_chs
        self.linear_transform = SMLP(in_chs, in_chs, False, 'none')
    def forward(self, x):
        bs = x.size(0)
        num_pts = x.size(1)
        assert x.ndim==3 and x.size(2)==self.in_chs
        scores = F.softmax(self.linear_transform(x), dim=1)
        y = (x * scores).sum(dim=1)
        return y


class CdwExtractor(nn.Module):
    def __init__(self):
        super(CdwExtractor, self).__init__()
        self.loc_agg = NbrAgg(16, 32)
        self.res_smlp_1 = ResSMLP(32, 64)
        self.res_smlp_2 = ResSMLP(128, 128)
        self.fuse = SMLP(352, 512, True, 'relu')
        self.att_pool = AttPool(512)
        self.fc = nn.Sequential(FC(1024, 512, True, 'relu'), FC(512, 1024, True, 'relu'), FC(1024, 1024, False, 'none'))
    def forward(self, pts):
        B, N, _ = pts.size()
        ftr_1 = self.loc_agg(pts)
        ftr_2 = self.res_smlp_1(ftr_1)
        ftr_3 = self.res_smlp_2(torch.cat((ftr_2, ftr_2.max(dim=1, keepdim=True)[0].repeat(1, N, 1)), dim=-1))
        ftr_4 = self.fuse(torch.cat((ftr_1, ftr_2, ftr_3, ftr_3.max(dim=1, keepdim=True)[0].repeat(1, N, 1)), dim=-1))
        cdw = self.fc(torch.cat((ftr_4.max(dim=1)[0], self.att_pool(ftr_4)), dim=-1))
        return cdw


class G2SD(nn.Module):
    def __init__(self, num_grids):
        super(G2SD, self).__init__()
        self.num_grids = num_grids
        self.grid_size = int(np.sqrt(num_grids))
        assert self.grid_size**2 == self.num_grids
        self.lattice = torch.tensor(build_lattice(self.grid_size, self.grid_size)[0])
        self.backbone = CdwExtractor()
        fold_1_1 = SMLP(1026, 256, True, 'relu')
        fold_1_2 = SMLP(256, 128, True, 'relu')
        fold_1_3 = SMLP(128, 64, True, 'relu')
        fold_1_4 = SMLP(64, 3, False, 'none')
        self.fold_1 = nn.Sequential(fold_1_1, fold_1_2, fold_1_3, fold_1_4)
        fold_2_1 = SMLP(1027, 256, True, 'relu')
        fold_2_2 = SMLP(256, 128, True, 'relu')
        fold_2_3 = SMLP(128, 64, True, 'relu')
        fold_2_4 = SMLP(64, 3, False, 'none')
        self.fold_2 = nn.Sequential(fold_2_1, fold_2_2, fold_2_3, fold_2_4)
    def forward(self, pts):
        B, N, device = pts.size(0), pts.size(1), pts.device
        grids = (self.lattice).unsqueeze(0).repeat(B, 1, 1).to(device)
        cdw = self.backbone(pts)
        cdw_dup = cdw.unsqueeze(1).repeat(1, self.num_grids, 1)
        concat_1 = torch.cat((cdw_dup, grids), dim=-1)
        rec_1 = self.fold_1(concat_1)
        concat_2 = torch.cat((cdw_dup, rec_1), dim=-1)
        rec_2 = self.fold_2(concat_2)
        return rec_2


class PatCdwExtractor(nn.Module):
    def __init__(self):
        super(PatCdwExtractor, self).__init__()
        self.lift = SMLP(3, 16, True, 'relu')
        self.res_smlp_1 = ResSMLP(16, 32)
        self.res_smlp_2 = ResSMLP(64, 64)
        self.fuse = SMLP(176, 128, True, 'relu')
        self.att_pool = AttPool(128)
        self.fc = nn.Sequential(FC(256, 128, True, 'relu'), FC(128, 128, True, 'relu'), FC(128, 128, False, 'none'))
    def forward(self, pts):
        B, N, _ = pts.size()
        ftr_1 = self.lift(pts)
        ftr_2 = self.res_smlp_1(ftr_1)
        ftr_3 = self.res_smlp_2(torch.cat((ftr_2, ftr_2.max(dim=1, keepdim=True)[0].repeat(1, N, 1)), dim=-1))
        ftr_4 = self.fuse(torch.cat((ftr_1, ftr_2, ftr_3, ftr_3.max(dim=1, keepdim=True)[0].repeat(1, N, 1)), dim=-1))
        cdw = self.fc(torch.cat((ftr_4.max(dim=1)[0], self.att_pool(ftr_4)), dim=-1))
        return cdw


class S2PF(nn.Module):
    def __init__(self, rescale_ebd=True):
        super(S2PF, self).__init__()
        self.rescale_ebd = rescale_ebd
        self.backbone = PatCdwExtractor()
        unfold_1_1 = SMLP(131, 128, True, 'relu')
        unfold_1_2 = SMLP(128, 128, True, 'relu')
        unfold_1_3 = SMLP(128, 64, True, 'relu')
        unfold_1_4 = SMLP(64, 2, False, 'none')
        self.unfold_1 = nn.Sequential(unfold_1_1, unfold_1_2, unfold_1_3, unfold_1_4)
        unfold_2_1 = SMLP(130, 128, True, 'relu')
        unfold_2_2 = SMLP(128, 128, True, 'relu')
        unfold_2_3 = SMLP(128, 64, True, 'relu')
        unfold_2_4 = SMLP(64, 2, False, 'none')
        self.unfold_2 = nn.Sequential(unfold_2_1, unfold_2_2, unfold_2_3, unfold_2_4)
    def forward(self, pts):
        B, N, device = pts.size(0), pts.size(1), pts.device
        cdw = self.backbone(pts)
        cdw_dup = cdw.unsqueeze(1).repeat(1, N, 1)
        ebd_mid = self.unfold_1(torch.cat((cdw_dup, pts), dim=-1))
        ebd = self.unfold_2(torch.cat((cdw_dup, ebd_mid), dim=-1))
        if self.rescale_ebd:
            return rescale_pe(ebd, 0, 1)
        else:
            return ebd


class FlatteningNet(nn.Module):
    def __init__(self, para_g_params, para_l_params, N_G, N_C, K):
        super(FlatteningNet, self).__init__()
        self.N_G = N_G
        self.N_C = N_C
        self.K = K
        is_square_number(N_G)
        is_square_number(K)
        n_G = int(N_G ** 0.5)
        k = int(K ** 0.5)
        M = N_G * K
        m = n_G * k
        self.n_G = n_G
        self.k = k
        self.M = M
        self.m = m
        self.para_g_module = G2SD(num_grids=N_G)
        self.para_g_module.load_state_dict(torch.load(para_g_params))
        self.para_g_module.eval()
        self.para_l_module = S2PF()
        self.para_l_module.load_state_dict(torch.load(para_l_params))
        self.para_l_module.eval()
    def forward(self, pts):
        B, N, device = pts.size(0), pts.size(1), pts.device
        N_G = self.N_G
        N_C = self.N_C
        K = self.K
        n_G = self.n_G
        k = self.k
        M = self.M
        m = self.m
        self.para_g_module.to(device)
        self.para_l_module.to(device)
        with torch.no_grad():
            pts_g = index_points(pts, fps(pts, N_G))
            rec_g = self.para_g_module(pts_g)
            pts_c = index_points(pts, knn_search(pts, rec_g, N_C))
            pts_c_n = normalize_anchor_patches(pts_c)
            ebd_c = rescale_pe(self.para_l_module(pts_c_n.view(B*N_G, N_C, 3)), 0+1e-6, 1-1e-6).view(B, N_G, N_C, 2)
            pgi_local = seperately_grid_resample_patches(pts_c, ebd_c, K)
            pgi_global = assemble_separate_patch_parameterizations(pgi_local)
            pgi = pgi_global.view(B, 3, M).permute(0, 2, 1).contiguous()
        return pgi


