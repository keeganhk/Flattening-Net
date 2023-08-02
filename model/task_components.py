import os, sys
prj_root = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(prj_root)
from cdbs.pkgs import *
from cdbs.utils import *



class ResidualConv2D3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualConv2D3x3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), 
            nn.BatchNorm2d(out_channels)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), 
            nn.BatchNorm2d(out_channels)
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False), nn.BatchNorm2d(out_channels))
        self.nl = nn.ReLU(inplace=True)
    def forward(self, in_ftr):
        out_ftr = self.conv_2(self.nl(self.conv_1(in_ftr)))
        if self.in_channels != self.out_channels:
            out_ftr = self.nl(self.shortcut(in_ftr) + out_ftr)
        else:
            out_ftr = self.nl(in_ftr + out_ftr)
        return out_ftr


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_neighbors, num_layers):
        super(EdgeConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_neighbors = num_neighbors
        self.num_layers = num_layers
        assert num_layers in [1, 2]
        if self.num_layers == 1:
            self.smlp = SMLP(in_channels*2, out_channels, is_bn=True, nl='leakyrelu', slope=0.20)
        if self.num_layers == 2:
            smlp_1 = SMLP(in_channels*2, out_channels, is_bn=True, nl='leakyrelu', slope=0.20)
            smlp_2 = SMLP(out_channels, out_channels, is_bn=True, nl='leakyrelu', slope=0.20)
            self.smlp = nn.Sequential(smlp_1, smlp_2)
    def forward(self, pc_ftr):
        num_neighbors = self.num_neighbors
        batch_size, num_points, in_channels = pc_ftr.size()
        knn_indices = knn_search(pc_ftr.detach(), pc_ftr.detach(), num_neighbors)
        nb_ftr = index_points(pc_ftr, knn_indices)
        pc_ftr_rep = pc_ftr.unsqueeze(2).repeat(1, 1, num_neighbors, 1)
        edge_ftr = torch.cat((pc_ftr_rep, nb_ftr-pc_ftr_rep), dim=-1)
        out_ftr = self.smlp(edge_ftr.view(batch_size, num_points*num_neighbors, -1)).view(batch_size, num_points, num_neighbors, -1)
        out_ftr_max_pooled = torch.max(out_ftr, dim=2)[0]
        return out_ftr_max_pooled


class ClsHead(nn.Module):
    def __init__(self, Ci, Nc):
        super(ClsHead, self).__init__()
        self.Ci = Ci
        self.Nc = Nc
        head_dims = [Ci, 512, 256, Nc]
        linear_1 = nn.Linear(head_dims[0], head_dims[1], bias=False)
        bn_1 = nn.BatchNorm1d(head_dims[1])
        nl_1 = nn.LeakyReLU(True, 0.2)
        dp_1 = nn.Dropout(0.5)
        self.fc_1 = nn.Sequential(linear_1, bn_1, nl_1, dp_1)
        linear_2 = nn.Linear(head_dims[1], head_dims[2], bias=False)
        bn_2 = nn.BatchNorm1d(head_dims[2])
        nl_2 = nn.LeakyReLU(True, 0.2)
        dp_2 = nn.Dropout(0.5)
        self.fc_2 = nn.Sequential(linear_2, bn_2, nl_2, dp_2)
        self.fc_3 = nn.Linear(head_dims[2], head_dims[3], bias=False)
    def forward(self, cdw):
        Ci, Nc = self.Ci, self.Nc
        B, D, device = cdw.size(0), cdw.size(1), cdw.device
        logits = self.fc_3(self.fc_2(self.fc_1(cdw)))
        return logits


class FlatNetCls(nn.Module):
    def __init__(self, N_G, N_C, K, num_classes):
        super(FlatNetCls, self).__init__()
        self.N_G = N_G
        self.N_C = N_C
        self.K = K
        self.n_G = int(N_G ** 0.5)
        self.k = int(K ** 0.5)
        self.M = N_G * K
        self.m = self.n_G * self.k
        self.csm = get_concentric_square_masks(self.k)
        self.inner_mask, self.inter_mask, self.outer_mask = merge_concentric_square_masks(self.csm)
        self.num_inner = int(self.inner_mask.sum().item())
        self.num_inter = int(self.inter_mask.sum().item())
        self.num_outer = int(self.outer_mask.sum().item())
        self.lift = SMLP(3, 32, True, 'leakyrelu', 0.20)
        self.smlp = FC(96, 128, True, 'leakyrelu', 0.20)
        self.edge_conv_1 = EdgeConv(128, 128, 16, 1)
        self.edge_conv_2 = EdgeConv(128, 128, 16, 1)
        self.edge_conv_3 = EdgeConv(128, 256, 16, 1)
        self.fuse = SMLP(512, 1024, True, 'leakyrelu', 0.20)
        self.head = ClsHead(2048, num_classes)
    def forward(self, pgi):
        B = pgi.size(0)
        device = pgi.device
        I = pgi.permute(0, 2, 1).contiguous().view(B, 3, self.m, self.m)
        blocks, _ = get_pgi_blocks(I, self.N_G, self.N_C, self.k)
        blk_pts_inner = square_partition(blocks, self.inner_mask.to(device))
        blk_pts_inter = square_partition(blocks, self.inter_mask.to(device))
        blk_pts_outer = square_partition(blocks, self.outer_mask.to(device))
        v_inner = self.lift(blk_pts_inner.view(B*self.N_G, -1, 3)).max(dim=1)[0]
        v_inter = self.lift(blk_pts_inter.view(B*self.N_G, -1, 3)).max(dim=1)[0]
        v_outer = self.lift(blk_pts_outer.view(B*self.N_G, -1, 3)).max(dim=1)[0]
        ftr_0 = self.smlp(torch.cat((v_inner, v_inter, v_outer), dim=-1)).view(B, self.N_G, -1)
        ftr_1 = self.edge_conv_1(ftr_0)
        ftr_2 = self.edge_conv_2(ftr_1)
        ftr_3 = self.edge_conv_3(ftr_2)
        ftr = self.fuse(torch.cat((ftr_1, ftr_2, ftr_3), dim=-1))
        cdw = torch.cat((ftr.max(dim=1)[0], ftr.mean(dim=1)), dim=-1)
        logits = self.head(cdw)
        return logits


class FlatNetSeg(nn.Module):
    def __init__(self, N_G, N_C, K, num_o_classes, num_p_classes):
        super(FlatNetSeg, self).__init__()
        self.N_G = N_G
        self.N_C = N_C
        self.K = K
        self.n_G = int(N_G ** 0.5)
        self.k = int(K ** 0.5)
        self.M = N_G * K
        self.m = self.n_G * self.k
        self.num_o_classes = num_o_classes
        self.num_p_classes = num_p_classes
        self.csm = get_concentric_square_masks(self.k)
        self.inner_mask, self.inter_mask, self.outer_mask = merge_concentric_square_masks(self.csm)
        self.num_inner = int(self.inner_mask.sum().item())
        self.num_inter = int(self.inter_mask.sum().item())
        self.num_outer = int(self.outer_mask.sum().item())
        self.cid_fc = FC(num_o_classes, 64, True, 'leakyrelu', 0.20)
        self.init_ebd = SMLP(3, 32, True, 'leakyrelu', 0.20)
        self.lift = SMLP(32, 32, True, 'leakyrelu', 0.20)
        self.smlp = FC(96, 128, True, 'leakyrelu', 0.20)
        self.edge_conv_1 = EdgeConv(128, 64, 16, 2)
        self.edge_conv_2 = EdgeConv(64, 64, 16, 2)
        self.edge_conv_3 = EdgeConv(64, 128, 16, 2)
        self.fuse = SMLP(320, 256, True, 'leakyrelu', 0.20)
        head_smlp_1 = nn.Sequential(SMLP(288, 128, True, 'leakyrelu', 0.2), nn.Dropout(p=0.5))
        head_smlp_2 = nn.Sequential(SMLP(128, 64, True, 'leakyrelu', 0.2), nn.Dropout(p=0.5))
        head_smlp_3 = SMLP(64, 32, True, 'leakyrelu', 0.2)
        head_smlp_4 = SMLP(32, num_p_classes, False, 'none')
        self.head = nn.Sequential(head_smlp_1, head_smlp_2, head_smlp_3, head_smlp_4)
    def forward(self, pgi, cid):
        B = pgi.size(0)
        device = pgi.device
        cid_vec = self.cid_fc(F.one_hot(cid, self.num_o_classes).float().to(device))
        ftr_pgi = self.init_ebd(pgi)
        I = ftr_pgi.permute(0, 2, 1).contiguous().view(B, -1, self.m, self.m)
        blocks = get_pgi_ftr_blocks(I, self.N_G, self.N_C, self.k)
        blk_ftr_inner = square_partition(blocks, self.inner_mask.to(device))
        blk_ftr_inter = square_partition(blocks, self.inter_mask.to(device))
        blk_ftr_outer = square_partition(blocks, self.outer_mask.to(device))
        v_inner = self.lift(blk_ftr_inner.view(B*self.N_G, self.num_inner, -1)).max(dim=1)[0]
        v_inter = self.lift(blk_ftr_inter.view(B*self.N_G, self.num_inter, -1)).max(dim=1)[0]
        v_outer = self.lift(blk_ftr_outer.view(B*self.N_G, self.num_outer, -1)).max(dim=1)[0]
        ftr_0 = self.smlp(torch.cat((v_inner, v_inter, v_outer), dim=-1)).view(B, self.N_G, -1)
        ftr_1 = self.edge_conv_1(ftr_0)
        ftr_2 = self.edge_conv_2(ftr_1)
        ftr_3 = self.edge_conv_3(ftr_2)
        ftr_4 = self.fuse(torch.cat((cid_vec.unsqueeze(1).repeat(1, self.N_G, 1), ftr_1, ftr_2, ftr_3), dim=-1))
        pts_4 = pgi.permute(0, 2, 1).contiguous().view(B, -1, self.m, self.m)
        pts_4 = get_pgi_blocks(pts_4, self.N_G, self.N_C, self.k)[0]
        pts_4 = pts_4.view(B, self.N_G, self.K, 3)[:, :, self.K//2, :]
        nn_idx = knn_search(pts_4, pgi, 1).squeeze(-1)
        nn_ftr = index_points(ftr_4, nn_idx)
        pwf = torch.cat((nn_ftr, ftr_pgi), dim=-1)
        logits = self.head(pwf)
        return logits


class FlatNetUps(nn.Module):
    def __init__(self):
        super(FlatNetUps, self).__init__()
        self.encode_1 = nn.Sequential(ResidualConv2D3x3(3, 128), ResidualConv2D3x3(128, 512))
        self.decode_1 = nn.Sequential(ResidualConv2D3x3(512, 128), CU(128, 64, 1, True, 'relu'), CU(64, 3, 1, False, 'none'))
        self.encode_2 = nn.Sequential(ResidualConv2D3x3(3, 128), ResidualConv2D3x3(128, 512))
        self.decode_2 = nn.Sequential(ResidualConv2D3x3(512, 128), CU(128, 64, 1, True, 'relu'), CU(64, 3, 1, False, 'none'))
    def forward(self, img_0):
        img_1 = img_0 + self.decode_1(self.encode_1(img_0))
        img_2 = img_1 + self.decode_2(self.encode_2(img_1))
        return img_1, img_2


