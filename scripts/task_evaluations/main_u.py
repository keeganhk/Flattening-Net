import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
prj_root = os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(prj_root)
from cdbs.pkgs import *
from cdbs.utils import *
from cdbs.datasets import *
from model.para_components import *
from model.task_components import *
data_root = '../../data'
ckpt_root = '../../ckpt'

dataset_folder = os.path.join(data_root, 'PU147')
ckpt_path = os.path.join(ckpt_root, 'flatnet_ups.pth')

para_net = G2SD(num_grids=256)
para_net.load_state_dict(torch.load(os.path.join(ckpt_root, 'para_glb' + '.pth')))
para_net.cuda().eval()
para_net_d = G2SD(num_grids=1024)
para_net_d.load_state_dict(torch.load(os.path.join(ckpt_root, 'para_glb' + '.pth')))
para_net_d.cuda().eval()
print('load para parameters.')

# training
train_bs = 128
train_set = PatchLoader_PU147_Ups(dataset_folder, True)
train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True, num_workers=8, worker_init_fn=seed_worker, drop_last=True)

net = FlatNetUps().cuda()
max_lr, min_lr, num_epc = 1e-3, 1e-4, 500
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=max_lr, weight_decay=1e-8)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epc, eta_min=min_lr)
criterion = nn.L1Loss()

net.train()
for epc in range(1, num_epc+1):
    num_samples = 0
    epoch_loss = 0
    for (pat_pts_s, pat_pts_d_gt, name_list) in tqdm(train_loader):
        optimizer.zero_grad()
        pat_pts_s = pat_pts_s.cuda()
        pat_pts_d_gt = pat_pts_d_gt.cuda()
        B = len(name_list)
        with torch.no_grad():
            pgi_s = para_net(pat_pts_s)
            pgi_s = index_points(pat_pts_s, knn_search(pat_pts_s, pgi_s, 1).squeeze(-1))
            pgi_s = pgi_s.permute(0, 2, 1).contiguous().view(B, 3, 16, 16)
            img_0 = F.interpolate(pgi_s, scale_factor=2, mode='bicubic')
        img_1, img_2 = net(img_0)
        pat_pts_d1 = img_1.view(B, 3, -1).permute(0, 2, 1).contiguous()
        pat_pts_d2 = img_2.view(B, 3, -1).permute(0, 2, 1).contiguous()
        loss_1 = unfm_loss(pat_pts_d2, 0.0002, 0.016)
        loss_2 = chamfer_distance_cuda(pat_pts_d2, pat_pts_d_gt)
        hidden_constraint = criterion(pat_pts_d1, para_net_d(pat_pts_s)) + earth_mover_distance_cuda(pat_pts_d1, pat_pts_d_gt)
        if epc <= (num_epc*0.1):
            loss = hidden_constraint*0.01 + loss_1*0.01 + loss_2
        elif epc <= (num_epc*0.25):
            loss = hidden_constraint*0.01 + loss_2
        else:
            loss = loss_2
        loss.backward()
        optimizer.step()
        num_samples += B
        epoch_loss += (loss_2.item() * B)
    scheduler.step()
    epoch_loss = np.around(epoch_loss/num_samples, 6)
    print('epoch: {}, loss: {}'.format(epc, epoch_loss))
    torch.save(net.state_dict(), ckpt_path)

# testing
net = FlatNetUps().cuda()
net.load_state_dict(torch.load(ckpt_path))
net.cuda().eval()

B = 24
test_file_list = os.listdir(os.path.join(dataset_folder, 'test_models', 'sparse'))
# if not os.path.exists('./upsampling_results'):
#     os.mkdir('./upsampling_results')
mean_cd = 0
for test_file in tqdm(test_file_list):
    test_model = test_file[:-4]
    load_path_s = os.path.join(dataset_folder, 'test_models', 'sparse', test_model + '.xyz')
    load_path_d = os.path.join(dataset_folder, 'test_models', 'dense', test_model + '.xyz')
    pc_s = torch.tensor(np.loadtxt(load_path_s).astype(np.float32)).unsqueeze(0).cuda()
    pc_d_gt = torch.tensor(np.loadtxt(load_path_d).astype(np.float32)).unsqueeze(0).cuda()
    pat_pts_s = decompose_points(pc_s, B, 256)[1].squeeze(0)
    centroids = torch.mean(pat_pts_s, dim=1)
    pat_pts_s_nrm = pat_pts_s - centroids.unsqueeze(1)
    scales = torch.sqrt(torch.abs(torch.sum(pat_pts_s_nrm**2, dim=-1))).max(dim=1)[0]
    pat_pts_s_nrm = pat_pts_s_nrm / scales.unsqueeze(-1).unsqueeze(-1)
    with torch.no_grad():
        pgi_s = para_net(pat_pts_s_nrm)
        pgi_s = index_points(pat_pts_s_nrm, knn_search(pat_pts_s_nrm, pgi_s, 1).squeeze(-1))
        pgi_s = pgi_s.permute(0, 2, 1).contiguous().view(B, 3, 16, 16)
        img_i = F.interpolate(pgi_s, scale_factor=2, mode='bicubic')
        img_h, img_d = net(img_i)
        pat_pts_d = img_d.view(B, 3, -1).permute(0, 2, 1).contiguous()
    pat_pts_d = pat_pts_d * scales.unsqueeze(-1).unsqueeze(-1) + centroids.unsqueeze(1)
    pc_d = index_points(pat_pts_d.view(1, -1, 3), fps(pat_pts_d.view(1, -1, 3), 8192))
    cd_metric = compute_chamfer_l2(np.asarray(pc_d.squeeze(0).float().cpu()), np.asarray(pc_d_gt.squeeze(0).float().cpu()))
    print('[{}] -> {} x10^(-3)'.format(test_model, np.around(cd_metric*1000, 2)))
    mean_cd += cd_metric
    # save_path = os.path.join('./upsampling_results', test_model + '_4x_upsampled.ply')
    # save_pc(save_path, pc_d.squeeze(0))
    # save_pc(save_path, vedo.load(save_path).smooth_mls_2d().points())
mean_cd /= len(test_file_list)
print('mean cd metric: {} x10^(-3)'.format(np.around(mean_cd*1000, 2)))


