import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
prj_root = os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(prj_root)
from cdbs.pkgs import *
from cdbs.utils import *
from cdbs.datasets import *
from model.para_components import *
data_root = '../../data'
ckpt_root = '../../ckpt'

dataset_folder = os.path.join(data_root, 'ShapeNetCore')
ckpt_path = os.path.join(ckpt_root, 'para_glb.pth')

tr_bs = 128
tr_set = ParaTrainLoader_ShapeNetCore(dataset_folder, 'rot_so3')
tr_loader = DataLoader(tr_set, batch_size=tr_bs, shuffle=True, num_workers=8, worker_init_fn=seed_worker, drop_last=True)

net = G2SD(num_grids=256).cuda()
max_lr, min_lr, num_epc = 5e-4, 5e-6, 500
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=max_lr, weight_decay=1e-8)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epc, eta_min=min_lr)

for epc_index in range(1, num_epc+1):
    net.cuda().train()
    num_accum = 0
    ckpt_loss = 0
    for (pts, names_list) in tqdm(tr_loader):
        optimizer.zero_grad()
        B = len(names_list)
        pts = pts.float().cuda()
        rec = net(pts)
        loss = chamfer_distance_cuda(pts, rec)
        num_accum += B
        ckpt_loss += (loss.item() * B)
        loss.backward()
        optimizer.step()
    scheduler.step()
    print('epoch: {}, cd loss: {}'.format(align_number(epc_index, 4), np.around(ckpt_loss/num_accum, 8)))
    torch.save(net.cpu().state_dict(), ckpt_path)


