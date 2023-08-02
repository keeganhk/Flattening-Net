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

dataset_folder = os.path.join(data_root, 'PatchCollection')
ckpt_path = os.path.join(ckpt_root, 'para_loc.pth')

tr_bs = 64
tr_set = ParaTrainLoader_PatchCollection(dataset_folder)
tr_loader = DataLoader(tr_set, batch_size=tr_bs, shuffle=True, num_workers=8, worker_init_fn=seed_worker, drop_last=True)

net = S2PF(rescale_ebd=False).cuda()
max_lr, min_lr, num_epc = 1e-4, 1e-4, 100
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=max_lr, weight_decay=1e-8)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epc, eta_min=min_lr)
for epc_index in range(1, num_epc+1):
    net.cuda().train()
    num_accum = 0
    ckpt_loss = 0
    for pts in tqdm(tr_loader):
        optimizer.zero_grad()
        B, N = pts.size(0), pts.size(1)
        pts = pts.cuda()
        min_nnd = 1 / (np.sqrt(N)-1)
        ebd = net(pts)
        loss = repulsion_loss(ebd, min_nnd)
        num_accum += B
        ckpt_loss += (loss.item() * B)
        loss.backward()
        optimizer.step()
    scheduler.step()
    print('epoch: {}, rep loss: {}'.format(align_number(epc_index, 4), np.around(ckpt_loss/num_accum, 8)))
    torch.save(net.state_dict(), os.path.join(ckpt_root, 'para_loc_intermediate.pth'))
print('finished intermediate training.')


net = S2PF(rescale_ebd=True).cuda()
net.load_state_dict(torch.load(os.path.join(ckpt_root, 'para_loc_intermediate.pth')))
max_lr, min_lr, num_epc = 1e-4, 1e-4, 100
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=max_lr, weight_decay=1e-8)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epc, eta_min=min_lr)
for epc_index in range(1, num_epc+1):
    net.cuda().train()
    num_accum = 0
    ckpt_loss = 0
    for pts in tqdm(tr_loader):
        optimizer.zero_grad()
        B, N = pts.size(0), pts.size(1)
        pts = pts.cuda()
        np.random.seed()
        K = np.random.randint(8, 64+1)
        with torch.no_grad():
            pts = index_points(pts, fps(pts, K))
        min_nnd = 1 / (np.sqrt(K)-1)
        ebd = net(pts)
        loss = repulsion_loss(ebd, min_nnd*0.50)
        num_accum += B
        ckpt_loss += (loss.item() * B)
        loss.backward()
        optimizer.step()
    scheduler.step()
    print('epoch: {}, rep loss: {}'.format(align_number(epc_index, 4), np.around(ckpt_loss/num_accum, 8)))
    torch.save(net.cpu().state_dict(), ckpt_path)


