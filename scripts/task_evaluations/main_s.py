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
dataset_folder = os.path.join(data_root, 'ShapeNetPart')
ckpt_path = os.path.join(ckpt_root, 'flatnet_seg.pth')

N = 2048
N_G = 256
N_C = 24
k = 8
n_G = int(N_G ** 0.5)
K = (k ** 2)
M = N_G * K
m = n_G * k

train_bs = 24
train_set = ParaLoader_ShapeNetPart_Seg(dataset_folder, 'train')
train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True, num_workers=8, worker_init_fn=seed_worker, drop_last=True)
test_bs = 24
test_set = ParaLoader_ShapeNetPart_Seg(dataset_folder, 'test')
test_loader = DataLoader(test_set, batch_size=test_bs, shuffle=False, num_workers=8, worker_init_fn=seed_worker, drop_last=False)
num_o_classes = 16
num_p_classes = 50
snp_objects_names, snp_objects_parts = ShapeNetPart_ObjectsParts()

# training
net = FlatNetSeg(N_G, N_C, K, num_o_classes, num_p_classes).cuda()
max_lr = 1e-1
min_lr = 1e-3
num_epc = 150
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=max_lr, momentum=0.9, weight_decay=1e-6)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epc, eta_min=min_lr)

best_val_miou = 0
for epc in range(1, num_epc+1):
    # training
    net.train()
    epoch_loss = 0
    num_samples = 0
    for (pgi_pt, pgi_lb, cid, name_list) in tqdm(train_loader):
        optimizer.zero_grad()
        pgi_pt = pgi_pt.float().cuda()
        pgi_lb = pgi_lb.long().cuda()
        cid = cid.long().cuda()
        B = pgi_pt.size(0)
        logits = net(pgi_pt, cid)
        loss = compute_smooth_cross_entropy(logits.view(-1, num_p_classes), pgi_lb.view(-1), eps=0.00)
        loss.backward()
        optimizer.step()
        num_samples += B
        epoch_loss += (loss.item() * B)
    scheduler.step()
    epoch_loss = np.around(epoch_loss/num_samples, 6)
    print('epoch: {}, loss: {}'.format(epc, epoch_loss))
    # online evaluation
    cond_1 = (epc<=int(num_epc*0.8) and np.mod(epc, 5)==0)
    cond_2 = (epc>=int(num_epc*0.8) and np.mod(epc, 1)==0)
    if epc<=3 or cond_1 or cond_2:
        net.eval()
        iou_list = []
        for (pgi_pt, pgi_lb, cid, name_list) in tqdm(test_loader):
            pgi_pt = pgi_pt.float().cuda()
            pgi_lb = pgi_lb.long().cuda()
            cid = cid.long().cuda()
            B = pgi_pt.size(0)
            with torch.no_grad():
                logits = net(pgi_pt, cid)
            preds = np.asarray(logits.argmax(dim=-1).cpu())
            labels = np.asarray(pgi_lb.cpu())
            for bid in range(B):
                L_this = labels[bid]
                P_this = preds[bid]
                class_name = name_list[bid][:-5]
                parts = snp_objects_parts[class_name]
                this_parts_iou = []
                for part_this in parts:
                    if (L_this==part_this).sum() == 0:
                        this_parts_iou.append(1.0)
                    else:
                        I = np.sum(np.logical_and(P_this==part_this, L_this==part_this))
                        U = np.sum(np.logical_or(P_this==part_this, L_this==part_this))
                        this_parts_iou.append(float(I) / float(U))
                this_iou = np.array(this_parts_iou).mean()
                iou_list.append(this_iou)
        val_miou = np.around(np.array(iou_list).mean()*100, 2)
        if val_miou >= best_val_miou:
            best_val_miou = val_miou
            torch.save(net.state_dict(), ckpt_path)
        print('epoch: {}: val miou: {}%,  best val miou: {}%'.format(epc, val_miou, best_val_miou))

# testing
net = FlatNetSeg(N_G, N_C, K, num_o_classes, num_p_classes).cuda()
net.load_state_dict(torch.load(ckpt_path))
net.eval()

net.eval()
iou_list = []
for (pgi_pt, pgi_lb, cid, name_list) in tqdm(test_loader):
    pgi_pt = pgi_pt.float().cuda()
    pgi_lb = pgi_lb.long().cuda()
    cid = cid.long().cuda()
    B = pgi_pt.size(0)
    with torch.no_grad():
        logits = net(pgi_pt, cid)
    preds = np.asarray(logits.argmax(dim=-1).cpu())
    labels = np.asarray(pgi_lb.cpu())
    for bid in range(B):
        L_this = labels[bid]
        P_this = preds[bid]
        class_name = name_list[bid][:-5]
        parts = snp_objects_parts[class_name]
        this_parts_iou = []
        for part_this in parts:
            if (L_this==part_this).sum() == 0:
                this_parts_iou.append(1.0)
            else:
                I = np.sum(np.logical_and(P_this==part_this, L_this==part_this))
                U = np.sum(np.logical_or(P_this==part_this, L_this==part_this))
                this_parts_iou.append(float(I) / float(U))
        this_iou = np.array(this_parts_iou).mean()
        iou_list.append(this_iou)
best_test_miou = np.array(iou_list).mean()
print('best test miou: {}%'.format(np.around(best_test_miou*100, 2)))


