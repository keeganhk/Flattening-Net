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
dataset_folder = os.path.join(data_root, 'ModelNet40')
ckpt_path = os.path.join(ckpt_root, 'flatnet_cls.pth')

N = 5000
N_G = 256
N_C = 50
k = 10
n_G = int(N_G ** 0.5)
K = (k ** 2)
M = N_G * K
m = n_G * k

train_bs = 64
train_set = ParaLoader_ModelNet40_Cls(dataset_folder, 'train')
train_loader = DataLoader(train_set, batch_size=train_bs, shuffle=True, num_workers=8, worker_init_fn=seed_worker, drop_last=True)
test_bs = 64
test_set = ParaLoader_ModelNet40_Cls(dataset_folder, 'test')
test_loader = DataLoader(test_set, batch_size=test_bs, shuffle=False, num_workers=8, worker_init_fn=seed_worker, drop_last=False)
num_classes = 40

# training
net = FlatNetCls(N_G, N_C, K, num_classes).cuda()
max_lr = 1e-1
min_lr = 5e-4
num_epc = 500
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=max_lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epc, eta_min=min_lr)

best_test_acc = 0
for epc in range(1, num_epc+1):
    net.train()
    epoch_loss = 0
    num_samples = 0
    num_correct = 0
    for (pgi, cid) in tqdm(train_loader):
        optimizer.zero_grad()
        pgi = pgi.cuda()
        cid = cid.long().cuda()
        B = pgi.size(0)
        logits = net(pgi)
        loss = compute_smooth_cross_entropy(logits, cid, 0.20)
        loss.backward()
        optimizer.step()
        preds = logits.argmax(dim=-1).detach()
        num_samples += B
        num_correct += (preds==cid).sum().item()
        epoch_loss += (loss.item() * B)
    scheduler.step()
    epoch_loss = np.around(epoch_loss/num_samples, 6)
    train_acc = np.around((num_correct/num_samples)*100, 2)
    print('epoch: {}, train acc: {}%, loss: {}'.format(epc, train_acc, epoch_loss))
    net.eval()
    num_samples = 0
    num_correct = 0
    for (pgi, cid) in tqdm(test_loader):
        pgi = pgi.cuda()
        cid = cid.long().cuda()
        B = pgi.size(0)
        with torch.no_grad():
            logits = net(pgi)
        preds = logits.argmax(dim=-1).detach()
        num_samples += B
        num_correct += (preds==cid).sum().item()
    test_acc = np.around((num_correct/num_samples)*100, 2)
    if test_acc >= best_test_acc:
        best_test_acc = test_acc
        torch.save(net.state_dict(), ckpt_path)
    print('epoch: {}: test acc: {}%,  best test acc: {}%'.format(epc, test_acc, best_test_acc))

# testing
net = FlatNetCls(N_G, N_C, K, num_classes).cuda()
net.load_state_dict(torch.load(ckpt_path))
net.eval()

num_samples = 0
num_correct = 0
for (pgi, cid) in tqdm(test_loader):
    pgi = pgi.cuda()
    cid = cid.long().cuda()
    B = pgi.size(0)
    with torch.no_grad():
        logits = net(pgi)
    preds = logits.argmax(dim=-1).detach()
    num_samples += B
    num_correct += (preds==cid).sum().item()
best_test_acc = np.around((num_correct/num_samples)*100, 2)
print('best test acc: {}%'.format(best_test_acc))


