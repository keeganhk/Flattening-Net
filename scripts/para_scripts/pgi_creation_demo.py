import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
prj_root = os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(prj_root)
from cdbs.pkgs import *
from cdbs.utils import *
from cdbs.datasets import *
from model.para_components import FlatteningNet
data_root = '../../data'
ckpt_root = '../../ckpt'

N = 5000
N_G = 256
N_C = 50
K = 100
n_G = int(N_G ** 0.5)
k = int(K ** 0.5)
M = N_G * K
m = n_G * k

para_g_params = os.path.join(ckpt_root, 'para_glb.pth')
para_l_params = os.path.join(ckpt_root, 'para_loc.pth')
net = FlatteningNet(para_g_params, para_l_params, N_G, N_C, K).cuda().eval()
print('config and load Flattening-Net parameters.')

pts = []
load_folder = './toy_examples'
prefix_list = []
for load_path in glob.glob(os.path.join(load_folder, '*.xyz')):
    prefix_list.append(load_path.split('/')[-1].split('.')[0])
    pts.append(np.loadtxt(load_path).reshape(1, -1, 3).astype(np.float32))
pts = torch.tensor(np.concatenate(pts, axis=0)).cuda()
B = pts.size(0)

with torch.no_grad():
    pgi = net(pts)
    pgi_as_img = pgi.permute(0, 2, 1).contiguous().view(-1, 3, m, m)

save_folder = './pgi_cache'
for bid in range(B):
    save_pc(os.path.join(save_folder, prefix_list[bid] + '_para.ply'), pgi[bid])
    visualize_pgi(pgi, 2)[bid].save(os.path.join(save_folder, prefix_list[bid] + '_para.png'))


