from .pkgs import *
from .utils import *



class ParaTrainLoader_ShapeNetCore(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, rotation_mode=None):
        self.dataset_folder = dataset_folder
        self.rotation_mode = rotation_mode
        assert rotation_mode in ['rot_z', 'rot_so3', None]
        self.class_list = parse_list_file(os.path.join(dataset_folder, 'class_list.txt'))
        self.total_list = parse_list_file(os.path.join(dataset_folder, 'total_list.txt'))
        self.num_models_total = len(self.total_list)
    def __getitem__(self, model_index):
        np.random.seed()
        model_name = self.total_list[model_index]
        class_name = model_name[:-5]
        cid = self.class_list.index(class_name)
        model_path = os.path.join(self.dataset_folder, '256', class_name, model_name + '.npy')
        pts = bounding_box_normalization(np.load(model_path).astype(np.float32))
        pts = bounding_box_normalization(random_anisotropic_scaling(pts, 2/3, 3/2))
        if self.rotation_mode == 'rot_z':
            pts = random_axis_rotation(pts, 'z')
        elif self.rotation_mode == 'rot_so3':
            pts = random_rotation(pts)
        pts = bounding_box_normalization(pts)
        return pts, model_name
    def __len__(self):
        return self.num_models_total


class ParaTestLoader_ShapeNetCore(torch.utils.data.Dataset):
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.class_list = parse_list_file(os.path.join(dataset_folder, 'class_list.txt'))
        self.total_list = parse_list_file(os.path.join(dataset_folder, 'total_list.txt'))
        self.num_models_total = len(self.total_list)
    def __getitem__(self, model_index):
        np.random.seed()
        model_name = self.total_list[model_index]
        class_name = model_name[:-5]
        cid = self.class_list.index(class_name)
        model_path = os.path.join(self.dataset_folder, '256', class_name, model_name + '.npy')
        pts = bounding_box_normalization(np.load(model_path).astype(np.float32))
        return pts, model_name
    def __len__(self):
        return self.num_models_total


class ParaTrainLoader_PatchCollection(torch.utils.data.Dataset):
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.h5_file = os.path.join(dataset_folder, 'points_100000_100_3.h5')
        self.num_patches_total = 100_000
    def __getitem__(self, patch_index):
        np.random.seed()
        fid = h5py.File(self.h5_file, 'r')
        pts = bounding_box_normalization(fid['points'][patch_index].astype(np.float32))
        fid.close()
        pts = bounding_box_normalization(random_rotation(bounding_box_normalization(pts)))
        return pts
    def __len__(self):
        return self.num_patches_total


class ParaTestLoader_PatchCollection(torch.utils.data.Dataset):
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.h5_file = os.path.join(dataset_folder, 'points_100000_100_3.h5')
        self.num_patches_total = 100_000
    def __getitem__(self, patch_index):
        np.random.seed()
        fid = h5py.File(self.h5_file, 'r')
        pts = bounding_box_normalization(fid['points'][patch_index].astype(np.float32))
        fid.close()
        return pts
    def __len__(self):
        return self.num_patches_total


class PatchLoader_PU147_Ups(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, use_augmentation):
        self.use_augmentation = use_augmentation
        self.sparse_folder = os.path.join(dataset_folder, 'train_patches', 'sparse')
        self.dense_folder = os.path.join(dataset_folder, 'train_patches', 'dense')
        self.num_samples = 24_000
    def __getitem__(self, index):
        np.random.seed()
        file_name = align_number(int(np.random.choice(self.num_samples, 1))+1, 5)
        pat_pts_s = np.loadtxt(os.path.join(self.sparse_folder, file_name + '.xyz')).astype(np.float32)
        pat_pts_d = np.loadtxt(os.path.join(self.dense_folder, file_name + '.xyz')).astype(np.float32)
        if self.use_augmentation:
            merge_rot = random_rotation(np.concatenate((pat_pts_s, pat_pts_d), axis=0))
            pat_pts_s = merge_rot[:256, :]
            pat_pts_d = merge_rot[256:, :]
        return pat_pts_s, pat_pts_d, file_name
    def __len__(self):
        return self.num_samples


class ParaLoader_ModelNet40_Cls(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, mode):
        self.dataset_folder = dataset_folder
        self.mode = mode
        self.data_file = os.path.join(dataset_folder, 'modelnet40_para_from_5000_' + mode + '.h5')
        f = h5py.File(self.data_file, 'r')
        self.num_models = f['data'].shape[0]
        f.close()
    def __getitem__(self, index):
        np.random.seed()
        f = h5py.File(self.data_file, 'r')
        pgi = bounding_box_normalization(f['data'][index].astype(np.float32))
        M = pgi.shape[0]
        m = int(np.sqrt(M))
        cid = f['labels'][index].astype(np.int64)
        f.close()
        if self.mode == 'train':
            pgi = random_anisotropic_scaling(pgi, 2/3, 3/2)
            pgi = random_translation(pgi, 0.20)
        return pgi, cid
    def __len__(self):
        return self.num_models


def ShapeNetPart_ObjectsParts():
    objects_names = [
        'airplane', 
        'bag', 
        'cap', 
        'car', 
        'chair', 
        'earphone', 
        'guitar', 
        'knife', 
        'lamp', 
        'laptop', 
        'motorbike', 
        'mug', 
        'pistol', 
        'rocket', 
        'skateboard', 
        'table'
    ]
    objects_parts = {
        'airplane': [0, 1, 2, 3], 
        'bag': [4, 5], 
        'cap': [6, 7], 
        'car': [8, 9, 10, 11], 
        'chair': [12, 13, 14, 15], 
        'earphone': [16, 17, 18], 
        'guitar': [19, 20, 21], 
        'knife': [22, 23],
        'lamp': [24, 25, 26, 27], 
        'laptop': [28, 29], 
        'motorbike': [30, 31, 32, 33, 34, 35], 
        'mug': [36, 37], 
        'pistol': [38, 39, 40], 
        'rocket': [41, 42, 43], 
        'skateboard': [44, 45, 46], 
        'table': [47, 48, 49]}
    return objects_names, objects_parts


def ShapeNetPart_PartsColors():
    objects_names, objects_parts = ShapeNetPart_ObjectsParts()
    num_parts = []
    for k, v in objects_parts.items():
        num_parts.append(len(v))
    cmap = cm.jet
    parts_colors = np.zeros((50, 3))
    i = 0
    for num in num_parts:
        base_colors = cmap(np.linspace(0, 1, num))[:, 0:3]
        for k in range(num):
            parts_colors[i, ...] = base_colors[k, ...]
            i += 1
    return parts_colors


def ShapeNetPart_ColorCode(points_with_labels):
    assert points_with_labels.ndim==2 and points_with_labels.size(-1)==4
    points = points_with_labels[:, 0:3].unsqueeze(0)
    labels = points_with_labels[:, -1].unsqueeze(0).long()
    parts_colors =  torch.tensor(ShapeNetPart_PartsColors()).unsqueeze(0).to(points_with_labels.device)
    color_codes = index_points(parts_colors, labels)
    points_color_coded = torch.cat((points, color_codes), dim=-1).squeeze(0)
    return points_color_coded


class ParaLoader_ShapeNetPart_Seg(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, mode):
        self.dataset_folder = dataset_folder
        self.mode = mode
        self.class_list = parse_list_file(os.path.join(dataset_folder, 'class_list.txt'))
        self.model_list = parse_list_file(os.path.join(dataset_folder, mode + '_list.txt'))
    def __getitem__(self, index):
        np.random.seed()
        model_name = self.model_list[index]
        class_name = model_name[:-5]
        cid = self.class_list.index(class_name)
        data = np.load(os.path.join(self.dataset_folder, 'shapenetpart_para_from_2048', class_name, model_name + '.npy'))
        M = data.shape[0]
        m = int(np.sqrt(M))
        pgi_pt = data[:, 0:3].astype(np.float32)
        pgi_lb = data[:, -1].astype(np.int64)
        if self.mode == 'train':
            pgi_pt = random_anisotropic_scaling(pgi_pt, 2/3, 3/2)
        return pgi_pt, pgi_lb, cid, model_name
    def __len__(self):
        return len(self.model_list)


