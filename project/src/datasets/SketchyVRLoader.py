import os
import warnings, traceback
import torch
from torch._six import container_abcs
import numpy as np
import pandas
from src.utils.point_cloud_utils import resample_mesh, normalize, rotate_point_cloud
from src.utils.io import read_mesh
import h5py
from src.utils import utils

POINT_CLOUD_FILE_EXT = '.npy'
MESH_FILE_EXT = '.obj'

log = utils.get_logger(__name__)

    
# Pair dataset for deformer
class Shapes(torch.utils.data.Dataset): 
    DO_NOT_BATCH = ['source_face', 'source_mesh', 'target_face', 'target_mesh', 
    'source_mesh_obj', 'target_mesh_obj', 'face', 'mesh', 'mesh_obj'
]
    def __init__(self, opt, mode='train', source_type='shape'):
        """
        mode: [train, val, test]
        """
        self.mode = mode
        self.opt = opt
        self.normalize_method = 'unit_box'
        self.source_type = source_type

        if self.source_type == 'sketch':
            # sketch->shape 
            name_list = os.path.join(self.opt['data_dir'], self.opt['{}_list'.format(mode)])
            self.name_list = [line.rstrip() for line in open(name_list)]
        else: 
            # shape->shape training and validation
            self.name_list = self._load_from_split_file(mode)

        if self.mode == 'test':
            name_list = '/vol/research/sketching/projects/VR_Sketch_lightning/project/data/shapenet_split/test_pair.txt'
            self.test_pair_list = [line.rstrip() for line in open(name_list)]

    def __len__(self):
        return len(self.name_list)

    def _get_mesh_path_old(self, name):
        return os.path.join(self.opt['mesh_dir'], self.opt['category'], name, "model.obj")
    def _get_sketch_path(self, name):
        return os.path.join(self.opt['data_dir'], self.opt['sketch_dir'], name + POINT_CLOUD_FILE_EXT)

    def _load_from_split_file(self, split):
        df = pandas.read_csv(self.opt['split_file'])
        # find names from the category and split
        df = df.loc[(df.synsetId == int(self.opt['category'])) & (df.split == split)]
        names = df.modelId.values
        return names
    def _get_mesh_path(self, name):
        return os.path.join(self.opt['data_dir'], 'shape', name + POINT_CLOUD_FILE_EXT)

    def get_item_by_name_mesh(self, name):
        mesh_path = self._get_mesh_path(name)
        points = np.load(mesh_path).astype(np.float32) #[n, 3]
        points = torch.tensor(points)
        points, center, scale = normalize(points, self.normalize_method) #[1024, 3]
        result = {'shape': points, 'file': name}

        return result

    def get_item_by_name_sketch(self, name):
        sketch_path = self._get_sketch_path(name)
        points = np.load(sketch_path).astype(np.float32) #[n, 3]
        points = torch.tensor(points)
        points, center, scale = normalize(points, self.normalize_method) #[1024, 3]
        result = {'shape': points, 'file': name}

        return result

    def get_item_by_name_mesh_old(self, name):
        mesh_path = self._get_mesh_path_old(name)
        V_mesh, F_mesh, mesh_obj = read_mesh(mesh_path, return_mesh=True)
        points = resample_mesh(mesh_obj, self.opt['num_points'])   
        points[:, :3], center, scale = normalize(points[:, :3], self.normalize_method)
        shape = points[:, :3].clone() #[1024, 3]

        result = {'shape': shape, 'file': name}

        if self.mode == 'test':
            V_mesh = V_mesh[:,:3]
            F_mesh = F_mesh[:,:3]
            V_mesh = (V_mesh - center) / scale
            result.update({'mesh': V_mesh, 'face': F_mesh, 'mesh_obj': mesh_obj})
        return result

    def get_sample(self, index):
        if self.mode == 'test':
            name, name_2 = self.test_pair_list[index].split('_')

        else:
            index_2 = np.random.randint(self.__len__())

            name = self.name_list[index]
            name_2 = self.name_list[index_2]

        if self.source_type == 'sketch': # sketch -> shape
            source_data = self.get_item_by_name_sketch(name) 
        else: # shape -> shape
            source_data = self.get_item_by_name_mesh(name) 

        target_data = self.get_item_by_name_mesh(name_2)
        
        result = {'source_' + k: v for k, v in source_data.items()} 
        result.update({'target_' + k: v for k, v in target_data.items()}) 

        return result

    def __getitem__(self, index):
        for _ in range(10):
            index = index % self.__len__()
            try:
                return self.get_sample(index)
            except Exception as e:
                warnings.warn(f"Error loading sample {index}: " + ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)))
                # import ipdb; ipdb.set_trace()
                index += 1

    @classmethod
    def collate(cls, batch):
        batched = {}
        elem = batch[0]
        for key in elem:
            if key in cls.DO_NOT_BATCH:
                batched[key] = [e[key] for e in batch]
            else:
                try:
                    batched[key] = torch.utils.data.dataloader.default_collate([e[key] for e in batch])
                except Exception as e:
                    print(e)
                    print(key)
                    # import ipdb; ipdb.set_trace()
                    print()

        return batched


    @staticmethod
    def uncollate(batched):
        for k, v in batched.items():
            if isinstance(v, torch.Tensor):
                batched[k] = v.cuda()
            elif isinstance(v, container_abcs.Sequence):
                if isinstance(v[0], torch.Tensor):
                    batched[k] = [e.cuda() for e in v]
        return batched

# Datloader for Deformer only
class Shapes_Deformer(Shapes):
    def __init__(self, opt, mode='train', source_type='shape'):
        """
        mode: [train, val, test]
        """
        self.mode = mode
        self.opt = opt
        self.normalize_method = 'unit_box'
        self.source_type = source_type

        # shape->shape training and validation
        name_list = self.opt['{}_list'.format(mode)].format(self.opt['category'])
        self.name_list = [line.rstrip() for line in open(name_list)]

    def _get_mesh_path(self, name):
        return os.path.join(self.opt['data_dir'], self.opt['category'], name + POINT_CLOUD_FILE_EXT)


class Mix_Shapes(Shapes):  # sketch (fixed)-> shape
    def __init__(self, opt, name_list, load_test_pairs=False, fixed_source_index=None, fixed_target_index=None):
        self.opt = opt
        self.fixed_source_index = fixed_source_index
        self.fixed_target_index = fixed_target_index
        self.load_test_pairs = load_test_pairs
        self.normalize_method = 'unit_box'

        if self.opt['source_type'] == 'sketch':
            # sketch->shape 
            self.name_list = [line.rstrip() for line in open(name_list)]
            # shape->shape training and validation
            if self.load_test_pairs:
                self.name_list_target = self._load_from_split_file('val')
            else:
                self.name_list_target = self._load_from_split_file('train')
        else:
            NotImplementedError

    def get_sample(self, index):
        index_2 = np.random.randint(len(self.name_list_target))

        if self.fixed_source_index is not None:
            index = self.fixed_source_index

        if self.fixed_target_index is not None:
            index_2 = self.fixed_target_index
        
        name = self.name_list[index]
        if self.load_test_pairs:
            # TODO: prepare test pairs list
            name_2 = self.name_list_target[index]
        else:
            name_2 = self.name_list_target[index_2]

        if self.opt['source_type'] == 'sketch': # sketch -> shape
            source_data = self.get_item_by_name_sketch(name) 
        else: # shape -> shape
            source_data = self.get_item_by_name_mesh(name) 

        target_data = self.get_item_by_name_mesh(name_2)
        
        result = {'source_' + k: v for k, v in source_data.items()} 
        result.update({'target_' + k: v for k, v in target_data.items()}) 

        return result

class Mix_Shapes_v2(Shapes):  # sketch -> shape (fixed)
    def __init__(self, opt, mode='train', source_type='shape'):
        super().__init__(opt, mode, source_type)
        if self.source_type == 'sketch':
            # sketch->shape 
            name_list = os.path.join(self.opt['data_dir'], self.opt['{}_list'.format(mode)])
            self.name_list_source = [line.rstrip() for line in open(name_list)]
            self.name_list = self._load_from_split_file(mode)
        else:
            NotImplementedError
    def __len__(self):
        return len(self.name_list)

    def get_sample(self, index):
        index_2 = np.random.randint(len(self.name_list_source))

        name = self.name_list[index]
        name_2 = self.name_list_source[index_2]

        if self.source_type == 'sketch': # sketch -> shape
            source_data = self.get_item_by_name_sketch(name_2) 
        else: # shape -> shape
            NotImplementedError

        target_data = self.get_item_by_name_mesh(name)
        
        result = {'source_' + k: v for k, v in source_data.items()} 
        result.update({'target_' + k: v for k, v in target_data.items()}) 

        return result

class SketchyVR(Shapes):
    def __init__(self, opt, mode='train', source_type='sketch', cache=True):
        super().__init__(opt, mode, source_type)
        assert source_type=='sketch'
        self.data = {}
        self.cache_dir = '/vol/vssp/datasets/multiview/3VS/datasets/cache'
        self.load_cache(mode, cache)

    def load_cache(self, mode, cache):
        for type in ['sketch', 'shape']:
            cache_path = os.path.join(self.cache_dir, '{}_{}_{}_{}.h5'.format(mode, type, len(self.name_list), self.opt['num_points']))
            if os.path.exists(cache_path) and cache:
                with h5py.File(cache_path, "r") as f:
                    # f = h5py.File(cache_path, 'r+')
                    self.data[type] = f['shape'][:].astype('float32')
            else:
                self.data[type] = None

    def get_sample(self, index):
        name = self.name_list[index]
        if self.data['sketch'] is not None:
            source_data = {'shape': self.data['sketch'][index], 'file': name}
        else:
            source_data = self.get_item_by_name_sketch(name) 
        if self.data['shape'] is not None:
            target_data = {'shape': self.data['shape'][index], 'file': name}
        else:
            target_data = self.get_item_by_name_mesh(name)
        
        result = {'source_' + k: v for k, v in source_data.items()} 
        result.update({'target_' + k: v for k, v in target_data.items()}) 
        result.update({'index': index})
        return result

class SketchyVR_original(SketchyVR):
    def load_cache(self, mode, cache):
        for type in ['sketch', 'shape']:
            cache_path = os.path.join(self.cache_dir, '{}_{}_{}_{}_original.h5'.format(mode, type, len(self.name_list), self.opt['num_points']))
            if os.path.exists(cache_path) and cache:
                with h5py.File(cache_path, "r") as f:
                    # f = h5py.File(cache_path, 'r+')
                    self.data[type] = f['shape'][:].astype('float32')
            else:
                self.data[type] = None
    def _get_mesh_path(self, name):
        return os.path.join(self.opt['data_dir'], 'shape', name + POINT_CLOUD_FILE_EXT)

    def get_item_by_name_mesh(self, name):
        mesh_path = self._get_mesh_path(name)
        points = np.load(mesh_path).astype(np.float32) #[n, 3]
        points = torch.tensor(points)
        points, center, scale = normalize(points, self.normalize_method) #[1024, 3]
        result = {'shape': points, 'file': name}

        return result
    def get_item_by_name_sketch(self, name):
        sketch_path = self._get_sketch_path(name)
        points = np.load(sketch_path).astype(np.float32) #[n, 3]
        points = torch.tensor(points)
        points, center, scale = normalize(points, self.normalize_method) #[1024, 3]
        result = {'shape': points, 'file': name}

        return result

class SketchyVR_original_category(SketchyVR_original):
    def __init__(self, opt, mode='train', source_type='sketch', cache=True):
        assert source_type=='sketch'

        self.mode = mode
        self.opt = opt
        self.normalize_method = 'unit_box'
        self.source_type = source_type

        self.cache_dir = '/vol/vssp/datasets/multiview/3VS/datasets/cache'
        self.data_dir = '/vol/vssp/datasets/multiview/3VS/datasets/ShapeNet'
        name_list = os.path.join(self.data_dir, 'lists', self.opt['category'], 'sketch_{}.txt'.format(mode))
        self.name_list = [line.rstrip() for line in open(name_list)]
        self.data = {}

        self.load_cache(mode, cache)

    def load_cache(self, mode, cache):

        for type in ['sketch', 'shape']:
            if type == 'sketch' and self.opt['category'] in ['02691156', '03636649']:
                cache_path = os.path.join(self.cache_dir, '{}_{}_{}_{}_{}_{}_original.h5'.format(self.opt['category'], self.opt['abstraction'], mode, type, len(self.name_list), self.opt['num_points']))
            else:
                cache_path = os.path.join(self.cache_dir, '{}_{}_{}_{}_{}_original.h5'.format(self.opt['category'], mode, type, len(self.name_list), self.opt['num_points']))

            if os.path.exists(cache_path) and cache:
                with h5py.File(cache_path, "r") as f:
                    # f = h5py.File(cache_path, 'r+')
                    self.data[type] = f['shape'][:].astype('float32')
                    log.info(f'Load data from cache: {cache_path}')
            else:
                self.data[type] = None

    def _get_mesh_path(self, name):
        return os.path.join(self.data_dir, 'original/pointcloud', self.opt['category'], name + POINT_CLOUD_FILE_EXT)

    def _get_sketch_path(self, name):
        return os.path.join(self.data_dir, 'synthetic_sketch/pointcloud', self.opt['category'], name +'_network_20_aggredated_sketch_{}'.format(self.opt['abstraction'])  + POINT_CLOUD_FILE_EXT)


class SketchyVR_single(SketchyVR_original):
    def __init__(self, opt, mode='test', type='sketch', cache=True, sketch_dir= 'aligned_sketch'):
        """
        mode: [train, val, test]
        """
        self.mode = mode
        self.opt = opt
        self.normalize_method = 'unit_box'
        self.type = type
        # self.cache_dir = '/vol/vssp/datasets/multiview/3VS/datasets/cache'
        self.cache_dir = '/vol/research/sketching/projects/VR_Sketch_lightning/project/data/cache'

        if self.type == 'sketch':
            # sketch
            name_list = os.path.join(self.opt['data_dir'], self.opt['{}_list'.format(mode)])
            self.name_list = [line.rstrip() for line in open(name_list)]
        else: 
            # shape
            name_list = os.path.join(self.opt['data_dir'], self.opt['{}_shape_list'.format(mode)])
            self.name_list = [line.rstrip() for line in open(name_list)]
        
        cache_path = os.path.join(self.cache_dir, '{}_{}_{}_{}_original.h5'.format(mode, type, len(self.name_list), self.opt['num_points']))
        # if self.opt['sketch_dir'] != 'aligned_sketch' and self.type == 'sketch':
        if sketch_dir != 'aligned_sketch' and self.type == 'sketch':
            cache_path = os.path.join(self.cache_dir, '{}_{}_{}_{}_new.h5'.format(mode, type, len(self.name_list), self.opt['num_points']))
        if os.path.exists(cache_path) and cache:
            with h5py.File(cache_path, "r") as f:
                self.data = f['shape'][:].astype('float32')
            log.info(f'Load data from cache: {cache_path}')
        else:
            self.data = None

    def __getitem__(self, index):
        name = self.name_list[index]
        if self.data is not None:
            return {'shape': self.data[index], 'file': name}

        if self.type == 'sketch': # sketch
            data = self.get_item_by_name_sketch(name) 
        else: # shape
            data = self.get_item_by_name_mesh(name) 
        return data

class SketchyVR_original_category_single(SketchyVR_single):
    def __init__(self, opt, mode='test', type='sketch', cache=True):
        """
        mode: [train, val, test]
        """
        self.mode = mode
        self.opt = opt
        self.normalize_method = 'unit_box'
        self.type = type
        
        self.data_dir = '/vol/vssp/datasets/multiview/3VS/datasets/ShapeNet'
        self.cache_dir = '/vol/vssp/datasets/multiview/3VS/datasets/cache'

        if self.type == 'shape' and self.mode == 'test':
            # shape test gallery
            name_list = os.path.join(self.data_dir, 'lists', self.opt['category'], self.opt['test_shape_list'])
        else: 
            # shape
            name_list = os.path.join(self.data_dir, 'lists', self.opt['category'], 'sketch_{}.txt'.format(mode))
        
        self.name_list = [line.rstrip() for line in open(name_list)]
        
        if type == 'sketch' and self.opt['category'] in ['02691156', '03636649']:
            cache_path = os.path.join(self.cache_dir, '{}_{}_{}_{}_{}_{}_original.h5'.format(self.opt['category'], self.opt['abstraction'], mode, type, len(self.name_list), self.opt['num_points']))
        else:
            cache_path = os.path.join(self.cache_dir, '{}_{}_{}_{}_{}_original.h5'.format(self.opt['category'], mode, type, len(self.name_list), self.opt['num_points']))

        if os.path.exists(cache_path) and cache:
            with h5py.File(cache_path, "r") as f:
                self.data = f['shape'][:].astype('float32')
            log.info(f'Load data from cache: {cache_path}')
        else:
            self.data = None

    def _get_mesh_path(self, name):
        return os.path.join(self.data_dir, 'original/pointcloud', self.opt['category'], name + POINT_CLOUD_FILE_EXT)

    def _get_sketch_path(self, name):
        return os.path.join(self.data_dir, 'synthetic_sketch/pointcloud', self.opt['category'], name +'_network_20_aggredated_sketch_{}'.format(self.opt['abstraction']) + POINT_CLOUD_FILE_EXT)


class SketchyVR_single_multifold(SketchyVR_original):
    def __init__(self, opt, type='sketch', fold=1, cache=True):
        """
        mode: [train, val, test]
        """
        self.opt = opt
        self.normalize_method = 'unit_box'
        self.type = type
        
        # self.cache_dir = '/vol/vssp/datasets/multiview/3VS/datasets/cache'
        self.cache_dir = '/vol/vssp/datasets/multiview/3VS/datasets/cache /vol/research/sketching/projects/VR_Sketch_lightning/project/data/cache'

        if self.type == 'sketch':
            # sketch
            name_list = os.path.join(self.opt['data_dir'], self.opt['val_list'].format(fold))
            self.name_list = [line.rstrip() for line in open(name_list)]
            cache_path = os.path.join(self.cache_dir, '{}_{}_{}_{}_original.h5'.format(fold, type, len(self.name_list), self.opt['num_points']))

        else: 
            # shape
            name_list = os.path.join(self.opt['data_dir'], 'list/multifold/all.txt')
            self.name_list = [line.rstrip() for line in open(name_list)]
            cache_path = os.path.join(self.cache_dir, 'all_{}_{}_{}_original.h5'.format(type, len(self.name_list), self.opt['num_points']))

        if os.path.exists(cache_path) and cache:
            with h5py.File(cache_path, "r") as f:
                self.data = f['shape'][:].astype('float32')
            log.info(f'Load data from cache: {cache_path}')
        else:
            self.data = None

    def __getitem__(self, index):
        name = self.name_list[index]
        if self.data is not None:
            return {'shape': self.data[index], 'file': name}

        if self.type == 'sketch': # sketch
            data = self.get_item_by_name_sketch(name) 
        else: # shape
            data = self.get_item_by_name_mesh(name) 
        return data


class PointCloudDataLoader(torch.utils.data.Dataset):
    def __init__(self, npoints, list_file, split='train', uniform=False, cache_size=15000, data_dir='', data_type='shape', debug=False, seed=0):

        self.npoints = npoints
        self.uniform = uniform
        self.split = split
        self.eval = self.split in ['test', 'val']
        if self.eval and data_type == 'shape':
            list_file = list_file.replace('.txt', '_shape.txt')
        self.name_list = [line.rstrip() for line in open(os.path.join(data_dir, 'list', list_file.format(split)))]

        self.seed = seed
        self.datapath = []
        self.shape_id = []
        np.random.seed(0)
        for model_name in self.name_list:
            self.shape_id.append(model_name)
            shape_path = os.path.join(data_dir, data_type, model_name + '.npy')
            self.datapath.append(shape_path)

        print('The size of %s data of type %s is %d' % (split, data_type, len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple


    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        if index in self.cache:
            point_set = self.cache[index]
        else:
            file_path = self.datapath[index]
            point_set = np.load(file_path).astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints, fix=self.eval, seed=self.seed)
            else:
                if self.eval:
                    np.random.seed(self.seed)
                farthest = np.random.randint(len(point_set), size=self.npoints)
                point_set = point_set[farthest, :]
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if len(self.cache) < self.cache_size:
                self.cache[index] = point_set

        return point_set, self.datapath[index]

# hydra imports
from omegaconf import DictConfig
import hydra
@hydra.main(config_path="../../configs/", config_name="config.yaml")
def main(config: DictConfig):
    from torchvision import transforms
    import data_utils as d_utils
    # print(config)
    dset = Shapes(config.datamodule)
    sample = dset.get_sample(0)
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)

if __name__ == "__main__":
    main()