import numpy as np
import h5py, os
from pytorch_lightning import (
    seed_everything,
)
from torch.utils.data import DataLoader
from src.utils.point_cloud_utils import normalize_to_box, sample_farthest_points
import torch
import omegaconf
# path = '/vol/research/sketching/projects/VR_Sketch_lightning/project/logs/experiments/Chamfer/.hydra/config.yaml'
# config = omegaconf.OmegaConf.load(path)
# if config.get("seed"):
seed_everything('12345', workers=True)

from src.datasets.SketchyVRLoader import SketchyVR_single, SketchyVR_single_multifold
num_points = 1024
batch_size = 10
out_dir = '/vol/vssp/datasets/multiview/3VS/datasets/cache'


def save_multifold():
    # type = 'shape'
    # datamodule = {
    # 'data_dir': '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch',
    # 'num_points': 1024,
    # 'val_shape_list': 'list/multifold/fold_{}_val.txt',
    # 'sketch_dir': 'aligned_sketch'
    # }
    # sketch_dataset = SketchyVR_single_multifold(datamodule, type=type, fold=1, cache=False)
    # sketch_loader = DataLoader(dataset=sketch_dataset, batch_size=batch_size, \
    #                 shuffle=False, drop_last=False, collate_fn=sketch_dataset.collate, \
    #                 num_workers=4, pin_memory=False, \
    #                 worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    # sketch_data = []
    # file_data = []

    # for i, data in enumerate(sketch_loader):
    #     sketch = data['shape'].cuda()
    #     sketch = sample_farthest_points(sketch.transpose(1, 2), num_points).transpose(1,2) # torch.Size([1, 1024, 3])
    #     sketch_data.append(sketch)
    #     file_data.extend(data['file'])

    # sketch_data = torch.cat(sketch_data, dim=0).data.cpu().numpy()
    # out_h5_file = os.path.join(out_dir, 'all_{}_{}_{}_original.h5'.format(type, len(sketch_dataset), num_points))

    # with h5py.File(out_h5_file, 'w') as f:
    #     f.create_dataset('shape', data=sketch_data, compression="gzip")
    #     f.create_dataset('file', data=file_data, compression="gzip")
    # print("Saved '{}'.".format(out_h5_file))

    # quit()

    type = 'sketch'
    for fold in [2,3,4,5]:
        datamodule = {
        'data_dir': '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch',
        'num_points': 1024,
        'val_list': 'list/multifold/fold_{}_val.txt',
        'sketch_dir': 'aligned_sketch'
        }
        sketch_dataset = SketchyVR_single_multifold(datamodule, type=type, fold=fold, cache=False)
        sketch_loader = DataLoader(dataset=sketch_dataset, batch_size=batch_size, \
                        shuffle=False, drop_last=False, collate_fn=sketch_dataset.collate, \
                        num_workers=4, pin_memory=False, \
                        worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

        sketch_data = []
        file_data = []

        for i, data in enumerate(sketch_loader):
            sketch = data['shape'].cuda()
            sketch = sample_farthest_points(sketch.transpose(1, 2), num_points).transpose(1,2) # torch.Size([1, 1024, 3])
            sketch_data.append(sketch)
            file_data.extend(data['file'])

        sketch_data = torch.cat(sketch_data, dim=0).data.cpu().numpy()

        out_h5_file = os.path.join(out_dir, '{}_{}_{}_{}_original.h5'.format(fold, type, len(sketch_dataset), num_points))

        with h5py.File(out_h5_file, 'w') as f:
            f.create_dataset('shape', data=sketch_data, compression="gzip")
            f.create_dataset('file', data=file_data, compression="gzip")
        print("Saved '{}'.".format(out_h5_file))

def save_category(split, type, category, abstraction='0.25'):
    
    from src.datasets.SketchyVRLoader import SketchyVR_original_category_single
    datamodule = {
        'data_dir': '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch',
        'num_points': 1024,
        'test_shape_list': 'sketch_test.txt',
        'category': category,
        'abstraction': abstraction
    }

    shape_dataset = SketchyVR_original_category_single(datamodule, mode=split, type=type, cache=False)

    # vis_dir = '/vol/vssp/datasets/multiview/3VS/visualization/{}_metrics'.format(split)
    # save_dir = os.path.join(cache_dir, '{}_metrics'.format(split))

    sketch_data = []
    file_data = []

    shape_loader = DataLoader(dataset=shape_dataset, batch_size=batch_size, \
                    shuffle=False, drop_last=False, collate_fn=shape_dataset.collate, \
                    num_workers=4, pin_memory=False, \
                    worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    for i, data in enumerate(shape_loader):
        sketch = data['shape'].cuda()
        sketch = sample_farthest_points(sketch.transpose(1, 2), num_points).transpose(1,2) # torch.Size([1, 1024, 3])
        sketch_data.append(sketch)
        file_data.extend(data['file'])

    sketch_data = torch.cat(sketch_data, dim=0).data.cpu().numpy()
    if type == 'sketch' and category in ['02691156', '03636649']:
        out_h5_file = os.path.join(out_dir, '{}_{}_{}_{}_{}_{}_original.h5'.format(category, abstraction, split, type, len(shape_dataset), num_points))
    else:
        out_h5_file = os.path.join(out_dir, '{}_{}_{}_{}_{}_original.h5'.format(category, split, type, len(shape_dataset), num_points))

    with h5py.File(out_h5_file, 'w') as f:
        f.create_dataset('shape', data=sketch_data, compression="gzip")
        # f.create_dataset('file', data=file_data, compression="gzip")
    print("Saved '{}'.".format(out_h5_file))

def save_cache(type):
    datamodule = {
    'data_dir': '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch',
    'num_points': 1024,
    'train_list': 'list/hs/train.txt',
    'train_shape_list': 'list/hs/train.txt',
    'test_shape_list': 'list/hs/test.txt',
    'test_list': 'list/hs/test.txt',
    'mesh_dir': '/vol/vssp/datasets/multiview/3VS/datasets/ShapeNet/shapenet',
    'category': '03001627',
    'sketch_dir': 'sketch_21_test'
}

    sketch_dataset = SketchyVR_single(datamodule, mode=split, type=type, cache=False)
    sketch_loader = DataLoader(dataset=sketch_dataset, batch_size=batch_size, \
                    shuffle=False, drop_last=False, collate_fn=sketch_dataset.collate, \
                    num_workers=4, pin_memory=False, \
                    worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    sketch_data = []
    file_data = []

    for i, data in enumerate(sketch_loader):
        sketch = data['shape'].cuda()
        sketch = sample_farthest_points(sketch.transpose(1, 2), num_points).transpose(1,2) # torch.Size([1, 1024, 3])
        sketch_data.append(sketch)
        file_data.extend(data['file'])

    sketch_data = torch.cat(sketch_data, dim=0).data.cpu().numpy()

    out_h5_file = os.path.join(out_dir, '{}_{}_{}_{}_original.h5'.format(split, type, len(sketch_dataset), num_points))
    if datamodule['sketch_dir'] != 'aligned_sketch' and type == 'sketch':
        out_h5_file = os.path.join(out_dir, '{}_{}_{}_{}_new.h5'.format(split, type, len(sketch_dataset), num_points))

    with h5py.File(out_h5_file, 'w') as f:
        f.create_dataset('shape', data=sketch_data, compression="gzip")
        f.create_dataset('file', data=file_data, compression="gzip")
    print("Saved '{}'.".format(out_h5_file))
    
    return out_h5_file



# h5_name = save_sketch()
# h5_name = save_cache('sketch')
# save_multifold()
for split in ['test']:
    for category in  ['02691156']:
        # for abstraction in ['0.0', '0.25', '0.5', '0.75', '1.0']:
        #     save_category(split, 'sketch', category, abstraction=abstraction)
        save_category(split, 'shape', category, abstraction='0.0')

# h5_name = '/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ll00931/dataset/h5/test_sketch_202_2048.h5'
# f = h5py.File(h5_name, 'r+')
# data_1 = f['shape'][:].astype('float32')

# h5_name = '/vol/research/NOBACKUP/CVSSP/scratch_4weeks/ll00931/dataset/h5/test_shape_202_2048.h5'
# f = h5py.File(h5_name, 'r+')
# data_2 = f['shape'][:].astype('float32')

pass
# print(data.shape)
# TODO: save shapes, sample_points_from_meshes

