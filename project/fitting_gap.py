import omegaconf
import hydra
from pytorch_lightning.utilities.cloud_io import load as pl_load
import os
from src.utils import utils
from pytorch_lightning import (
    LightningModule,
    seed_everything,
)
from src.datasets.SketchyVRLoader import SketchyVR_single
import numpy as np
from src.utils.point_cloud_utils import normalize_to_box, sample_farthest_points
import torch
# template utils imports
from src.utils import template_utils as utils
from src.test import get_deformer
from src.utils.vis import draw_subplot, rotate_point_cloud
import matplotlib.pyplot as plt
save_dir = '/vol/vssp/datasets/multiview/3VS/datasets/cache/train_fitting_gap'
cache_dir = '/vol/vssp/datasets/multiview/3VS/datasets/cache'



def vis(pcs, fn):
    num = len(pcs)
    fig = plt.figure(figsize=plt.figaspect(1 / num))
    for index, pc in enumerate(pcs):
        pc = rotate_point_cloud(pc, dim='x')
        pc = rotate_point_cloud(pc, dim='z', angle=90)

        ax = fig.add_subplot(1, num, index+1, projection='3d')
        ax.scatter(pc[:, 0],pc[:, 1], pc[:, 2], c=pc[:, 2])
    plt.savefig(
        fn,
        bbox_inches='tight',
        pad_inches=0,
        dpi=300,)
    plt.close()


def vis_fitting_gap(target, source, deformed, cd_dist_before, cd_dist_after, fn):
    num = len(source)
    fig = plt.figure(figsize=plt.figaspect(2.5 / (num+1)))
    draw_subplot(fig, 2, num+1, 1, target, 'GT', None)
    for i, pc in enumerate(source):
        draw_subplot(fig, 2, num+1, i+2, pc, format(cd_dist_before[i] * 100, '.2f'), None)
    for i, pc in enumerate(deformed):
        draw_subplot(fig, 2, num+1, i+num+3, pc, format(cd_dist_after[i] * 100, '.2f'), None)

    plt.savefig(
        fn,
        bbox_inches='tight',
        pad_inches=0,
        dpi=300,)
    plt.close()

def compute_fitting_gap(shape_dataset, sample_num):
    import pytorch3d.loss
    dataset_size = shape_dataset.__len__()
    from torch.utils.data import DataLoader

    batch_size = 10
    # Load Deformer

    shape_loader = DataLoader(dataset=shape_dataset, batch_size=batch_size, \
                    shuffle=True, drop_last=False, collate_fn=shape_dataset.collate, \
                    num_workers=4, pin_memory=False, \
                    worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    deformer = get_deformer('deformer_cage_sh2sh_template')

    selected_list = np.random.choice(dataset_size, 30, replace=False)
    with torch.no_grad():
        for index in range(dataset_size):
            data = shape_dataset.__getitem__(index)
            shape, name = torch.tensor(data['shape']), data['file']
            GT_shape = shape.cuda().unsqueeze(0).repeat(batch_size, 1, 1) # torch.Size([6, 1024, 3])
            fitting_gap = []
            for i, data in enumerate(shape_loader):
                source_shape = data['shape'].cuda()
                B, _, _ = source_shape.shape
                deformed = deformer(source_shape, GT_shape[:B, :, :])['deformed'] # torch.Size([6, 1024, 3])
                cd_dist = pytorch3d.loss.chamfer_distance(deformed, GT_shape[:B, :, :], batch_reduction=None)[0]
                cd_dist_after = cd_dist.data.cpu().numpy()
                fitting_gap.extend(cd_dist)

                if i == 0 and index in selected_list:
                    # before deform
                    cd_dist = pytorch3d.loss.chamfer_distance(source_shape, GT_shape, batch_reduction=None)[0]
                    cd_dist_before = cd_dist.data.cpu().numpy()

                    # Vis deformed 
                    fn = os.path.join(save_dir, '{}_{}.png'.format(name, i))
                    vis_fitting_gap(shape, source_shape.cpu(), deformed.cpu(), cd_dist_before, cd_dist_after, fn)
                if len(fitting_gap) >= sample_num:
                    save_path = os.path.join(save_dir, name+'.npy')
                    np.save(save_path, fitting_gap)
                    utils.log_string('Save fitting gap for: {}'.format(index))

                    break

def compute_fitting_gap_category(shape_dataset, sample_num):
    import pytorch3d.loss
    dataset_size = shape_dataset.__len__()
    from torch.utils.data import DataLoader

    batch_size = 10
    # Load Deformer

    shape_loader = DataLoader(dataset=shape_dataset, batch_size=batch_size, \
                    shuffle=True, drop_last=False, collate_fn=shape_dataset.collate, \
                    num_workers=4, pin_memory=False, \
                    worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    deformer = get_deformer('deformer_cage_sh2sh_template')

    selected_list = np.random.choice(dataset_size, 30, replace=False)
    with torch.no_grad():
        for index in range(dataset_size):
            data = shape_dataset.__getitem__(index)
            shape, name = torch.tensor(data['shape']), data['file']
            GT_shape = shape.cuda().unsqueeze(0).repeat(batch_size, 1, 1) # torch.Size([6, 1024, 3])
            fitting_gap = []
            for i, data in enumerate(shape_loader):
                source_shape = data['shape'].cuda()
                B, _, _ = source_shape.shape
                deformed = deformer(source_shape, GT_shape[:B, :, :])['deformed'] # torch.Size([6, 1024, 3])
                cd_dist = pytorch3d.loss.chamfer_distance(deformed, GT_shape[:B, :, :], batch_reduction=None)[0]
                cd_dist_after = cd_dist.data.cpu().numpy()
                fitting_gap.extend(cd_dist)

                if i == 0 and index in selected_list:
                    # before deform
                    cd_dist = pytorch3d.loss.chamfer_distance(source_shape, GT_shape, batch_reduction=None)[0]
                    cd_dist_before = cd_dist.data.cpu().numpy()

                    # Vis deformed 
                    fn = os.path.join(save_dir, '{}_{}.png'.format(name, i))
                    vis_fitting_gap(shape, source_shape.cpu(), deformed.cpu(), cd_dist_before, cd_dist_after, fn)
                if len(fitting_gap) >= sample_num:
                    save_path = os.path.join(save_dir, name+'.npy')
                    np.save(save_path, fitting_gap)
                    utils.log_string('Save fitting gap for: {}'.format(index))

                    break

def comupte_sigma(shape_dataset):
    sigma_list = []
    for name in shape_dataset.name_list:
        save_path = os.path.join(save_dir, name+'.npy')
        dists = np.load(save_path, allow_pickle=True)
        max_val = dists.max()
        sigma_list.append(max_val.cpu().numpy())
    save_path = os.path.join(cache_dir, 'sigmas.npy')
    np.save(save_path, sigma_list)
    utils.log_string('Save fitting gap for {} shapes'.format(len(sigma_list)))

def comupte_top2(shape_dataset):
    sigma_list = []
    for name in shape_dataset.name_list:
        save_path = os.path.join(save_dir, name+'.npy')
        dists = np.load(save_path, allow_pickle=True)
        dists = np.sort(dists)[1:]
        max_val = np.min(dists) # [np.where(dists > 1e-7)]
        sigma_list.append(max_val.cpu().numpy())
    save_path = os.path.join(cache_dir, 'top2.npy')
    np.save(save_path, sigma_list)
    utils.log_string('Save fitting gap for {} shapes'.format(len(sigma_list)))


def compute_train():
    # Prepare data
    split = 'train'
    datamodule = {
        'data_dir': '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch',
        'num_points': 1024,
        'train_list': 'list/hs/train.txt',
        'train_shape_list': 'list/hs/train.txt',
    }
    shape_dataset = SketchyVR_single(datamodule, mode=split, type='shape')
    # sketch_dataset = SketchyVR_single(datamodule, mode=split, type='sketch')

    # compute_fitting_gap(shape_dataset, sample_num=200)
    # comupte_sigma(shape_dataset)
    # comupte_top2(shape_dataset)

if __name__ == "__main__":
    seed_everything('12345', workers=True)

    compute_train()
