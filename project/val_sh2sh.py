
from unicodedata import category
from numpy.lib.npyio import save
from src.datasets.SketchyVRLoader import SketchyVR_single, SketchyVR_single_multifold
from src.test import get_deformer
import torch
import pytorch3d.loss
import numpy as np
from torch.utils.data import DataLoader
from src.utils.distance import f_score, chamfer_single
import os
import matplotlib.pyplot as plt
from src.utils.vis import draw_subplot, draw_txt
from src.utils import utils
import time
from argparse import ArgumentParser

cache_dir = '/vol/vssp/datasets/multiview/3VS/datasets/cache'
K = 10
radius = 0.01

def compute_metrics(metrics, save_dir, shape_dataset, target_shape_dataset, deformer_name='deformer_cage_sh2sh_template', index_list=None):


    batch_size = 10

    dataset_size = shape_dataset.__len__()

    # Load Deformer

    shape_loader = DataLoader(dataset=target_shape_dataset, batch_size=batch_size, \
                    shuffle=False, drop_last=False, collate_fn=shape_dataset.collate, \
                    num_workers=4, pin_memory=False, \
                    worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    deformer = get_deformer(deformer_name)

    if index_list is None:
        index_list = range(dataset_size)

    with torch.no_grad():
        query_num = 0

        for index in index_list:
            tic = time.perf_counter()

            print(index)
        # for index in [0]:
            data = shape_dataset.__getitem__(index)
            shape, name = torch.tensor(data['shape']), data['file']
            gt_shape = shape.cuda().unsqueeze(0).repeat(batch_size, 1, 1) # torch.Size([6, 1024, 3])

            for i, data in enumerate(shape_loader):
                source_shape = data['shape'].cuda()
                B, _, _ = source_shape.shape
                # Before deformation
                cd_dist = pytorch3d.loss.chamfer_distance(source_shape, gt_shape[:B, :, :], batch_reduction=None)[0]
                metrics['CD'].extend(cd_dist.data.cpu().numpy())
                # After deformation
                deformed = deformer(source_shape, gt_shape[:B, :, :])['deformed'] # torch.Size([6, 1024, 3])
                actual_distance = pytorch3d.loss.chamfer_distance(
                    deformed, gt_shape[:B, :, :], batch_reduction=None)[0]
                metrics['CD_d'].extend(actual_distance.data.cpu().numpy())
                # Bi-direction fitting gap
                deformed_2 = deformer(gt_shape[:B, :, :], source_shape)['deformed'] # torch.Size([6, 1024, 3])
                actual_distance_2 = pytorch3d.loss.chamfer_distance(
                    deformed_2, source_shape, batch_reduction=None)[0]
                bi_fitting_gap = (actual_distance + actual_distance_2) / 2
                metrics['bi_CD_d'].extend(bi_fitting_gap.data.cpu().numpy())
                f_score_1 = f_score(deformed, gt_shape[:B, :, :], radius=0.01)
                metrics['F_0.01_d'].extend(f_score_1.data.cpu().numpy())
                f_score_2 = f_score(deformed_2, source_shape, radius=0.01)
                bi_f_score = (f_score_1 + f_score_2) / 2
                metrics['bi_F_0.01_d'].extend(bi_f_score.data.cpu().numpy())

            query_num += 1
            toc = time.perf_counter()
            print(f'Finished in {toc - tic:0.4f} seconds')

        for item in metrics.keys():
            save_path = os.path.join(save_dir, item+'.npy')
            np.save(save_path, np.array(metrics[item]).reshape(query_num, -1))
            print('Save metric for: {}'.format(save_path))

    return metrics

def compute_metrics_category(metrics, save_dir, shape_dataset, target_shape_dataset, index_list=None):


    batch_size = 10

    dataset_size = shape_dataset.__len__()

    # Load Deformer

    shape_loader = DataLoader(dataset=target_shape_dataset, batch_size=batch_size, \
                    shuffle=False, drop_last=False, collate_fn=shape_dataset.collate, \
                    num_workers=4, pin_memory=False, \
                    worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    deformer = get_deformer('deformer_cage_sh2sh_shapenet_template_{}'.format(category))

    if index_list is None:
        index_list = range(dataset_size)

    with torch.no_grad():
        query_num = 0

        for index in index_list:
            tic = time.perf_counter()

            print(index)
        # for index in [0]:
            data = shape_dataset.__getitem__(index)
            shape, name = torch.tensor(data['shape']), data['file']
            gt_shape = shape.cuda().unsqueeze(0).repeat(batch_size, 1, 1) # torch.Size([6, 1024, 3])

            for i, data in enumerate(shape_loader):
                source_shape = data['shape'].cuda()
                B, _, _ = source_shape.shape
                # Before deformation
                cd_dist = pytorch3d.loss.chamfer_distance(source_shape, gt_shape[:B, :, :], batch_reduction=None)[0]
                metrics['CD'].extend(cd_dist.data.cpu().numpy())
                # After deformation
                deformed = deformer(source_shape, gt_shape[:B, :, :])['deformed'] # torch.Size([6, 1024, 3])
                actual_distance = pytorch3d.loss.chamfer_distance(
                    deformed, gt_shape[:B, :, :], batch_reduction=None)[0]
                metrics['CD_d'].extend(actual_distance.data.cpu().numpy())
                # Bi-direction fitting gap
                deformed_2 = deformer(gt_shape[:B, :, :], source_shape)['deformed'] # torch.Size([6, 1024, 3])
                actual_distance_2 = pytorch3d.loss.chamfer_distance(
                    deformed_2, source_shape, batch_reduction=None)[0]
                bi_fitting_gap = (actual_distance + actual_distance_2) / 2
                metrics['bi_CD_d'].extend(bi_fitting_gap.data.cpu().numpy())
                # f_score_1 = f_score(deformed, gt_shape[:B, :, :], radius=0.01)
                # metrics['F_0.01_d'].extend(f_score_1.data.cpu().numpy())
                # f_score_2 = f_score(deformed_2, source_shape, radius=0.01)
                # bi_f_score = (f_score_1 + f_score_2) / 2
                # metrics['bi_F_0.01_d'].extend(bi_f_score.data.cpu().numpy())

            query_num += 1
            toc = time.perf_counter()
            print(f'Finished in {toc - tic:0.4f} seconds')

        for item in metrics.keys():
            save_path = os.path.join(save_dir, '{}_{}.npy'.format(category, item))
            np.save(save_path, np.array(metrics[item]).reshape(query_num, -1))
            print('Save metric for: {}'.format(save_path))

    return metrics
def compute_chamfer(shape_dataset):
    vis_dir = '/scratch/visualization/{}_chamfer'.format(split)
    save_dir = os.path.join(cache_dir, '{}_chamfer'.format(split))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    metrics = {
    'x':[],
    "y":[],
    'xy': []
    }
    batch_size = 10

    dataset_size = shape_dataset.__len__()

    # Load Deformer

    shape_loader = DataLoader(dataset=shape_dataset, batch_size=batch_size, \
                    shuffle=False, drop_last=False, collate_fn=shape_dataset.collate, \
                    num_workers=4, pin_memory=False, \
                    worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    deformer = get_deformer('deformer_cage_sh2sh_template')


    with torch.no_grad():
        query_num = 0

        for index in range(dataset_size):

        # for index in [0, 1]:
            data = shape_dataset.__getitem__(index)
            shape, name = torch.tensor(data['shape']), data['file']
            gt_shape = shape.cuda().unsqueeze(0).repeat(batch_size, 1, 1) # torch.Size([6, 1024, 3])

            for i, data in enumerate(shape_loader):
                source_shape = data['shape'].cuda()
                B, _, _ = source_shape.shape

                # After deformation
                deformed = deformer(source_shape, gt_shape[:B, :, :])['deformed'] # torch.Size([6, 1024, 3])
                actual_distance = pytorch3d.loss.chamfer_distance(
                    deformed, gt_shape[:B, :, :], batch_reduction=None)[0]
                metrics['xy'].extend(actual_distance.data.cpu().numpy())
                cham_x, cham_y = chamfer_single(deformed, gt_shape[:B, :, :], batch_reduction=None)
                metrics['x'].extend(cham_x.data.cpu().numpy())
                metrics['y'].extend(cham_y.data.cpu().numpy())
               
            query_num += 1

        for item in metrics.keys():
            save_path = os.path.join(save_dir, item+'.npy')
            np.save(save_path, np.array(metrics[item]).reshape(query_num, -1))
            print('Save metric for: {}'.format(save_path))
    return metrics, vis_dir, save_dir

def vis_metrics(index, shape, top_shapes, metrics, pair_sort, fn):
    cols = K + 1
    rows = len(metrics.keys())
    fig = plt.figure(figsize=plt.figaspect((rows + .5) / cols))

    # draw_subplot(fig, rows, cols, 1, shape, 'GT', None)
    for line, item in enumerate(metrics.keys()):
        draw_txt(fig, rows, cols, 1 + line * cols, item)

        for i, pc in enumerate(top_shapes[item]):
            id = pair_sort[item][index, i]
            # label = item + ':' + format(metrics[item][index, id] * 100, '.2f')
            label = format(metrics[item][index, id] * 100, '.2f')

            draw_subplot(fig, rows, cols, i+2+cols*line, pc, label, None)

    plt.savefig(
        fn,
        bbox_inches='tight',
        pad_inches=0,
        dpi=300,)
    plt.close()
    print('save file:'+fn)

def vis(shape_dataset, metrics, vis_dir, save_dir):
    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)

    for item in metrics.keys():
        save_path = os.path.join(save_dir, '{}.npy'.format(item))
        metrics[item] = np.load(save_path)
    
        if 'F' in item:
            metrics[item] = 1 - metrics[item]
    pair_sort = {item: np.argsort(metrics[item]) for item in metrics.keys()}

    # Vis top-k by metrics
    dataset_size = shape_dataset.__len__()
    def get_top(tops):
        top_shapes = []
        for id in tops:
            top = torch.tensor(shape_dataset.__getitem__(id)['shape']).unsqueeze(0)
            top_shapes.append(top)
        top_shapes = torch.cat(top_shapes, 0)
        return top_shapes

    for index in range(dataset_size):
    # for index in [0]:

        data = shape_dataset.__getitem__(index)
        shape, name = torch.tensor(data['shape']), data['file']
        top_shapes = {item:[] for item in metrics.keys()}
        for item in metrics.keys():
            tops = pair_sort[item][index, :K]
            top_shapes[item] = get_top(tops)
        fn = os.path.join(vis_dir, '{}_{}.png'.format(index, name))
        vis_metrics(index, shape, top_shapes, metrics, pair_sort, fn)

def compute_val():
    # Prepare data
    datamodule = {
        'data_dir': '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch',
        'num_points': 1024,
        'val_list': 'list/multifold/all.txt',
        'val_shape_list': 'list/multifold/all.txt',
        'test_list': 'list/hs/test.txt',
        'test_shape_list': 'list/hs/test.txt',
        'sketch_dir': 'aligned_sketch'
    }
    split = 'val'

    shape_dataset = SketchyVR_single_multifold(datamodule, type='shape', fold=1)
    vis_dir = '/vol/vssp/datasets/multiview/3VS/visualization/{}_metrics'.format(split)
    # save_dir = os.path.join(cache_dir, '{}_metrics'.format(split))
    save_dir = os.path.join(cache_dir, 'multifold')
    
    metrics = {
    'CD':[],
    'CD_d':[],
    'bi_CD_d':[],
    }

    metrics, save_dir = compute_metrics(metrics, save_dir, shape_dataset, shape_dataset)
    # metrics, vis_dir, save_dir = compute_chamfer(shape_dataset)

    # vis(shape_dataset, metrics, vis_dir, save_dir)

def compute_val_category(category):
    # Prepare data
    from src.datasets.SketchyVRLoader import SketchyVR_original_category_single
    datamodule = {
        'data_dir': '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch',
        'num_points': 1024,
        'category': category
    }
    split = 'val'

    shape_dataset = SketchyVR_original_category_single(datamodule, mode=split, type='shape', cache=False)

    # vis_dir = '/vol/vssp/datasets/multiview/3VS/visualization/{}_metrics'.format(split)
    # save_dir = os.path.join(cache_dir, '{}_metrics'.format(split))
    save_dir = os.path.join(cache_dir, 'category')
    
    metrics = {
    'CD':[],
    'CD_d':[],
    'bi_CD_d':[],
    }

    metrics = compute_metrics_category(metrics, save_dir, shape_dataset, shape_dataset)
    # metrics, vis_dir, save_dir = compute_chamfer(shape_dataset)

    # vis(shape_dataset, metrics, vis_dir, save_dir)

def compute_test_category(category):
    # Prepare data
    split = 'test'
    from src.datasets.SketchyVRLoader import SketchyVR_original_category_single
    datamodule = {
        'data_dir': '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch',
        'num_points': 1024,
        'test_shape_list': 'sketch_test.txt',
        'category': category
    }
    shape_dataset = SketchyVR_original_category_single(datamodule, mode=split, type='shape')
    target_datamodule = {
        'data_dir': '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch',
        'num_points': 1024,
        'test_shape_list': 'sketch_test_shape.txt',
        'category': category
    }
    save_dir = os.path.join(cache_dir, '{}_selected_metrics'.format(split), category)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    deformer = 'deformer_cage_sh2sh_shapenet_template_{}'.format(category)

    metrics = {
    'CD':[],
    'CD_d':[],
    'bi_CD_d':[],
    # "F_0.02":[],
    "F_0.01_d":[],
    "bi_F_0.01_d":[],
    }

    target_shape_dataset = SketchyVR_original_category_single(target_datamodule, mode=split, type='shape')

    metrics = compute_metrics(metrics, save_dir, shape_dataset, target_shape_dataset, deformer_name=deformer)
    # vis(shape_dataset, metrics, vis_dir, save_dir)

def compute_selected():
    # Prepare data
    datamodule = {
        'data_dir': '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch',
        'num_points': 1024,
        'val_list': 'list/hs/val.txt',
        'val_shape_list': 'list/hs/val.txt',
        'test_list': 'list/hs/test.txt',
        'test_shape_list': 'list/hs/test.txt',

    }
    shape_dataset = SketchyVR_single(datamodule, mode=split, type='shape')
    target_datamodule = {
        'data_dir': '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch',
        'num_points': 1024,
        'val_list': 'list/hs/val.txt',
        'val_shape_list': 'list/hs/val.txt',
        'test_list': 'list/hs/test.txt',
        'test_shape_list': 'list/hs/test_shape.txt',

    }
    vis_dir = '/scratch/visualization/{}_selected_metrics'.format(split)
    save_dir = os.path.join(cache_dir, '{}_selected_metrics'.format(split))
    deformer = 'deformer_cage_sh2sh_template'
    model_dir = '/vol/research/sketching/projects/VR_Sketch_lightning/project/logs/experiments/{}/inference'.format(deformer)

    metrics = {
    'CD':[],
    'CD_d':[],
    'bi_CD_d':[],
    # "F_0.02":[],
    "F_0.01_d":[],
    # "bi_F_0.01_d":[],
    }

    log = utils.logger_config(os.path.join(model_dir, 'run.log'), __name__)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # index_list = [8, 17, 57, 68, 89, 91, 95, 114, 127, 166, 172, 182, 201]
    target_shape_dataset = SketchyVR_single(target_datamodule, mode=split, type='shape')

    # save_dir = compute_metrics(metrics, save_dir, shape_dataset, target_shape_dataset)
    # vis(shape_dataset, metrics, vis_dir, save_dir)
    # from val_sk2sh import evaluate
    # evaluate(target_shape_dataset, metrics, save_dir, log)

if __name__ == "__main__":
    parser = ArgumentParser(description='Compute and save fitting gap values.')
    parser.add_argument('--category', type=str, default='',
                        help='category ID of ShapeNet')
    parser.add_argument('--model_dir', type=str, default='./logs/deformer',
                        help='Model checkpoint directory')
    args = parser.parse_args()

    for category in ['02691156']:
        # Compute fitting gap for validation set
        compute_val_category(args.category)
        # Compute fitting gap for test set
        compute_test_category(args.category)