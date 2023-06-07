
from numpy.lib.npyio import save
from src.datasets.SketchyVRLoader import SketchyVR_single
from src.test import get_deformer
import torch
import pytorch3d.loss
import numpy as np
from torch.utils.data import DataLoader
from src.utils.distance import f_score, chamfer_single
import os
import matplotlib.pyplot as plt
from src.utils.vis import draw_subplot
from src.utils.custom_loss import ordered_l2

from src.utils import utils
import os
import logging



cache_dir = '/vol/vssp/datasets/multiview/3VS/datasets/cache'
K = 10
split = 'test'

def compute_metrics_old(shape_dataset):
    vis_dir = '/vol/vssp/datasets/multiview/3VS/visualization/{}_metrics'.format(split)
    save_dir = os.path.join(cache_dir, '{}_metrics'.format(split))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    metrics = {
    'CD':[],
    "F":[],
    'CD_d':[],
    "F_d":[]

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
                # Before deformation
                cd_dist = pytorch3d.loss.chamfer_distance(source_shape, gt_shape[:B, :, :], batch_reduction=None)[0]
                metrics['CD'].extend(cd_dist.data.cpu().numpy())
                metrics['F'].extend(f_score(gt_shape[:B, :, :], source_shape, radius=radius).data.cpu().numpy())

                # After deformation
                deformed = deformer(source_shape, gt_shape[:B, :, :])['deformed'] # torch.Size([6, 1024, 3])
                actual_distance = pytorch3d.loss.chamfer_distance(
                    deformed, gt_shape[:B, :, :], batch_reduction=None)[0]
                metrics['CD_d'].extend(actual_distance.data.cpu().numpy())
                metrics['F_d'].extend(f_score(deformed, gt_shape[:B, :, :], radius=radius).data.cpu().numpy())
            query_num += 1

        for item in metrics.keys():
            save_path = os.path.join(save_dir, item+'.npy')
            np.save(save_path, np.array(metrics[item]).reshape(query_num, -1))
            logging.info('Save metric for: {}'.format(save_path))

    return metrics, vis_dir, save_dir

def compute_chamfer(sketch_dataset, shape_dataset, metrics, save_dir, args, log):
    batch_size = 10
    radius = args.radius
    deformer = args.deformer
    dataset_size = sketch_dataset.__len__()

    # Load Deformer

    shape_loader = DataLoader(dataset=shape_dataset, batch_size=batch_size, \
                    shuffle=False, drop_last=False, collate_fn=shape_dataset.collate, \
                    num_workers=4, pin_memory=False, \
                    worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    deformer = get_deformer(args.deformer)

    KP_dist_fun = ordered_l2()

    with torch.no_grad():
        query_num = 0

        for index in range(dataset_size):

        # for index in [0, 1]:
            data = sketch_dataset.__getitem__(index)
            shape, name = torch.tensor(data['shape']), data['file']
            gt_shape = shape.cuda().unsqueeze(0).repeat(batch_size, 1, 1) # torch.Size([6, 1024, 3])

            for i, data in enumerate(shape_loader):
                target_shape = data['shape'].cuda()
                B, _, _ = target_shape.shape
                # Before deformation
                top_dist = pytorch3d.loss.chamfer_distance(gt_shape[:B, :, :], target_shape, batch_reduction=None)[0]
                if 'CD' in metrics.keys():
                    metrics['CD'].extend(top_dist.data.cpu().numpy())
                # if 'F' in metrics.keys():
                #     metrics['F'].extend(f_score(gt_shape[:B, :, :], source_shape, radius=radius).data.cpu().numpy())


                # After deformation
                output = deformer(gt_shape[:B, :, :], target_shape) # deform: sketch->shape
                deformed = output['deformed'] # torch.Size([6, 1024, 3])
                actual_distance = pytorch3d.loss.chamfer_distance(
                    deformed, target_shape, batch_reduction=None)[0]
                if 'CD_d' in metrics.keys():
                    metrics['CD_d'].extend(actual_distance.data.cpu().numpy())
                if 'F_d' in metrics.keys():
                    metrics['F_d'].extend(f_score(deformed, target_shape, radius=radius).data.cpu().numpy())
                # if 'KP' in metrics.keys():
                #     metrics['KP'].extend(KP_dist_fun(output["target_keypoints"], output["source_keypoints"], reduce=False))
                # if 'KP_d' in metrics.keys():
                #     metrics['KP_d'].extend(KP_dist_fun(output["target_keypoints"], output["deformed_keypoints"], reduce=False))

            query_num += 1

        for item in metrics.keys():
            save_path = os.path.join(save_dir, item+'.npy')
            np.save(save_path, np.array(metrics[item]).reshape(query_num, -1))
            log.info('Save metric for: {}'.format(save_path))
    return metrics

def vis_metrics(index, shape, top_shapes, metrics, pair_sort, fn):
    cols = K + 1
    rows = len(metrics.keys())
    fig = plt.figure(figsize=plt.figaspect((rows + .5) / cols))

    draw_subplot(fig, rows, cols, 1, shape, 'GT', None)
    for line, item in enumerate(metrics.keys()):
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

def vis(sketch_dataset, shape_dataset, metrics, save_dir, args):
    vis_dir = '/vol/research/sketching/projects/VR_Sketch_lightning/project/logs/experiments/{}/inference/vis'.format(args.deformer)
    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)

    for item in metrics.keys():
        save_path = os.path.join(save_dir, '{}.npy'.format(item))
        metrics[item] = np.load(save_path, allow_pickle=True)
    if 'F' in metrics.keys():
        metrics['F'] = 1 - metrics['F']
    if 'F_d' in metrics.keys():
        metrics['F_d'] = 1 - metrics['F_d']
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

        data = sketch_dataset.__getitem__(index)
        shape, name = torch.tensor(data['shape']), data['file']
        top_shapes = {item:[] for item in metrics.keys()}
        for item in metrics.keys():
            tops = pair_sort[item][index, :K]
            top_shapes[item] = get_top(tops)
        fn = os.path.join(vis_dir, '{}_{}.png'.format(index, name))
        vis_metrics(index, shape, top_shapes, metrics, pair_sort, fn)

def evaluate(shape_dataset, metrics, save_dir, log):
    from src.test import compute_metrics
    for item in metrics.keys():
        save_path = os.path.join(save_dir, '{}.npy'.format(item))
        metrics[item] = np.load(save_path, allow_pickle=True)
        log.info('**************' + item + '**************')
        if 'F' in item:
            metrics[item] = 1 - metrics[item]
        
        dist_list = compute_metrics(shape_dataset, metrics[item], None, log)


def compute_val(args, log):
    # Prepare data
    datamodule = {
        'data_dir': '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch',
        'num_points': 1024,
        'val_list': 'list/hs/val.txt',
        'val_shape_list': 'list/hs/val.txt',
        'test_list': 'list/hs/test.txt',
        'test_shape_list': 'list/hs/test_shape.txt',

    }
    shape_dataset = SketchyVR_single(datamodule, mode=split, type='shape')
    sketch_dataset = SketchyVR_single(datamodule, mode=split, type='sketch')
    save_dir = '/vol/research/sketching/projects/VR_Sketch_lightning/project/logs/experiments/{}/inference'.format(args.deformer)


    metrics = {
    'CD':[],
    'CD_d': [],
    'F_d':[],

    # 'KP_d':[]
    }
    metrics = compute_chamfer(sketch_dataset, shape_dataset, metrics, save_dir, args, log)

    # vis(sketch_dataset, shape_dataset, metrics, save_dir, args)
    evaluate(shape_dataset, metrics, save_dir, log)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--deformer', type=str, default='deformer_cage_sk2sh_template', 
                        help='deformer')
    parser.add_argument('--radius', type=float, default=0.01, 
                        help='deformer')

    args = parser.parse_args()
    save_dir = '/vol/research/sketching/projects/VR_Sketch_lightning/project/logs/experiments/{}/inference'.format(args.deformer)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    log = utils.logger_config(os.path.join(save_dir, 'run.log'), __name__)

    compute_val(args, log)
