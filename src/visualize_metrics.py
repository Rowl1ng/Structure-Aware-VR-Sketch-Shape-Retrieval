from functools import reduce
from src.datasets.SketchyVRLoader import SketchyVR_single
import omegaconf
import hydra
from pytorch_lightning.utilities.cloud_io import load as pl_load
import os
from src.utils import template_utils as utils
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
import torch
import numpy as np
import pytorch3d.loss

from src.utils.vis import draw_subplot
import matplotlib.pyplot as plt
from src.utils.distance import f_score
from src.test import get_deformer

save_dir = '/vol/vssp/datasets/multiview/3VS/visualization'

def get_label(metrics, i):
    label = []
    for k, v in metrics.items():
        val = k + ':' + format(v[i] * 100, '.2f')
        # val = format(v[i] * 100, '.2f')
        label.append(val)

    return '\n'.join(label)



def vis_metrics(query, target, source, deformed, metrics_before, metrics_after, fn, gt_KP, source_KP, deformed_KP):
    cols = len(source) + 1
    rows = 4
    fig = plt.figure(figsize=plt.figaspect((rows + .5) / cols))
    draw_subplot(fig, rows, cols, 1, target, 'GT', gt_KP)
    for i, pc in enumerate(source):
        label = get_label(metrics_before, i)
        draw_subplot(fig, rows, cols, i+2, pc, '', source_KP[i])
        draw_txt(fig, rows, cols, i+2+cols, label)
    draw_subplot(fig, rows, cols, 1+cols*2, query, 'Query', None)

    for i, pc in enumerate(deformed):
        label = get_label(metrics_after, i)
        draw_subplot(fig, rows, cols, i+cols*2+2, pc, '', deformed_KP[i])
        draw_txt(fig, rows, cols, i+cols*3+2, label)

    plt.savefig(
        fn,
        bbox_inches='tight',
        pad_inches=0,
        dpi=300,)
    plt.close()
    print('save file:'+fn)

def shape_to_shape(shape_dataset, sketch_dataset):


    dataset_size = shape_dataset.__len__()

    deformer = get_deformer('deformer_cage_sh2sh_template')
    # Prepare model retrieved results
    model_name = '3dv21_margin_0.6_step_size_80'
    exp_dir = '/vol/research/sketching/projects/VR_Sketch_lightning/project/logs/experiments/{}'.format(model_name)
    save_path = os.path.join(exp_dir, 'inference', '{}_rank_{}.npy'.format(split, dataset_size))

    dis_mat = np.load(save_path)   
    pair_sort = np.argsort(dis_mat)
    num_queries = pair_sort.shape[0]

    source_metrics = {
    }
    deformed_metrics = {
    }
    from src.utils.custom_loss import ordered_l2
    KP_dist_fun = ordered_l2()
    K = 10
    radius = 0.01
    with torch.no_grad():
        for index in range(num_queries):
        # for index in range(10):
            tops = pair_sort[index, :K]
            sketch = torch.tensor(sketch_dataset.__getitem__(index)['shape'])
            data = shape_dataset.__getitem__(index)
            shape, name = torch.tensor(data['shape']), data['file']
            gt_shapes = shape.unsqueeze(0).repeat(K, 1, 1).cuda()
            top_shapes = []
            for item in tops:
                item = torch.tensor(shape_dataset.__getitem__(item)['shape']).unsqueeze(0)
                top_shapes.append(item)
            top_shapes = torch.cat(top_shapes, 0).cuda()
            
            # Before deformation
            top_dist = pytorch3d.loss.chamfer_distance(gt_shapes, top_shapes, batch_reduction=None)[0]
            source_metrics['CD'] = top_dist.data.cpu().numpy()
            source_metrics['F'] = f_score(gt_shapes, top_shapes, radius=radius).data.cpu().numpy()

            # After deformation

            output = deformer(top_shapes, gt_shapes)
            deformed_shape = output['deformed'] # torch.Size([6, 1024, 3])
            actual_distance = pytorch3d.loss.chamfer_distance(
                deformed_shape, gt_shapes, batch_reduction=None)[0]
            deformed_metrics['CD'] = actual_distance.data.cpu().numpy()
            deformed_metrics['F'] = f_score(deformed_shape, top_shapes, radius=radius).data.cpu().numpy()
            ## KP distance
            source_metrics['KP'] = KP_dist_fun(output["target_keypoints"], output["source_keypoints"], reduce=False)
            deformed_metrics['KP'] = KP_dist_fun(output["target_keypoints"], output["deformed_keypoints"], reduce=False)

            # Vis deformed 
            fn = os.path.join(save_dir, '3dv_retrieved', '{}_{}.png'.format(index, name))
            vis_metrics(sketch, shape, top_shapes.cpu() , deformed_shape.cpu() , source_metrics, deformed_metrics, fn, \
                output["target_keypoints"][0].cpu(), output["source_keypoints"].cpu(), output["deformed_keypoints"].cpu())

if __name__ == "__main__":
    seed_everything('12345', workers=True)
    # Prepare data
    split = 'test'
    datamodule = {
        'data_dir': '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch',
        'num_points': 1024,
        'test_list': 'list/hs/test.txt',
        'test_shape_list': 'list/hs/test_shape.txt',
        'mesh_dir': '/vol/vssp/datasets/multiview/3VS/datasets/ShapeNet/shapenet',
        'category': '03001627'
    }
    shape_dataset = SketchyVR_single(datamodule, mode=split, type='shape')
    sketch_dataset = SketchyVR_single(datamodule, mode=split, type='sketch')

    shape_to_shape(shape_dataset, sketch_dataset)
