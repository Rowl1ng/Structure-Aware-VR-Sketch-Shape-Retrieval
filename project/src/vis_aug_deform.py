from src.datasets.SketchyVRLoader import SketchyVR_original
import src.datamodules.data_utils as d_utils
from torch.utils.data import DataLoader
import numpy as np
from src.utils.point_cloud_utils import normalize_to_box, sample_farthest_points
from torchvision.transforms import transforms
from src.test import get_deformer
import torch
import pytorch3d
from pytorch_lightning import (
    LightningModule,
    seed_everything,
)
import os
import cv2
batch_size = 10
vis_dir = '/scratch/visualization/aug_deform_vis'

def vis():
    seed_everything('12345', workers=True)

        # Prepare data
    datamodule = {
        'data_dir': '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch',
        'num_points': 1024,
        'train_list': 'list/hs/train.txt',
        'train_shape_list': 'list/hs/train.txt',
        'sketch_dir': 'aligned_sketch'
    }
    split = 'val'
    num_points = 1024
    shape_dataset = SketchyVR_original(datamodule, mode='train', source_type='sketch')
    sketch_loader = DataLoader(dataset=shape_dataset, batch_size=batch_size, \
                shuffle=False, drop_last=False, collate_fn=shape_dataset.collate, \
                num_workers=4, pin_memory=False, \
                worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))
    
    transform = transforms.Compose(
        [
            # d_utils.PointcloudToTensor(),
            d_utils.PointcloudScale(),
            # d_utils.PointcloudRotate(),
            # d_utils.PointcloudRotatePerturbation(),
            d_utils.PointcloudTranslate(),
            # d_utils.PointcloudJitter(),
            # RandomDropout(),
            d_utils.PointcloudRandomInputDropout()
        ]
    )
    deformer = get_deformer('deformer_cage_sh2sh_template')

    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)
    for i, data in enumerate(sketch_loader):
        subdir = os.path.join(vis_dir, f'batch_{i}')
        if not os.path.exists(subdir):
            os.mkdir(subdir)

        sketch = torch.tensor(data['source_shape']).cuda() # torch.Size([4, 1024, 3])
        shape = torch.tensor(data['target_shape']).cuda() # torch.Size([4, 1024, 3])
        # file_names = data['source_file']
        B, N, C = sketch.shape       
        B, M, C = shape.shape

        if N != num_points: # N can be larger than M
            sketch = sample_farthest_points(sketch.transpose(1,2), num_points).transpose(1,2)
        if M != num_points: # N can be larger than M
            shape = sample_farthest_points(shape.transpose(1,2), num_points).transpose(1,2)

        aug_sketch = d_utils.apply_random_scale_xyz(sketch)
        aug_shape = d_utils.apply_random_scale_xyz(shape)
        source_shape = torch.repeat_interleave(aug_shape, torch.ones(B, dtype=torch.long).cuda()*B, dim=0)
        target_shape = aug_shape.repeat(B, 1, 1)

        deformed_shape_1 = deformer(source_shape, target_shape)['deformed'].detach().data # torch.Size([6, 1024, 3])
        distance_1 = pytorch3d.loss.chamfer_distance(
            deformed_shape_1, target_shape, batch_reduction=None)[0].view([B, B])
        deformed_shape_2 = deformer(target_shape, source_shape)['deformed'].detach().data # torch.Size([6, 1024, 3])
        distance_2 = pytorch3d.loss.chamfer_distance(
            deformed_shape_2, source_shape, batch_reduction=None)[0].view([B, B])
        actual_distance = (distance_1 + distance_2) / 2

        for index in range(shape.shape[0]):
            np.save(os.path.join(subdir, f'pc_{index}_shape.npy'), shape[index].cpu().numpy())
            np.save(os.path.join(subdir, f'pc_{index}_sketch.npy'), sketch[index].cpu().numpy())
            np.save(os.path.join(subdir, f'pc_{index}_aug_shape.npy'), aug_shape[index].cpu().numpy())
            np.save(os.path.join(subdir, f'pc_{index}_aug_sketch.npy'), aug_sketch[index].cpu().numpy())

            for index_2 in range(shape.shape[0]):
                np.save(os.path.join(subdir, f'pc_{index}_{index_2}_deform1.npy'), deformed_shape_1[index * batch_size + index_2].cpu().numpy())
                np.save(os.path.join(subdir, f'pc_{index}_{index_2}_deform2.npy'), deformed_shape_2[index * batch_size + index_2].cpu().numpy())

        np.save(os.path.join(subdir, 'distance.npy'), distance_2.cpu().numpy())
        np.save(os.path.join(subdir, 'sym_distance.npy'), actual_distance.cpu().numpy())

        if i > 10:
            break

def save_png():
    for batch_id in range(12):
        subdir = os.path.join(vis_dir, f'batch_{batch_id}')

        for distance in ['distance', 'sym_distance']:
            distance_mat = np.load(os.path.join(subdir, f'{distance}.npy'))
            for direction in ['deform1', 'deform2']:
                fig_all = []
                shape_dx=20

                gts = [cv2.imread(os.path.join(subdir, f'pc_{index}_aug_shape_00.jpg'))[shape_dx:224-shape_dx,shape_dx:224-shape_dx,:] for index in range(batch_size)]
                target = cv2.hconcat(gts)
                final_list = []
                output_target = cv2.copyMakeBorder(target, 0,0,224-shape_dx*2,0, cv2.BORDER_CONSTANT, value=(255,255,255))
                final_list.append(output_target)
                for index in range(batch_size):
                    #GT
                    all_fig = []
                    gt = cv2.imread(os.path.join(subdir, f'pc_{index}_aug_shape_00.jpg'))[shape_dx:224-shape_dx,shape_dx:224-shape_dx,:]
                    all_fig.append(gt)
                    shape_imgs = []
                    for index_2 in range(batch_size):
                        img = cv2.imread(os.path.join(subdir, f'pc_{index}_{index_2}_{direction}_00.jpg'))[shape_dx:224-shape_dx,shape_dx:224-shape_dx,:] 
                        img = cv2.putText(img, f'{distance_mat[index][index_2] * 100 :.2f}', (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 0, 0), 2, cv2.LINE_AA)
                        shape_imgs.append(img)
                    all_fig.extend(shape_imgs)
                    fig_row = cv2.hconcat(all_fig)

                    fig_all.append(fig_row)
                final_list.extend(fig_all)
                fig_final = cv2.vconcat(final_list)
                        
                save_path = os.path.join(vis_dir, 'results', f'{batch_id}_{distance}_{direction}.png')
                cv2.imwrite(save_path,fig_final) 

if __name__ == "__main__":
    save_png()
    # vis()
