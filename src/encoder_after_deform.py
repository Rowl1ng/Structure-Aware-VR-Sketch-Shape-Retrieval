import hydra
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
import torch.nn.functional as F
import os
import torch
import time
from pytorch_lightning.utilities.cloud_io import load as pl_load
from src.utils import utils
from torch.utils.data import DataLoader
from src.test import get_best_ckpt, get_deformer, compute_metrics
import numpy as np
import omegaconf
def rank_by_encoder_w_deformer(args, sketch_loader, shape_loader, log):
    sketch_loader = DataLoader(dataset=sketch_dataset, batch_size=1, \
                shuffle=False, drop_last=False, collate_fn=sketch_dataset.collate, \
                    num_workers=4, pin_memory=False, \
                    worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))
    shape_loader = DataLoader(dataset=shape_dataset, batch_size=10, \
                    shuffle=False, drop_last=False, collate_fn=shape_dataset.collate, \
                    num_workers=4, pin_memory=False, \
                    worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))


    path = '/vol/research/sketching/projects/VR_Sketch_lightning/project/logs/experiments/{}/.hydra/config.yaml'.format(args.encoder)
    config = omegaconf.OmegaConf.load(path)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)
    
    # 1. Load encoder
    checkpoint_file = get_best_ckpt(config, ckpt_path=args.encoder)
    encoder_ckpt = pl_load(checkpoint_file, map_location=lambda storage, loc: storage)  
    model.load_state_dict(encoder_ckpt['state_dict'])

    deformer = get_deformer(args.deformer)

    pdist = torch.nn.PairwiseDistance(p=2)
    # Init lightning model
    model = model.eval().cuda()
    shape_features = []
    with torch.no_grad():
        for i, data in enumerate(shape_loader):
            shape_points = data['shape'].cuda()
            shape_z = model(shape_points)
            shape_features.append(shape_z)
        shape_features = torch.cat(shape_features, 0)
        shape_features = F.normalize(shape_features, p=2, dim=1)

        dist_list = []
        # M sketches, N shapes
        for bidx, sketch_batch in enumerate(sketch_loader):
            sketch = torch.Tensor(sketch_batch['shape']).cuda()
            sketch = sketch.repeat(config.datamodule.batch_size, 1, 1) # torch.Size([6, 1024, 3])
            tic = time.perf_counter()
            sketch_features = []
            for bidy, shape_batch in enumerate(shape_loader):
                shape = torch.Tensor(shape_batch['shape']).cuda() # torch.Size([6, 1024, 3])
                B, _, _ = shape.shape
                deformed = deformer(sketch[:B, :, :], shape)['deformed'] # torch.Size([6, 1024, 3])
                sketch_z = model.encoder(deformed.transpose(1, 2))
                sketch_features.append(sketch_z)
            # sketch_features: N x K
            sketch_features = torch.cat(sketch_features, 0)
            # dist(sketch_features, shape_features)
            sketch_features = F.normalize(sketch_features, p=2, dim=1)
            dist = pdist(sketch_features, shape_features)
            dist_list.extend(dist.data.cpu().numpy())
            toc = time.perf_counter()
            log.info(f'Finished: {bidx} in {toc - tic:0.4f} seconds')
    return dist_list

def evaluate(shape_dataset, save_path, log):
    dis_mat = np.load(save_path).reshape(202,202)

    log.info(f' Metrics of {dis_mat.shape[0]} to {dis_mat.shape[1]}: ')

    dist_list = compute_metrics(shape_dataset, dis_mat, None, log)

    save_dir = os.path.dirname(save_path)
    # fitting_gap_path = os.path.join(save_dir, 'CD_d.npy')
    # np.save(fitting_gap_path, np.array(dist_list['chamfer_deform']))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--encoder', type=str, default='retrieval_fitting_gap_triplet_2', 
                    help='encoder')

    parser.add_argument('--deformer', type=str, default='deformer_cage_sk2sh_template', 
                        help='deformer')
    args = parser.parse_args()
    save_dir = '/vol/research/sketching/projects/VR_Sketch_lightning/project/logs/experiments/{}/inference'.format(args.encoder)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    log = utils.logger_config(os.path.join(save_dir, 'run.log'), __name__)
    datamodule = {
        'data_dir': '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch',
        'num_points': 1024,
        'val_list': 'list/hs/val.txt',
        'val_shape_list': 'list/hs/val.txt',
        'test_list': 'list/hs/test.txt',
        'test_shape_list': 'list/hs/test.txt',

    }
    split = 'test'
    from src.datasets.SketchyVRLoader import SketchyVR_single

    shape_dataset = SketchyVR_single(datamodule, mode=split, type='shape')
    sketch_dataset = SketchyVR_single(datamodule, mode=split, type='sketch')

    save_path = os.path.join(save_dir, 'encoder_after_deform.npy')
    
    if not os.path.exists(save_path):
        tic = time.perf_counter()

        dist_list = rank_by_encoder_w_deformer(args, sketch_dataset, shape_dataset, log)
        toc = time.perf_counter()
        log.info(f'Finished in {toc - tic:0.4f} seconds')

        np.save(save_path, dist_list)

    evaluate(shape_dataset, save_path, log)