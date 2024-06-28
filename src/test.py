import numpy as np
import os
import torch
import time
from typing import List, Optional
from torch.serialization import load
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from src.utils import utils
from glob import glob
# from pytorch_lightning.utilities.cloud_io import load as pl_load
from lightning_fabric.utilities.cloud_io import _load as pl_load

# from src.datasets.SketchyVRLoader import SketchyVR_single
import omegaconf
from src.utils.evaluation import compute_distance, compute_acc_at_k

log = utils.get_logger(__name__)

def get_deformer(model_name):
    # Load Deformer:shape->shape
    path = 'project/logs/deformer/{}/.hydra/config.yaml'.format(model_name)
    cfg = omegaconf.OmegaConf.load(path)
    checkpoint_file = 'project/logs/deformer/{}/checkpoints/last.ckpt'.format(model_name)
    print(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    ckpt = pl_load(checkpoint_file, map_location=lambda storage, loc: storage)  
    model.load_state_dict(ckpt['state_dict'])

    print('Load deformer model from: {}'.format(checkpoint_file))
    model = model.eval().cuda()
    return model

def get_retrieval_model(model_name):
    # Load Retrieval model
    exp_dir = '/vol/research/sketching/projects/VR_Sketch_lightning/project/logs/multifold'
    # path = os.path.join(exp_dir, '{}/.hydra/config.yaml'.format(model_name))
    path = '/vol/research/sketching/projects/VR_Sketch_lightning/project/logs/multifold/adaptive_triplet_multickpt_0.3_1.2_1/.hydra/config.yaml'
    cfg = omegaconf.OmegaConf.load(path)
    checkpoint_file = os.path.join(exp_dir, '{}/checkpoints/last.ckpt'.format(model_name))
    print(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    ckpt = pl_load(checkpoint_file, map_location=lambda storage, loc: storage)  
    model.load_state_dict(ckpt['state_dict'])

    print('Load retrieval model from: {}'.format(checkpoint_file))
    model = model.eval().cuda()
    return model


def get_latest_ckpt(config):
    checkpoint_file = os.path.join(config.paths["log_dir"], 'experiments', config["name"], "checkpoints", "last.ckpt")
    if os.path.exists(checkpoint_file):
        log.info(f'Load latest model: {checkpoint_file}')

    return checkpoint_file

def get_best_ckpt(config, ckpt_path=None):
    # import pdb
    # pdb.set_trace()
    if ckpt_path is None:
        checkpoint_files = glob(os.path.join(config["work_dir"], 'logs/experiments', config["name"], "checkpoints", "val*.ckpt"))
    else:
        checkpoint_files = glob(os.path.join(config["work_dir"], 'logs/experiments', ckpt_path, "checkpoints", "val*.ckpt"))
    if len(checkpoint_files)>0:
        if 'val_loss' in os.path.basename(checkpoint_files[0]):
            checkpoint_file = sorted(checkpoint_files)[0]
        else:
            checkpoint_file = sorted(checkpoint_files)[-1]
        log.info(f'Load best model: {checkpoint_file}')

    return checkpoint_file

# Inference: d_R(D(s;t), t) train with deformer triplet loss
# Inference: d_R(D(s;t), t) train without deformer triplet loss

def rank_by_encoder(config, sketch_loader, shape_loader):
    if config.test_ckpt == 'best':
        checkpoint_file = get_best_ckpt(config)
    elif config.test_ckpt == 'last':
        checkpoint_file = get_latest_ckpt(config)
    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model, checkpoint_path=checkpoint_file)
    ckpt = pl_load(checkpoint_file, map_location=lambda storage, loc: storage)  

    model = model.cuda()
    model.load_state_dict(ckpt['state_dict'])
    
    model = model.eval()

    sketch_features = []
    shape_features = []
    with torch.no_grad():
        for i, data in enumerate(sketch_loader):
            sketch = data['shape'].cuda()
            feat = model.encoder(sketch.transpose(1, 2))
            sketch_features.append(feat)

        for i, data in enumerate(shape_loader):
            shape = data['shape'].cuda()
            feat = model.encoder(shape.transpose(1, 2))
            shape_features.append(feat)
    shape_features = torch.cat(shape_features, 0)
    sketch_features = torch.cat(sketch_features, 0)

    d_feat_z = compute_distance(sketch_features, shape_features, l2=True).cpu().numpy()

    return d_feat_z

def rank_by_encoder_egocentric(config, sketch_loader, shape_loader):
    checkpoint_file = get_best_ckpt(config)
    # checkpoint_file = get_latest_ckpt(config)
    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model, checkpoint_path=checkpoint_file)
    ckpt = pl_load(checkpoint_file, map_location=lambda storage, loc: storage)  

    model = model.cuda()
    model.load_state_dict(ckpt['state_dict'])
    
    model = model.eval()

    sketch_features = []
    shape_features = []
    matrix = []
    with torch.no_grad():
        for i, data in enumerate(sketch_loader):
            sketch = data['shape'].cuda()
            feat, mat = model.encoder(sketch.transpose(1, 2), matrix=True)
            sketch_features.append(feat)
            matrix.append(mat)
        for i, data in enumerate(shape_loader):
            shape = data['shape'].cuda()
            feat = model.encoder(shape.transpose(1, 2))
            shape_features.append(feat)
    shape_features = torch.cat(shape_features, 0)
    sketch_features = torch.cat(sketch_features, 0)
    matrix = torch.cat(matrix, 0)
    from src.utils.custom_loss import egocentric
    import torch.nn.functional as F
    sketch_features = F.normalize(sketch_features, p=2, dim=1)
    shape_features = F.normalize(shape_features, p=2, dim=1)

    # d_feat_z = compute_distance(sketch_features, shape_features, l2=True).cpu().numpy()
    d_feat_z = egocentric(sketch_features, shape_features, matrix).cpu().numpy()

    return d_feat_z

#TODO: pytorch3d.loss.point_mesh_edge_distance
def rank_by_point_mesh_edge_distance(config, sketch_loader, shape_loader):
    pass

def compute_metrics(shape_dataset, dis_mat, writer, log):
    pair_sort = np.argsort(dis_mat)

    # 1. Top-K Acc
    acc_at_k = compute_acc_at_k(pair_sort)
    for acc, k in zip(acc_at_k, [1, 5, 10]):
        log.info(f' * Acc@{k}: {acc * 100 :.2f}')
        if writer is not None:
            writer.add_scalar(f"test/Acc@{k}", acc * 100)
    
    # 2. Mean CD (retrieved, GT)
    import pytorch3d.loss
    from src.utils.distance import f_score

    # config.datamodule.num_points = 1024 # 2048
    # dataset = SketchyVR_single(config.datamodule, mode='test', type='shape')

    num_queries = pair_sort.shape[0]
    dist_list = {
        'chamfer': [],
        # 'f_score_0.01': [],
        # 'f_score_0.02': [],

        'chamfer_deform': [],
        'bi_chamfer_deform': [],

        # 'f_score_0.01_deform': [],
        # 'f_score_0.02_deform': []

    }
    # Deformer: shape->shape
    deformer = get_deformer('deformer_cage_sh2sh_shapenet_template_03001627')

    K = 10
    for index in range(num_queries):
        top5 = pair_sort[index, :K]
        gt_shape = shape_dataset.__getitem__(index)['shape']
        gt_shape = torch.tensor(gt_shape).unsqueeze(0).repeat(K, 1, 1).cuda()
        top_shapes = []
        for item in top5:
            shape = torch.tensor(shape_dataset.__getitem__(item)['shape']).unsqueeze(0)
            top_shapes.append(shape)
        top_shapes = torch.cat(top_shapes, 0).cuda()
        top_dist = pytorch3d.loss.chamfer_distance(gt_shape, top_shapes, batch_reduction=None)[0]
        dist_list['chamfer'].append(top_dist.data.cpu().numpy())
        # dist = f_score(gt_shape, top_shapes, radius=0.01)
        # dist_list['f_score_0.01'].append(dist.data.cpu().numpy())
        # dist = f_score(gt_shape, top_shapes, radius=0.02)
        # dist_list['f_score_0.02'].append(dist.data.cpu().numpy())

        deformed_shape = deformer(top_shapes, gt_shape)['deformed'] # torch.Size([6, 1024, 3])

        distance_1 = pytorch3d.loss.chamfer_distance(
            deformed_shape, gt_shape, batch_reduction=None)[0]
        dist_list['chamfer_deform'].append(distance_1.data.cpu().numpy())
        deformed_shape = deformer(gt_shape, top_shapes)['deformed'] # torch.Size([6, 1024, 3])
        distance_2 = pytorch3d.loss.chamfer_distance(
            deformed_shape, top_shapes, batch_reduction=None)[0]
        dist_list['bi_chamfer_deform'].append(((distance_1 + distance_2) / 2).data.cpu().numpy())

        # dist_list['f_score_0.01_deform'].append(f_score(deformed_shape, gt_shape, radius=0.01).data.cpu().numpy())
        # dist_list['f_score_0.02_deform'].append(f_score(deformed_shape, gt_shape, radius=0.02).data.cpu().numpy())

    for metric in dist_list.keys():
        dist_list[metric] = np.array(dist_list[metric])
        for k in [1, 5, 10]:
            log.info(f' * Mean {metric}@{k}: {dist_list[metric][:, :k].mean() * 100 :.2f}')
            if writer is not None:
                writer.add_scalar(f"test/{metric}@{k}", dist_list[metric][:, :k].mean() * 100)
    return dist_list

def evaluate(config, shape_dataset, save_path):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(config["work_dir"], 'logs/experiments', config["name"], 'tb_logs'))
    dis_mat = np.load(save_path) 
    log.info(f' Metrics of {dis_mat.shape[0]} to {dis_mat.shape[0]}: ')
  
    dist_list = compute_metrics(shape_dataset, dis_mat[:, :dis_mat.shape[0]], None, log)

    log.info(f' Metrics of {dis_mat.shape[0]} to {dis_mat.shape[1]}: ')

    dist_list = compute_metrics(shape_dataset, dis_mat, writer, log)
    writer.flush()
    writer.close()

    save_dir = os.path.dirname(save_path)
    fitting_gap_path = os.path.join(save_dir, 'CD_{}_{}.npy'.format(config.datamodule.sketch_dir, config.test_ckpt))
    np.save(fitting_gap_path, np.array(dist_list['chamfer']))
    fitting_gap_path = os.path.join(save_dir, 'CD_d_{}_{}.npy'.format(config.datamodule.sketch_dir, config.test_ckpt))
    np.save(fitting_gap_path, np.array(dist_list['chamfer_deform']))
    fitting_gap_path = os.path.join(save_dir, 'bi_CD_d_{}_{}.npy'.format(config.datamodule.sketch_dir, config.test_ckpt))
    np.save(fitting_gap_path, np.array(dist_list['bi_chamfer_deform']))
    log.info(f'Save FG npy to: {fitting_gap_path}')


def test(config: DictConfig) -> Optional[float]:
    """Contains testing pipeline.
    Instantiates all PyTorch Lightning objects from config.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    split = 'test'
    config.datamodule.test_shape_list = 'list/hs/test_shape.txt'
    from src.datasets.SketchyVRLoader import SketchyVR_single
    shape_dataset = SketchyVR_single(config.datamodule, mode=split, type='shape')
                         

    config['model']['save_dir'] = os.path.join(config["work_dir"], 'logs/experiments', config["name"], 'inference')

    num_shapes = shape_dataset.__len__()
    
    sketch_dataset = SketchyVR_single(config.datamodule, mode=split, type='sketch')

    save_path = os.path.join(config['model']['save_dir'], '{}_rank_{}_{}_{}.npy'.format(split, num_shapes, config.datamodule.sketch_dir, config.test_ckpt))

    if not os.path.exists(save_path):

        if config.get("rank_by") in ['Chamfer', 'f_score', 'deformer', 'encoder_deformer']:
            sketch_batchsize = 1
        else:
            sketch_batchsize = config.datamodule.batch_size

        sketch_loader = DataLoader(dataset=sketch_dataset, batch_size=sketch_batchsize, \
                        shuffle=False, drop_last=False, collate_fn=sketch_dataset.collate, \
                        num_workers=config.datamodule.num_workers, pin_memory=config.datamodule.pin_memory, \
                        worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))
        shape_loader = DataLoader(dataset=shape_dataset, batch_size=config.datamodule.batch_size, \
                        shuffle=False, drop_last=False, collate_fn=shape_dataset.collate, \
                        num_workers=config.datamodule.num_workers, pin_memory=config.datamodule.pin_memory, \
                        worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

        tic = time.perf_counter()

        if 'egocentric' in config.model._target_:
            dist_list = rank_by_encoder_egocentric(config, sketch_loader, shape_loader)
        else:
            dist_list = rank_by_encoder(config, sketch_loader, shape_loader)

        toc = time.perf_counter()
        log.info(f'Finished in {toc - tic:0.4f} seconds')

        dist_list = np.array(dist_list).reshape([-1, num_shapes])

        if not os.path.exists(config['model']['save_dir']):
            os.mkdir(config['model']['save_dir'])
        np.save(save_path, dist_list)
        log.info(f'Save dist npy to: {save_path}')

    # if os.path.exists(save_path):
    evaluate(config, shape_dataset, save_path)
        # quit()

def test_category(config: DictConfig) -> Optional[float]:
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    split = 'test'
    from src.datasets.SketchyVRLoader import SketchyVR_original_category_single
    datamodule = {
        'data_dir': '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch',
        'num_points': 1024,
        'test_shape_list': 'sketch_test.txt',
        'category': config.datamodule.category,
        'abstraction': '1.0'
    }
    sketch_dataset = SketchyVR_original_category_single(datamodule, mode=split, type='sketch')
    target_datamodule = {
        'data_dir': '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch',
        'num_points': 1024,
        'test_shape_list': 'sketch_test_shape.txt',
        'category': config.datamodule.category
    }
    shape_dataset = SketchyVR_original_category_single(target_datamodule, mode=split, type='shape')
    num_shapes = shape_dataset.__len__()

    save_path = os.path.join(config['model']['save_dir'], '{}_rank_{}_{}_{}.npy'.format(split, num_shapes, config.datamodule.category, config.test_ckpt))

    if not os.path.exists(save_path):

        if config.get("rank_by") in ['Chamfer', 'f_score', 'deformer', 'encoder_deformer']:
            sketch_batchsize = 1
        else:
            sketch_batchsize = config.datamodule.batch_size

        sketch_loader = DataLoader(dataset=sketch_dataset, batch_size=sketch_batchsize, \
                        shuffle=False, drop_last=False, collate_fn=sketch_dataset.collate, \
                        num_workers=config.datamodule.num_workers, pin_memory=config.datamodule.pin_memory, \
                        worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))
        shape_loader = DataLoader(dataset=shape_dataset, batch_size=config.datamodule.batch_size, \
                        shuffle=False, drop_last=False, collate_fn=shape_dataset.collate, \
                        num_workers=config.datamodule.num_workers, pin_memory=config.datamodule.pin_memory, \
                        worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

        tic = time.perf_counter()

        dist_list = rank_by_encoder(config, sketch_loader, shape_loader)

        
        toc = time.perf_counter()
        log.info(f'Finished in {toc - tic:0.4f} seconds')

        dist_list = np.array(dist_list).reshape([-1, num_shapes])

        if not os.path.exists(config['model']['save_dir']):
            os.mkdir(config['model']['save_dir'])
        np.save(save_path, dist_list)
        log.info(f'Save dist npy to: {save_path}')

    # if os.path.exists(save_path):
    evaluate(config, shape_dataset, save_path)
        # quit()

if __name__ == "__main__":
    deformer = get_deformer('deformer_cage_sh2sh_shapenet_template_03001627')

