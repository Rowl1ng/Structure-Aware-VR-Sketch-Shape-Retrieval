import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG

from src.models.pointnet2_msg_retrieval import PointNet2RetrievalMSG
from src.utils.custom_loss import ChamferLoss
import os
import numpy as np
import random
from src.utils.point_cloud_utils import write_points_off

def create_color_list(num):
    colors = np.ndarray(shape=(num, 3))
    random.seed(30)
    for i in range(0, num):
        colors[i, 0] = random.randint(0, 255)
        colors[i, 1] = random.randint(0, 255)
        colors[i, 2] = random.randint(0, 255)
    return colors
COLOR_LIST = create_color_list(5000)

class ComputeLoss3d(nn.Module):
    def __init__(self):
        super(ComputeLoss3d, self).__init__()

        self.mse_func = nn.MSELoss()
        # self.cd_loss_fun = chamfer_distance.ComputeCDLoss()
        self.cd_loss_fun = ChamferLoss()
        self.loss = None
        self.consistent_loss = None
        self.cd_loss = None

    def forward(self, gt_points, structure_points, transed_gt_points=None, transed_structure_points=None, trans_func_list=None):

        gt_points = gt_points.cuda()
        structure_points = structure_points.cuda()

        batch_size = gt_points.shape[0]
        pts_num = gt_points.shape[1]
        dim = 3
        stpts_num = structure_points.shape[1]

        self.cd_loss = self.cd_loss_fun(structure_points, gt_points)

        trans_num = 0
        if transed_structure_points is not None:
            transed_structure_points = transed_structure_points.cuda()
            transed_gt_points = transed_gt_points.cuda()
            trans_num = transed_structure_points.shape[0]
            self.cd_loss = self.cd_loss + self.cd_loss_fun(transed_structure_points.view(trans_num * batch_size, stpts_num, dim),
                                                                             transed_gt_points.view(trans_num * batch_size, pts_num, dim))
            self.consistent_loss = None
            for i in range(0, trans_num):
                tmp_structure_points = trans_func_list[i](structure_points)
                tmp_structure_points = tmp_structure_points.detach()
                tmp_structure_points.requires_grad = False
                tmp_consistent_loss = self.mse_func(tmp_structure_points, transed_structure_points[i])
                if self.consistent_loss is None:
                    self.consistent_loss = tmp_consistent_loss
                else:
                    self.consistent_loss = self.consistent_loss + tmp_consistent_loss
            self.consistent_loss = self.consistent_loss / trans_num * 1000


        self.cd_loss = self.cd_loss / (trans_num + 1)

        self.loss = self.cd_loss

        if transed_structure_points is not None:
            self.loss = self.loss + self.consistent_loss
        return self.loss

    def get_cd_loss(self):
        return self.cd_loss

    def get_consistent_loss(self):
        return self.consistent_loss

class PointNet2RetrievalMSG_KP(PointNet2RetrievalMSG):
    def _build_model(self):
        super()._build_model()
        conv1d_stpts_prob_modules = []
        self.num_structure_points = self.hparams['num_structure_points']
        self.tl_loss = self.hparams['tl_loss']
        if self.num_structure_points <= 128 + 256 + 256:
            conv1d_stpts_prob_modules.append(nn.Dropout(0.2))
            conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=128 + 256 + 256, out_channels=512, kernel_size=1))
            conv1d_stpts_prob_modules.append(nn.BatchNorm1d(512))
            conv1d_stpts_prob_modules.append(nn.ReLU())
            in_channels = 512
            while in_channels >= self.num_structure_points * 2:
                out_channels = int(in_channels / 2)
                conv1d_stpts_prob_modules.append(nn.Dropout(0.2))
                conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))
                conv1d_stpts_prob_modules.append(nn.BatchNorm1d(out_channels))
                conv1d_stpts_prob_modules.append(nn.ReLU())
                in_channels = out_channels

            conv1d_stpts_prob_modules.append(nn.Dropout(0.2))
            conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=in_channels, out_channels=self.num_structure_points, kernel_size=1))

            conv1d_stpts_prob_modules.append(nn.BatchNorm1d(self.num_structure_points))
            conv1d_stpts_prob_modules.append(nn.Softmax(dim=2))
        else:
            conv1d_stpts_prob_modules.append(nn.Dropout(0.2))
            conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=128 + 256 + 256, out_channels=1024, kernel_size=1))
            conv1d_stpts_prob_modules.append(nn.BatchNorm1d(1024))
            conv1d_stpts_prob_modules.append(nn.ReLU())

            in_channels = 1024
            while in_channels <= self.num_structure_points / 2:
                out_channels = int(in_channels * 2)
                conv1d_stpts_prob_modules.append(nn.Dropout(0.2))
                conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))
                conv1d_stpts_prob_modules.append(nn.BatchNorm1d(out_channels))
                conv1d_stpts_prob_modules.append(nn.ReLU())
                in_channels = out_channels

            conv1d_stpts_prob_modules.append(nn.Dropout(0.2))
            conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=in_channels, out_channels=self.num_structure_points, kernel_size=1))

            conv1d_stpts_prob_modules.append(nn.BatchNorm1d(self.num_structure_points))
            conv1d_stpts_prob_modules.append(nn.Softmax(dim=2))

        self.conv1d_stpts_prob = nn.Sequential(*conv1d_stpts_prob_modules)

        self.stpts_prob_map = None
        self.ComputeLoss3d = ComputeLoss3d()

    def forward(self, xyz):
        # xyz, features = self._break_up_pc(pc)
        xyz = xyz.transpose(2, 1)
        features = None
        for module in self.SA_modules[:-1]:
            xyz, features = module(xyz, features)
        self.stpts_prob_map = self.conv1d_stpts_prob(features)
        xyz_transform = xyz.transpose(2, 1)
        weighted_xyz = torch.sum(self.stpts_prob_map[:, :, :, None] * xyz_transform[:, None, :, :], dim=2)
        return weighted_xyz, xyz, features

    def inference(self, pc):
        _, xyz, features = self.forward(pc)
        _, features = self.SA_modules[-1](xyz, features)
        return features.squeeze()

    def training_step(self, batch, batch_idx):
        shape = batch['shape']
        sketch = batch['sketch']
        pc = torch.cat([sketch, shape])
        weighted_xyz, xyz, features = self.forward(pc)
        kp_loss = self.ComputeLoss3d(pc, weighted_xyz, None, None, None)

        if self.tl_loss:
            xyz, features = self.SA_modules[-1](xyz, features)
            tl_loss = self.crt_tl(self.fc_layer(features.squeeze(-1)), pc)
            loss = tl_loss + kp_loss
            self.log('tl_loss', tl_loss, on_step=False, on_epoch=True)
        else:
            loss = kp_loss
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('kp_loss', kp_loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        shape = batch['shape']
        sketch = batch['sketch']
        pc = torch.cat([sketch, shape])
        weighted_xyz, xyz, features = self.forward(pc)
        kp_loss = self.ComputeLoss3d(pc, weighted_xyz, None, None, None)

        return dict(val_loss=kp_loss)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("val_loss", avg_loss)

    def vis_single(self, batch, vis_dump_dir):
        shape = batch['shape'].cuda()
        sketch = batch['sketch'].cuda()
        labels = batch['label']
        pc = torch.cat([sketch, shape])
        label = labels[0]
        # xyz, features = self._break_up_pc(pc)
        #
        # for module in self.SA_modules[:-1]:
        #     xyz, features = module(xyz, features)
        # self.stpts_prob_map = self.conv1d_stpts_prob(features)
        # xyz = xyz.transpose(2, 1)

        # weighted_xyz = torch.sum(self.stpts_prob_map[:, :, :, None] * xyz[:, None, :, :], dim=2)
        weighted_xyz, xyz, features = self.forward(pc)

        sketch_stpts, shape_stpts = torch.split(weighted_xyz, [1, 1], dim=0)

        sketch_stpts = sketch_stpts.cpu().detach().numpy()
        shape_stpts = shape_stpts.cpu().detach().numpy()

        sketch_outfname = os.path.join(vis_dump_dir, label + '_sketch_stpts.off')
        sketch_fname = os.path.join(vis_dump_dir, label + '_sketch.off')

        write_points_off(sketch_outfname, sketch_stpts[0], COLOR_LIST[:sketch_stpts.shape[1], :])
        write_points_off(sketch_fname, sketch[0].cpu().detach().numpy())


        shape_outfname = os.path.join(vis_dump_dir, label + '_shape_stpts.off')
        shape_fname = os.path.join(vis_dump_dir, label + '_shape.off')

        write_points_off(shape_outfname, shape_stpts[0], COLOR_LIST[:shape_stpts.shape[1], :])
        write_points_off(shape_fname, shape[0].cpu().detach().numpy())

        print(shape_outfname)

    def vis(self, data_loader, save_dir):
        idx = 0
        import random
        vis_idx = random.sample(range(1, data_loader.dataset.__len__()), 20)

        for data in data_loader:
            idx += 1
            if idx not in vis_idx:
                continue
            vis_dir = f"{save_dir}/vis/dump_vis_{idx}"
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)

            with torch.no_grad():
                self.vis_single(data, vis_dir)

