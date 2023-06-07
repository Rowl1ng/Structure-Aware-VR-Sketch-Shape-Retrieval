from glob import glob
import os
import importlib
import pytorch3d.loss
import torch
import torch.nn as nn
import sys
import pytorch_lightning as pl
# sys.path.append("./emd/")
# import emd_module as emd
import os
import src.utils.io as io
from src.utils.utils import normalize_to_box, sample_farthest_points
from src.utils.custom_loss import ordered_l2
from src.utils.io import vis




def save_pts(f, points, normals=None):
    if normals is not None:
        normals = normals.cpu().detach().numpy()
    io.save_pts(f, points.cpu().detach().numpy(), normals=normals)

class MorphNet(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(MorphNet, self).__init__()
        # self.config = config
        indim = 3 #self.config.indim
        npoints = 1024 # self.config.num_pts # 1024
        encoder_mlp = [16, 64, 128, 256, 512]
        drift_mlp = [256, 128]
        self.npoints = npoints
        self.enc_sizes = [indim] + encoder_mlp
        self.drift_sizes = [self.enc_sizes[-1] + indim] + drift_mlp + [indim]
        encoder_blocks = [nn.Linear(in_f, out_f)
                          for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]

        self.encoder = nn.Sequential(*encoder_blocks)

        drift_blocks = [nn.Linear(in_f, out_f)
                        for in_f, out_f in zip(self.drift_sizes, self.drift_sizes[1:])]

        self.drift_net = nn.Sequential(*drift_blocks)

    def forward(self, sketch):
        x = sketch

        x_feat = self.encoder(x)
        x_feat = torch.max(x_feat, 1, keepdim=True)[0]  # .view(-1, self.enc_sizes[-1])
        x_feat = x_feat.repeat([1, self.npoints, 1])
        feat = torch.cat((x, x_feat), axis=2)
        dw = self.drift_net(feat)
        x_deform = x + dw
        return x_deform

class WPSNet(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(WPSNet, self).__init__()
        indim = 3#self.config.indim
        npoints = 1024#self.config.num_pts # 1024
        encoder_mlp = [16, 64, 128, 256, 512]
        drift_mlp = [256, 128]
        self.npoints = npoints
        self.enc_sizes = [indim] + encoder_mlp
        self.drift_sizes = [self.enc_sizes[-1] * 2 + indim] + drift_mlp + [indim]
        encoder_blocks = [nn.Linear(in_f, out_f)
                          for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]

        self.encoder = nn.Sequential(*encoder_blocks)

        drift_blocks = [nn.Linear(in_f, out_f)
                        for in_f, out_f in zip(self.drift_sizes, self.drift_sizes[1:])]

        self.drift_net = nn.Sequential(*drift_blocks)
        # self.EMD = emd.emdModule()


    def forward(self, source_shape, target_shape):
        z = target_shape
        x = source_shape

        x_feat = self.encoder(x)
        x_feat = torch.max(x_feat, 1, keepdim=True)[0]  # .view(-1, self.enc_sizes[-1])
        z_feat = self.encoder(z)
        z_feat = torch.max(z_feat, 1, keepdim=True)[0]  # .view(-1, self.enc_sizes[-1])
        x_feat = x_feat.repeat([1, self.npoints, 1])
        z_feat = z_feat.repeat([1, self.npoints, 1])

        feat = torch.cat((x, x_feat, z_feat), axis=2)
        dw = self.drift_net(feat)
        x_deform = x + dw
        return x_deform

    def training_step(self, shape, sketch):
        x_deform = self.forward(shape, sketch)
        dist, _ = self.EMD(x_deform, shape, 0.005, 50)
        emd = torch.sqrt(dist).mean(1)
        if mode in ["test", "valid"]:
            acc = {}
            acc['EMD_error'] = emd
            return acc, {}
        return x_deform, emd.mean()


class Deformer_KP(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self._build_model(*args, **kwargs)
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.ordered_l2 = ordered_l2()

    def _build_model(self, *args, **kwargs):
        self.deform_net = WPSNet(*args, **kwargs)
        self.KP_net = PointNet2RetrievalMSG_KP(*args, **kwargs)
        the_module = importlib.import_module('src.utils.custom_loss')
        self.KP_dist_fun = getattr(the_module, self.hparams.KP_dist)()

    def forward(self, source_shape, target_shape, inference=False):
        """
        source_shape (B,3,N)
        target_shape (B,3,M)
        """
        B, _, _ = source_shape.shape

        deformed_shape = self.deform_net.forward(source_shape, target_shape)
        if inference:
            input = torch.cat((source_shape, target_shape, deformed_shape), axis=0)
            KP_list, xyz, features = self.KP_net.forward(input)
            source_keypoints, target_keypoints, deformed_keypoints = torch.split(KP_list, [B, B, B])
        else:
            input = torch.cat((target_shape, deformed_shape), axis=0)
            KP_list, xyz, features = self.KP_net.forward(input)
            target_keypoints, deformed_keypoints  = torch.split(KP_list, [B, B])
            source_keypoints = None

        outputs = {
            "source_keypoints": source_keypoints,
            "target_keypoints": target_keypoints,
            "deformed_keypoints": deformed_keypoints,
            "deformed": deformed_shape,
        }

        return outputs


    def _sum_losses(self, losses, names):
        return sum(v for k, v in losses.items() if k in names)
    

    def training_step(self, batch, batch_idx):
        source_shape, target_shape = self.prepare_batch(batch)

        losses = {}

        optimizer, KP_optimizer = self.optimizers()
        KP_optimizer.zero_grad()
        optimizer.zero_grad()

        # Manual optimization
        if self.trainer.global_step < self.hparams.steps_KP:
            input = torch.cat((source_shape, target_shape), axis=0)
            KP_list, xyz, features = self.KP_net.forward(input)
            losses['KP_net'] = self.KP_net.ComputeLoss3d(input, KP_list, None, None, None)
            losses['KP_net'].backward(retain_graph=False) # set to True cause OOM
            KP_optimizer.step()
            losses['loss'] = losses['KP_net']
        else:
            outputs = self(source_shape, target_shape)
            if self.hparams.lambda_KP_dist > 0:
                KP_dist = self.KP_dist_fun(outputs["target_keypoints"], outputs["deformed_keypoints"])
                losses['KP_dist'] = self.hparams.lambda_KP_dist * KP_dist
            
            losses['PC_dist'] = pytorch3d.loss.chamfer_distance(outputs["deformed"], target_shape)[0]
            # Symmetry loss
            if self.hparams.lambda_sym > 0:
                deformed_sym = outputs["deformed"] * torch.tensor([-1, 1, 1]).cuda()  # for shapenet shapes
                sym_loss = self.sym_loss_fun(outputs["deformed"], deformed_sym)
                losses['sym'] = self.lambda_sym * sym_loss
                loss = self._sum_losses(losses, ['KP_dist', 'sym'])
            else:
                loss = self._sum_losses(losses, ['KP_dist'])
            losses['loss'] = loss
            loss.backward()
            optimizer.step()
            if not self.hparams.stage2_deform_only:
                KP_optimizer.step()
        for item in losses.keys():
            self.log('train/{}'.format(item), losses[item], on_step=False, on_epoch=True, logger=True)

        # self.log('train/KP_dist', losses['KP_dist'], on_step=False, on_epoch=True, logger=True)
        # self.log('train/PC_dist', losses['PC_dist'], on_step=False, on_epoch=True, logger=True)
        self.log('train/loss', losses['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return losses['loss']

    
    def prepare_batch(self, batch):
        source_shape = batch['source_shape'] # torch.Size([4, 1024, 3])
        target_shape = batch['target_shape'] # torch.Size([4, 1024, 3])
        
        B, N, C = source_shape.shape       
        B, M, C = target_shape.shape

        if N != M: # N can be larger than M
            source_shape = sample_farthest_points(source_shape.transpose(1,2), M).transpose(1,2)
        
        return source_shape, target_shape

    def validation_step(self, batch, batch_idx):
        source_shape, target_shape = self.prepare_batch(batch)
        losses = {}

        if self.trainer.global_step < self.hparams.steps_KP:
            input = torch.cat((source_shape, target_shape), axis=0)
            KP_list, xyz, features = self.KP_net.forward(input)
            losses['KP_net'] = self.KP_net.ComputeLoss3d(input, KP_list, None, None, None)
            losses['loss'] = torch.tensor(0.2).cuda()
            outputs = {}
            B, _, _ = source_shape.shape
            source_keypoints, target_keypoints  = torch.split(KP_list, [B, B])
            outputs["source_keypoints"] = source_keypoints          
            outputs["target_keypoints"] = target_keypoints
        else:
            outputs = self(source_shape, target_shape, inference=True)
            losses['KP_dist'] = self.KP_dist_fun(outputs["target_keypoints"], outputs["deformed_keypoints"])
            losses['PC_dist'] = pytorch3d.loss.chamfer_distance(outputs["deformed"], target_shape)[0]
            losses['loss'] = losses['KP_dist']
        for item in losses.keys():
            self.log('val/{}'.format(item), losses[item], on_step=False, on_epoch=True, logger=True)
        self.log('val_loss', losses['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.hparams.vis and batch_idx==0:
            save_dir = os.path.join(self.hparams.save_dir, '%07d' % self.trainer.global_step)
            vis(source_shape, target_shape,
                batch, outputs, 
                save_dir)

        return losses['loss']

    def configure_optimizers(self):
        params = [{"params": self.deform_net.parameters()}]
        optimizer = torch.optim.Adam(params, lr=self.hparams.lr)
        params = [{"params": self.KP_net.parameters()}]
        KP_optimizer = torch.optim.Adam(params, lr=self.hparams.lr)

        return optimizer, KP_optimizer

    def test_step(self, batch, batch_idx):
        source_shape, target_shape = self.prepare_batch(batch)
        outputs = self(source_shape, target_shape, inference=True)
        losses = {}
        # Metric 1: PC_dist
        losses['PC_dist'] = pytorch3d.loss.chamfer_distance(
            outputs['deformed'], target_shape)[0]
         
        # Metric 2: KP_dist
        # step 1: predict KP for deformed
        # step 2: compute KP distance
        ## a. Chamfer distance
        losses['KP_dist_CD'] = pytorch3d.loss.chamfer_distance(
                outputs["deformed_keypoints"], 
                outputs["target_keypoints"])[0]
        ## b. Ordered L2 distance
        losses['KP_dist_L2'] = self.ordered_l2(                
                outputs["deformed_keypoints"], 
                outputs["target_keypoints"])
        
        self.log('test/PC_dist', losses['PC_dist'], on_step=False, on_epoch=True, logger=True)
        self.log('test/KP_dist_CD', losses['KP_dist_CD'], on_step=False, on_epoch=True, logger=True)
        self.log('test/KP_dist_L2', losses['KP_dist_L2'], on_step=False, on_epoch=True, logger=True)

        if self.hparams.vis:
            save_dir = os.path.join(self.hparams.save_dir, 'test')
            vis(source_shape, target_shape, 
                batch, outputs, 
                save_dir)
        return losses
