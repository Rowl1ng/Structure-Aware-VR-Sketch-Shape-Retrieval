import pytorch3d.loss
import pytorch3d.utils
import torch
import torch.nn as nn
import sys
import pytorch_lightning as pl
# sys.path.append("./emd/")
# import emd_module as emd
from src.utils.networks import Linear, MLPDeformer2, PointNetfeat
from src.utils.point_cloud_utils import normalize_to_box, sample_farthest_points
from src.utils.io import vis
from src.utils.cages import deform_with_MVC
from einops import rearrange
from src.utils.custom_loss import ordered_l2
from src.utils.nn import weights_init
import os

class Deformer_KP(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.automatic_optimization=False
        self.save_hyperparameters()
        self._build_model(*args, **kwargs)
        self.ordered_l2 = ordered_l2()
        self.apply(weights_init)
        import importlib

        the_module = importlib.import_module('src.utils.custom_loss')
        self.KP_dist_fun = getattr(the_module, self.hparams.KP_dist)()

    def _build_model(self, *args, **kwargs):
        template_vertices, template_faces = self.create_cage()
        self.init_template(template_vertices, template_faces)
        self.init_networks()

    def create_cage(self):
        # cage (1, N, 3)
        mesh = pytorch3d.utils.ico_sphere(self.hparams.ico_sphere_div, device='cuda:0')
        init_cage_V = mesh.verts_padded()
        init_cage_F = mesh.faces_padded()
        init_cage_V = self.hparams.cage_size * normalize_to_box(init_cage_V)[0]
        init_cage_V = init_cage_V.transpose(1, 2)
        return init_cage_V, init_cage_F
    
    def init_template(self, template_vertices, template_faces):
        # save template as buffer
        self.register_buffer("template_faces", template_faces)
        self.register_buffer("template_vertices", template_vertices)
        
        # n_keypoints x number of vertices
        self.influence_param = nn.Parameter(torch.zeros(self.hparams.num_structure_points, self.template_vertices.shape[2]), requires_grad=True)

    def init_networks(self):
        dim = 3
        num_point = self.hparams.num_points
        n_keypoints = self.hparams.num_structure_points
        bottleneck_size = self.hparams.feat_dim #256
        d_residual = True
        normalization = None
        # keypoint predictor
        shape_encoder_kpt = nn.Sequential(
            PointNetfeat(dim=dim, num_points=num_point, bottleneck_size=bottleneck_size),
            Linear(bottleneck_size, bottleneck_size, activation="lrelu", normalization=normalization))
        nd_decoder_kpt = MLPDeformer2(dim=dim, bottleneck_size=bottleneck_size, npoint=n_keypoints,
                                residual=d_residual, normalization=normalization)
        self.keypoint_predictor = nn.Sequential(shape_encoder_kpt, nd_decoder_kpt)

        # influence predictor
        influence_size = n_keypoints * self.template_vertices.shape[2]
        shape_encoder_influence = nn.Sequential(
            PointNetfeat(dim=dim, num_points=num_point, bottleneck_size=influence_size),
            Linear(influence_size, influence_size, activation="lrelu", normalization=normalization))
        dencoder_influence = nn.Sequential(
                Linear(influence_size, influence_size, activation="lrelu", normalization=normalization),
                Linear(influence_size, influence_size, activation=None, normalization=None))
        self.influence_predictor = nn.Sequential(shape_encoder_influence, dencoder_influence)

    def optimize_cage(self, cage, shape, distance=0.4, iters=100, step=0.01):
        """
        pull cage vertices as close to the origin, stop when distance to the shape is bellow the threshold
        """
        for _ in range(iters):
            vector = -cage
            current_distance = torch.sum((cage[..., None] - shape[:, :, None]) ** 2, dim=1) ** 0.5
            min_distance, _ = torch.min(current_distance, dim=2)
            do_update = min_distance > distance
            cage = cage + step * vector * do_update[:, None]
        return cage

    def forward(self, source_shape, target_shape):
        source_shape = source_shape.transpose(1, 2)
        target_shape = target_shape.transpose(1, 2)
        
        """
        source_shape (B,3,N)
        target_shape (B,3,M)
        """
        B, _, _ = source_shape.shape

        if target_shape is not None:
            shape = torch.cat([source_shape, target_shape], dim=0)
        else:
            shape = source_shape
        
        keypoints = self.keypoint_predictor(shape)
        keypoints = torch.clamp(keypoints, -1.0, 1.0)
        if target_shape is not None:
            source_keypoints, target_keypoints = torch.split(keypoints, B, dim=0)
        else:
            source_keypoints = keypoints

        self.keypoints = keypoints

        n_fps = 2 * self.hparams.num_structure_points
        self.init_keypoints = sample_farthest_points(shape, n_fps)

        if target_shape is not None:
            source_init_keypoints, target_init_keypoints = torch.split(self.init_keypoints, B, dim=0)
        else:
            source_init_keypoints = self.init_keypoints
            target_init_keypoints = None

        cage = self.template_vertices
        cage = self.optimize_cage(cage, source_shape)

        outputs = {
            "cage": cage.transpose(1, 2),
            "cage_face": self.template_faces,
            "source_keypoints": source_keypoints.transpose(1, 2),
            "target_keypoints": target_keypoints.transpose(1, 2),
            'source_init_keypoints': source_init_keypoints,
            'target_init_keypoints': target_init_keypoints
        }

        self.influence = self.influence_param[None]
        self.influence_offset = self.influence_predictor(source_shape)
        self.influence_offset = rearrange(
            self.influence_offset, 'b (k c) -> b k c', k=self.influence.shape[1], c=self.influence.shape[2])
        self.influence = self.influence + self.influence_offset

        distance = torch.sum((source_keypoints[..., None] - cage[:, :, None]) ** 2, dim=1)
        n_influence = int((distance.shape[2] / distance.shape[1]) * self.hparams.n_influence_ratio)
        n_influence = max(5, n_influence)
        threshold = torch.topk(distance, n_influence, largest=False)[0][:, :, -1]
        threshold = threshold[..., None]
        keep = distance <= threshold
        influence = self.influence * keep

        base_cage = cage
        keypoints_offset = target_keypoints - source_keypoints
        cage_offset = torch.sum(keypoints_offset[..., None] * influence[:, None], dim=2)
        new_cage = base_cage + cage_offset

        cage = cage.transpose(1, 2)
        new_cage = new_cage.transpose(1, 2)
        deformed_shapes, weights, _ = deform_with_MVC(cage, new_cage, self.template_faces.expand(B, -1, -1), source_shape.transpose(1, 2), verbose=True)

        deformed_keypoints = self.keypoint_predictor(deformed_shapes.transpose(1, 2))
        outputs.update({
            "cage": cage,
            "cage_face": self.template_faces,
            "new_cage": new_cage,
            "deformed": deformed_shapes,
            "deformed_keypoints": deformed_keypoints.transpose(1, 2),
            "weight": weights,
            "influence": influence})

        return outputs

    def prepare_batch(self, batch):
        source_shape = batch['source_shape'] # torch.Size([4, 1024, 3])
        target_shape = batch['target_shape'] # torch.Size([4, 1024, 3])        

        B, N, C = source_shape.shape       
        B, M, C = target_shape.shape

        if N != self.hparams.num_points: # N can be larger than M
            source_shape = sample_farthest_points(source_shape.transpose(1,2), self.hparams.num_points).transpose(1,2)
        if M != self.hparams.num_points: # N can be larger than M
            target_shape = sample_farthest_points(target_shape.transpose(1,2), self.hparams.num_points).transpose(1,2)

        return source_shape, target_shape

    def training_step(self, batch, batch_idx):
        source_shape, target_shape = self.prepare_batch(batch)
        outputs = self(source_shape, target_shape)
        losses = {}

        if self.hparams.lambda_init_points > 0:
            init_points_loss = pytorch3d.loss.chamfer_distance(
                rearrange(self.keypoints, 'b d n -> b n d'), 
                rearrange(self.init_keypoints, 'b d n -> b n d'))[0]
            losses['KP_net'] = self.hparams.lambda_init_points * init_points_loss

        if self.hparams.lambda_chamfer > 0:
            chamfer_loss = pytorch3d.loss.chamfer_distance(
                outputs["deformed"], target_shape)[0]
            losses['PC_dist'] = self.hparams.lambda_chamfer * chamfer_loss

        if self.hparams.lambda_KP_dist > 0:
            KP_dist = self.KP_dist_fun(outputs["target_keypoints"], outputs["deformed_keypoints"])
            losses['KP_dist'] = self.hparams.lambda_KP_dist * KP_dist

        if self.hparams.lambda_influence_predict_l2 > 0:
            losses['influence_predict_l2'] = self.hparams.lambda_influence_predict_l2 * torch.mean(self.influence_offset ** 2)

        loss = self.optimize(losses)
        for item in losses.keys():
            self.log('train/{}'.format(item), losses[item], on_step=False, on_epoch=True, logger=True)

        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss #tensor(1.6280, grad_fn=<AddBackward0>)

    def validation_step(self, batch, batch_idx):
        source_shape, target_shape = self.prepare_batch(batch)
        outputs = self(source_shape, target_shape)
        losses = {}

        if self.hparams.lambda_init_points > 0:
            init_points_loss = pytorch3d.loss.chamfer_distance(
                rearrange(self.keypoints, 'b d n -> b n d'), 
                rearrange(self.init_keypoints, 'b d n -> b n d'))[0]
            losses['KP_net'] = init_points_loss

        # if self.hparams.lambda_chamfer > 0:
        chamfer_loss = pytorch3d.loss.chamfer_distance(
            outputs['deformed'], target_shape)[0]
        losses['PC_dist'] = chamfer_loss

        if self.hparams.lambda_KP_dist > 0:
            KP_dist = self.KP_dist_fun(outputs["target_keypoints"], outputs["deformed_keypoints"])
            losses['KP_dist'] = KP_dist

        if self.hparams.lambda_influence_predict_l2 > 0:
            losses['influence_predict_l2'] = torch.mean(self.influence_offset ** 2)
    
        for item in losses.keys():
            self.log('val/{}'.format(item), losses[item], on_step=False, on_epoch=True, logger=True)

        self.log('val_loss', losses['PC_dist'], on_step=False, on_epoch=True, prog_bar=True)

        if self.hparams.vis and batch_idx==0:
            save_dir = os.path.join(self.hparams.save_dir, '%07d' % self.trainer.global_step)
            vis(source_shape, target_shape,
                batch, outputs, 
                save_dir)

        return losses['PC_dist']

    def _sum_losses(self, losses, names):
        return sum(v for k, v in losses.items() if k in names)

    def optimize(self, losses):
        optimizer, keypoint_optimizer = self.optimizers()
        
        keypoint_optimizer.zero_grad()
        optimizer.zero_grad()

        if self.trainer.global_step < self.hparams.steps_KP:
            keypoints_loss = self._sum_losses(losses, ['KP_net'])
            keypoints_loss.backward(retain_graph=True) 
            keypoint_optimizer.step()
            return keypoints_loss
        else:
            loss = self._sum_losses(losses, losses.keys())
            loss.backward()
            optimizer.step()
            keypoint_optimizer.step()
            return loss

    def test_step(self, batch, batch_idx):
        source_shape, target_shape = self.prepare_batch(batch)
        outputs = self(source_shape, target_shape)
        losses = {}
        # Metric 1: PC_dist
        losses['PC_dist'] = pytorch3d.loss.chamfer_distance(
            outputs['deformed'], target_shape)[0]
         
        # Metric 2: KP_dist
        # step 1: predict KP for deformed
        ## outputs['deformed'] : torch.Size([16, 1024, 3])
        deformed_keypoints = self.keypoint_predictor(
                rearrange(outputs['deformed'], 'b n d -> b d n')) #(B,3,N)
        outputs['deformed_keypoints'] = torch.clamp(deformed_keypoints, -1.0, 1.0).transpose(1, 2)
        # step 2: compute KP distance
        ## a. Chamfer distance
        losses['KP_dist_CD'] = pytorch3d.loss.chamfer_distance(
                outputs['deformed_keypoints'].transpose(1, 2), 
                outputs["target_keypoints"].transpose(1, 2))[0]
        ## b. Ordered L2 distance
        losses['KP_dist_L2'] = self.ordered_l2(                
                outputs['deformed_keypoints'], 
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

    def configure_optimizers(self):
        params = [{"params": self.influence_predictor.parameters()}]
        optimizer = torch.optim.Adam(params, lr=self.hparams.lr)
        optimizer.add_param_group({'params': self.influence_param, 'lr': 10 * self.hparams.lr})
        params = [{"params": self.keypoint_predictor.parameters()}]
        keypoint_optimizer = torch.optim.Adam(params, lr=self.hparams.lr)

        return optimizer, keypoint_optimizer
