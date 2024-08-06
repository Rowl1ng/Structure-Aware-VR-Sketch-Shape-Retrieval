import torch
from logging import captureWarnings
import pytorch_lightning as pl
import importlib
from src.models.deformer_cage import Deformer_KP
from src.utils.point_cloud_utils import normalize_to_box, sample_farthest_points
from src.utils.custom_loss import OnlineTripletLoss, egocentric_iter, egocentric
from src.datamodules.TripletSampler import AllNegativeTripletSelector
from pl_bolts.losses.self_supervised_learning import FeatureMapContrastiveTask
from src.utils.evaluation import compute_distance, compute_acc_at_k
from torchvision.transforms import transforms
import src.datamodules.data_utils as d_utils
import torch.nn.functional as F
import os
from src.utils.io import vis
import pytorch3d.loss
from src.utils import template_utils as utils
import numpy as np
import math

class Deformer_Retrieval(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.automatic_optimization=False
        self.save_hyperparameters()
        self._build_model(*args, **kwargs)
        # self.crt_tl = torch.nn.TripletMarginLoss(margin=self.hparams.margin)
        self.deform_triplet_loss = torch.nn.TripletMarginLoss(margin=self.hparams.deform_margin, p=2)
        self.crt_tl = OnlineTripletLoss(self.hparams.margin, AllNegativeTripletSelector())
        if self.hparams.transform:
            self.transforms = transforms.Compose(
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
        # TODO: FeatureMapContrastiveTask
        self.save_dir = os.path.join(self.hparams.save_dir, 'inference')
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.get_shape_loader()
    def get_shape_loader(self):
        from torch.utils.data import DataLoader
        from src.datasets.SketchyVRLoader import SketchyVR_single, SketchyVR_original_category_single

        datamodule = {
        'data_dir': '/vol/vssp/datasets/multiview/3VS/datasets/FineGrained_3DSketch',
        'num_points': 1024,
        'test_shape_list': 'sketch_test_shape.txt',
        'sketch_dir': 'aligned_sketch',
        'num_workers': 4,
        'pin_memory': False,
        'category': self.hparams.category
        }

        val_shape_dataset = SketchyVR_original_category_single(datamodule, mode='val', type='shape')
        self.val_shape_loader = DataLoader(dataset=val_shape_dataset, batch_size=self.hparams.batch_size, \
                        shuffle=False, drop_last=False, collate_fn=val_shape_dataset.collate, \
                        num_workers=datamodule['num_workers'], pin_memory=datamodule['pin_memory'], \
                        worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))
        test_shape_dataset = SketchyVR_original_category_single(datamodule, mode='test', type='shape')
        self.test_shape_loader = DataLoader(dataset=test_shape_dataset, batch_size=self.hparams.batch_size, \
                        shuffle=False, drop_last=False, collate_fn=test_shape_dataset.collate, \
                        num_workers=datamodule['num_workers'], pin_memory=datamodule['pin_memory'], \
                        worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

        fitting_gap_path = '/vol/vssp/datasets/multiview/3VS/datasets/cache/category/{}_{}.npy'
        self.val_fitting_gap = {item: torch.tensor(np.load(fitting_gap_path.format(self.hparams.category, item))).cuda() for item in ['CD_d', 'bi_CD_d']}

    def _build_model(self, *args, **kwargs):
        # Deformer network D
        self.deformer = Deformer_KP(*args, **kwargs)
        
        # Embedding network F
        the_module = importlib.import_module('src.architectures.{}'.format(self.hparams.encoder))
        self.encoder = getattr(the_module, 'get_model')(feat_dim=self.hparams.feat_dim, normal_channel=False)
    
        #TODO: load pretrained Deformer and freeze it 
        # if self.hparams.lambda_deform_triplet > 0 or self.hparams.val_deform:
        import os
        from pytorch_lightning.utilities.cloud_io import load as pl_load

        checkpoint_file = os.path.join(self.hparams.work_dir, self.hparams.deformer_ckpt)
        if os.path.exists(checkpoint_file):
            ckpt = pl_load(checkpoint_file, map_location=lambda storage, loc: storage)  
            utils.log_string('Load deformer model: {}'.format(checkpoint_file))
            self.deformer.load_state_dict(ckpt['state_dict'])

        else:
            utils.log_string("Can't find the deformer ckpt: {}".format(checkpoint_file))
            return 0

        if self.hparams.use_sketch_deformer > 0:
            self.sketch_deformer = Deformer_KP(*args, **kwargs)
            checkpoint_file = os.path.join(self.hparams.work_dir, self.hparams.sketch_deformer_ckpt)
            if os.path.exists(checkpoint_file):
                ckpt = pl_load(checkpoint_file, map_location=lambda storage, loc: storage)  
                utils.log_string('Load deformer model: {}'.format(checkpoint_file))
                self.sketch_deformer.load_state_dict(ckpt['state_dict'])
            else:
                utils.log_string("Can't find the deformer ckpt: {}".format(checkpoint_file))
                return 0


    def prepare_batch(self, batch, mode='test'):
        source_shape = batch['source_shape'] # torch.Size([4, 1024, 3])
        target_shape = batch['target_shape'] # torch.Size([4, 1024, 3])
        
        B, N, C = source_shape.shape       
        B, M, C = target_shape.shape

        if N != self.hparams.num_points: # N can be larger than M
            source_shape = sample_farthest_points(source_shape.transpose(1,2), self.hparams.num_points).transpose(1,2)
        if M != self.hparams.num_points: # N can be larger than M
            target_shape = sample_farthest_points(target_shape.transpose(1,2), self.hparams.num_points).transpose(1,2)

        if mode=='train' and self.hparams.transform:
            source_shape = self.transforms(source_shape)
            target_shape = self.transforms(target_shape)
        if self.hparams.aug_sketch:
            source_shape = d_utils.apply_random_scale_xyz(source_shape)
        if self.hparams.aug_shape:
            target_shape = d_utils.apply_random_scale_xyz(target_shape)

        return source_shape, target_shape

    def _sum_losses(self, losses, names):
        return sum(v for k, v in losses.items() if k in names)
    def forward(self, inputs):
        features = self.encoder(inputs.transpose(1, 2)) #torch.Size([8, 512])
        return features
    def training_step(self, batch, batch_idx):
        # torch.Size([6, 1024, 3])
        source_shape, target_shape = self.prepare_batch(batch, mode='train')
        # from src.fitting_gap import vis
        # save_dir = '/scratch/visualization'
        # fn = os.path.join(save_dir, 'batch.png')
        # vis([source_shape[0].cpu(), target_shape[0].cpu()], fn)

        B, _, _ = source_shape.shape

        # Encode [source_sketch, target_shape]
        inputs = torch.cat([source_shape, target_shape])#torch.Size([12, 1024, 3])
        features = self.encoder(inputs.transpose(1, 2)) #torch.Size([8, 512])
        
        # Losses
        losses = {}

        ## 1. triplet loss from retrieval baseline
        source_feat, target_feat = torch.split(features, [B, B])
        losses['triplet'] = self.crt_tl(source_feat, target_feat) #tensor(0.2779, grad_fn=<MeanBackward0>)


        ## 2. triplet loss using [source_sketch, deformed_sketch, target_shape]
        if self.hparams.lambda_deform_triplet > 0:
            # step 1: deform source sketch towards target shape
            deformed_shape = self.deformer(source_shape, target_shape)['deformed'] # torch.Size([6, 1024, 3])
            deformed_feat =  self.encoder(deformed_shape.transpose(1, 2))
            # step 2: compute deform triplet loss
            source_feat = F.normalize(source_feat, p=2, dim=1)
            target_feat = F.normalize(target_feat, p=2, dim=1)
            deformed_feat = F.normalize(deformed_feat, p=2, dim=1)
            losses['deform_triplet'] = self.hparams.lambda_deform_triplet * self.deform_triplet_loss(target_feat, deformed_feat, source_feat)
            # source_target_dist = (source_feat - target_feat).pow(2).sum(1)
            # deformed_target_dist = (deformed_feat - target_feat).pow(2).sum(1)
            # deform_triplet = F.relu(deformed_target_dist - source_target_dist + self.hparams.deform_margin).mean()
            # losses['deform_triplet']= self.hparams.lambda_deform_triplet * deform_triplet


        loss = self.optimize(losses)
        for item in losses.keys():
            self.log('train/{}'.format(item), losses[item], on_step=False, on_epoch=True, logger=True)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss #tensor(0.2779, grad_fn=<AddBackward0>)

    def optimize(self, losses):
        deformer_optimizer, encoder_optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        deformer_optimizer.zero_grad()
        encoder_optimizer.zero_grad()

        #TODO: freeze KP for given epochs
        # if self.trainer.current_epoch > self.hparams.epochs_deformer:
        
        loss = self._sum_losses(losses, losses.keys())
        loss.backward()
        encoder_optimizer.step()
        # deformer_optimizer.step()
        if self.trainer.is_last_batch: 
            scheduler.step()
        return loss

    def validation_step(self, batch, batch_idx):
        sketch_feat = self.encoder(batch['shape'].transpose(1, 2))
        return sketch_feat

    def validation_epoch_end(self, outputs):
        if not self.trainer.running_sanity_check:
            sketch_features = []
            shape_features = []
            for output in outputs:
                sketch_features.append(output)
            for i, data in enumerate(self.val_shape_loader):
                shape = torch.tensor(data['shape']).cuda()
                feat = self.encoder(shape.transpose(1, 2))
                shape_features.append(feat)
            shape_features = torch.cat(shape_features, 0)
            sketch_features = torch.cat(sketch_features, 0)
            d_feat_z = compute_distance(sketch_features, shape_features, l2=True)
            pair_sort = torch.argsort(d_feat_z, dim=1)

            # Top-k fitting_gap
            K = 5
            for metric in self.val_fitting_gap.keys():
                top_fitting_gap = torch.gather(self.val_fitting_gap[metric], 1, pair_sort[:, :K])
                for k in [1, 5]:
                    self.log(f'val_{metric}@{k}', top_fitting_gap[:, :k].mean() * 100, prog_bar=True)
                    # self.log(f'val_bi_CD_d@{k}', top_fitting_gap[:, :k].mean() * 100, prog_bar=True)               
    
    def test_step(self, batch, batch_idx):
        sketch_feat = self.encoder(batch['shape'].transpose(1, 2))
        return sketch_feat
    def test_epoch_end(self, outputs):
        sketch_features = []
        shape_features = []
        for output in outputs:
            sketch_features.append(output)
        for i, data in enumerate(self.test_shape_loader):
            shape = torch.tensor(data['shape']).cuda()
            feat = self.encoder(shape.transpose(1, 2))
            shape_features.append(feat)
        shape_features = torch.cat(shape_features, 0)
        sketch_features = torch.cat(sketch_features, 0)
        dis_mat = compute_distance(sketch_features, shape_features, l2=True).cpu().numpy()
        self.evaluate(dis_mat)

    def evaluate(self, dis_mat):
        ckpt_name = os.path.basename(self.trainer.tested_ckpt_path)
        if 'last' in ckpt_name:
            monitor = 'last'
        else:
            monitor = ckpt_name.split('=')[0]
        save_path = os.path.join(self.save_dir, '{}_rank_{}_{}.npy'.format('test', self.hparams.test_data, monitor))
        np.save(save_path, dis_mat)

        pair_sort = np.argsort(dis_mat)
        # 1. Top-K Acc
        acc_at_k = compute_acc_at_k(pair_sort)
        for acc, k in zip(acc_at_k, [1, 5, 10]):
            self.log(f'test_{monitor}_acc@{k}', acc * 100, prog_bar=True)               
        
        # 2. Mean CD (retrieved, GT)
        file_path = os.path.join('/vol/vssp/datasets/multiview/3VS/datasets/cache/test_selected_metrics', self.hparams.category,'{}.npy')

        metrics = ['CD','CD_d','bi_CD_d',"F_0.01_d", "bi_F_0.01_d"]
        metrics_dict = {item: np.load(file_path.format(item)) for item in metrics}
        K = 10
        for metric in metrics:
            top_fitting_gap = torch.gather(torch.tensor(metrics_dict[metric]), 1, torch.tensor(pair_sort[:, :K]))
            for k in [1, 5, 10]:
                self.log(f'test_{monitor}_{metric}@{k}', top_fitting_gap[:, :k].mean() * 100, prog_bar=True)

    
    def configure_optimizers(self):
        params = [{"params": self.deformer.parameters()}]
        deformer_optimizer = torch.optim.Adam(params, lr=self.hparams.lr)
        # deformer_optimizer.add_param_group({'params': self.influence_param, 'lr': 10 * self.hparams.lr})
        params = [{"params": self.encoder.parameters()}]
        if self.hparams.optimizer == 'adam':
            encoder_optimizer = torch.optim.Adam(params, lr=self.hparams.lr)
        else:
            encoder_optimizer = torch.optim.SGD(params, lr=self.hparams.lr, momentum=0.9)
        
        scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        return [deformer_optimizer, encoder_optimizer], [scheduler]

class Deformer_Retrieval_fitting_gap(Deformer_Retrieval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # LEN_SOURCES = 702
        import numpy as np
        sigma_path = '/vol/vssp/datasets/multiview/3VS/datasets/cache/sigmas.npy'
        self.source_sigmas = torch.tensor(np.load(sigma_path)).cuda().view(-1, 1)
        # self.source_sigmas = 0.997/3 * self.source_sigmas
        if not self.hparams.fixed_sigma:
            self.source_sigmas = torch.nn.Parameter(self.source_sigmas, requires_grad=True)
            # self.source_sigmas = torch.autograd.Variable(torch.randn(self.source_sigmas.shape, dtype=torch.float).cuda(), requires_grad=True)
        self.KL = torch.nn.KLDivLoss()
        self.num_negs = int(self.hparams.neg_ratio * (self.hparams.batch_size - 1))
        top2_path = '/vol/vssp/datasets/multiview/3VS/datasets/cache/top2.npy'
        self.source_top2 = torch.tensor(np.load(top2_path)).cuda().view(-1, 1)
        self.top2_min = self.source_top2.min()
        if self.hparams.top2_max is None:
            self.top2_max = self.source_top2.max()
        else:
            self.top2_max = self.hparams.top2_max
        self.pdist = torch.nn.PairwiseDistance(p=2)
        if self.hparams.learn_margin:
            self.margin = torch.nn.Parameter(torch.tensor(self.hparams.margin), requires_grad=True)


    def regression_loss(self, embedding_distance, actual_distance, source_index):
        # if not self.hparams.fixed_sigma:
        # obj_sigmas = torch.sigmoid(obj_sigmas)
        obj_sigmas = self.source_sigmas[source_index, :]

        qij = F.softmax(-embedding_distance, dim= -1)

        pij = F.softmax(-torch.div(actual_distance * self.hparams.softmax_weight, obj_sigmas), dim= -1)

        # loss = torch.sum(torch.square(pij-qij), dim=-1)
        if self.hparams.regression_loss == 'KL':
            loss = self.KL(qij.log(), pij)
            return loss
        loss = torch.sum(torch.abs(pij-qij), dim= -1)
        return torch.mean(loss)

    def regression_triplet_loss(self, embedding_distance, actual_distance, obj_sigmas):
        embedding_distance = embedding_distance.pow(2)
        sorted_loss, sorted_indices = torch.sort(actual_distance, dim=1)
        #select the positive to be the closest by fitting loss
        positive_idx = sorted_indices[:, 0].unsqueeze(-1)
        negative_idx = sorted_indices[:, -self.num_negs:]

        negative_actual_distance = torch.gather(actual_distance, 1, negative_idx)
        w = torch.div(negative_actual_distance * self.hparams.softmax_weight, obj_sigmas)
        weighted_margin = self.hparams.margin * w

        #gather corresponding distances
        positive_distances = torch.gather(embedding_distance, 1, positive_idx)
        positive_distances = positive_distances.repeat(1, self.num_negs)

        negative_distances = torch.gather(embedding_distance, 1, negative_idx)
        return torch.mean(F.relu(positive_distances - negative_distances + weighted_margin))
    
    def regression_triplet_loss_v2(self, embedding_distance, actual_distance, obj_sigmas):
        embedding_distance = embedding_distance.pow(2)
        sorted_loss, sorted_indices = torch.sort(actual_distance, dim=1)
        #select the positive to be the closest by fitting loss
        positive_idx = sorted_indices[:, 0].unsqueeze(-1)
        negative_idx = sorted_indices[:, -self.num_negs:]

        negative_actual_distance = torch.gather(actual_distance, 1, negative_idx).pow(2)
        w = torch.div(negative_actual_distance, negative_actual_distance[:, -1].unsqueeze(-1))
        # w = torch.sigmoid(w)
        weighted_margin = self.hparams.margin * w

        #gather corresponding distances
        positive_distances = torch.gather(embedding_distance, 1, positive_idx)
        positive_distances = positive_distances.repeat(1, self.num_negs)

        negative_distances = torch.gather(embedding_distance, 1, negative_idx)
        return torch.mean(F.relu(positive_distances - negative_distances + weighted_margin))

    def regression_triplet_loss_v3(self, embedding_distance, actual_distance, obj_sigmas):
        embedding_distance = embedding_distance.pow(2)
        sorted_loss, sorted_indices = torch.sort(actual_distance, dim=1)
        #select the positive to be the closest by fitting loss
        positive_idx = sorted_indices[:, 0].unsqueeze(-1)
        negative_idx = sorted_indices[:, -self.num_negs:]

        negative_actual_distance = torch.gather(actual_distance, 1, negative_idx)
        w = torch.div(negative_actual_distance, negative_actual_distance[:, -1].unsqueeze(-1))
        # w = torch.sigmoid(w)
        weighted_margin = self.hparams.margin * w

        #gather corresponding distances
        positive_distances = torch.gather(embedding_distance, 1, positive_idx)
        positive_distances = positive_distances.repeat(1, self.num_negs)

        negative_distances = torch.gather(embedding_distance, 1, negative_idx)
        return torch.mean(F.relu(positive_distances - negative_distances + weighted_margin))

    def regression_triplet_loss_v4(self, embedding_distance, actual_distance, obj_sigmas):
        embedding_distance = embedding_distance.pow(2)
        sorted_loss, sorted_indices = torch.sort(actual_distance, dim=1)
        #select the positive to be the closest by fitting loss
        positive_idx = sorted_indices[:, 0].unsqueeze(-1)
        negative_idx = sorted_indices[:, -self.num_negs:]

        negative_actual_distance = torch.gather(actual_distance, 1, negative_idx)
        w = torch.div(negative_actual_distance, negative_actual_distance[:, -1].unsqueeze(-1))
        w = torch.exp(w)
        weighted_margin = self.hparams.margin * w

        #gather corresponding distances
        positive_distances = torch.gather(embedding_distance, 1, positive_idx)
        positive_distances = positive_distances.repeat(1, self.num_negs)

        negative_distances = torch.gather(embedding_distance, 1, negative_idx)
        return torch.mean(F.relu(positive_distances - negative_distances + weighted_margin))

    def regression_triplet_loss_v5(self, embedding_distance, actual_distance, obj_sigmas):
        embedding_distance = embedding_distance.pow(2)
        sorted_loss, sorted_indices = torch.sort(actual_distance, dim=1)
        #select the positive to be the closest by fitting loss
        positive_idx = sorted_indices[:, 0].unsqueeze(-1)
        negative_idx = sorted_indices[:, -self.num_negs:]

        negative_actual_distance = torch.gather(actual_distance, 1, negative_idx)
        w = torch.div(negative_actual_distance, obj_sigmas)
        w = torch.exp(w)
        weighted_margin = self.hparams.margin * w

        #gather corresponding distances
        positive_distances = torch.gather(embedding_distance, 1, positive_idx)
        positive_distances = positive_distances.repeat(1, self.num_negs)

        negative_distances = torch.gather(embedding_distance, 1, negative_idx)
        return torch.mean(F.relu(positive_distances - negative_distances + weighted_margin))

    def regression_triplet_loss_v6(self, embedding_distance, actual_distance, obj_sigmas):
        embedding_distance = embedding_distance.pow(2)
        sorted_loss, sorted_indices = torch.sort(actual_distance, dim=1)
        #select the positive to be the closest by fitting loss
        positive_idx = sorted_indices[:, 0].unsqueeze(-1)
        negative_idx = sorted_indices[:, -self.num_negs:]

        negative_actual_distance = torch.gather(actual_distance, 1, negative_idx)
        w = torch.div(negative_actual_distance, negative_actual_distance[:, -1].unsqueeze(-1))
        w = torch.exp(w) - 1
        weighted_margin = self.hparams.margin * w

        #gather corresponding distances
        positive_distances = torch.gather(embedding_distance, 1, positive_idx)
        positive_distances = positive_distances.repeat(1, self.num_negs)

        negative_distances = torch.gather(embedding_distance, 1, negative_idx)
        return torch.mean(F.relu(positive_distances - negative_distances + weighted_margin))
    def regression_triplet_loss_v7(self, embedding_distance, actual_distance, source_index, eps=1e-8):
        embedding_distance = embedding_distance.pow(2)
        sorted_loss, sorted_indices = torch.sort(actual_distance, dim=1)
        #select the positive to be the closest by fitting loss
        positive_idx = sorted_indices[:, 0].unsqueeze(-1)
        negative_idx = sorted_indices[:, -self.num_negs:]

        negative_actual_distance = torch.gather(actual_distance, 1, negative_idx)
        max_val = negative_actual_distance[:, -1] + eps
        w = torch.div(negative_actual_distance, max_val.unsqueeze(-1)) 
        # w = torch.exp(w) - 1
        # weighted_margin = self.hparams.margin + w * (self.hparams.margin_max - self.hparams.margin)
        weighted_margin = 0.3 + w * (1.2 - 0.3)

        #gather corresponding distances
        positive_distances = torch.gather(embedding_distance, 1, positive_idx)
        positive_distances = positive_distances.repeat(1, self.num_negs)

        negative_distances = torch.gather(embedding_distance, 1, negative_idx)
        return torch.mean(F.relu(positive_distances - negative_distances + weighted_margin))

    def regression_triplet_loss_v8(self, embedding_distance, actual_distance, obj_sigmas, eps=1e-8):
        embedding_distance = embedding_distance.pow(2)
        sorted_loss, sorted_indices = torch.sort(actual_distance, dim=1)
        #select the positive to be the closest by fitting loss
        positive_idx = sorted_indices[:, 0].unsqueeze(-1)
        negative_idx = sorted_indices[:, -self.num_negs:]

        negative_actual_distance = torch.gather(actual_distance, 1, negative_idx)
        max_val = negative_actual_distance[:, -1] + eps
        w = torch.div(negative_actual_distance, max_val.unsqueeze(-1)) 
        w = (torch.exp(w) - 1)/(math.e - 1)
        weighted_margin = self.hparams.margin + w * (self.hparams.margin_max - self.hparams.margin)

        #gather corresponding distances
        positive_distances = torch.gather(embedding_distance, 1, positive_idx)
        positive_distances = positive_distances.repeat(1, self.num_negs)

        negative_distances = torch.gather(embedding_distance, 1, negative_idx)
        return torch.mean(F.relu(positive_distances - negative_distances + weighted_margin))
    
    def regression_triplet_loss_v9(self, embedding_distance, actual_distance, source_index, eps=1e-8):
        obj_top2 = self.source_top2[source_index, :]

        embedding_distance = embedding_distance.pow(2)
        sorted_loss, sorted_indices = torch.sort(actual_distance, dim=1)
        #select the positive to be the closest by fitting loss
        positive_idx = sorted_indices[:, 0].unsqueeze(-1)
        negative_idx = sorted_indices[:, -self.num_negs:]

        negative_actual_distance = torch.gather(actual_distance, 1, negative_idx)
        max = torch.max(negative_actual_distance[:, -1].unsqueeze(-1), obj_top2 + eps)
        min = torch.min(negative_actual_distance[:, 0].unsqueeze(-1), obj_top2)
        w = torch.div(negative_actual_distance - min, (max - min))
        # w = torch.exp(w) - 1
        weighted_margin = self.hparams.margin + w * (self.hparams.margin_max - self.hparams.margin)

        #gather corresponding distances
        positive_distances = torch.gather(embedding_distance, 1, positive_idx)
        positive_distances = positive_distances.repeat(1, self.num_negs)

        negative_distances = torch.gather(embedding_distance, 1, negative_idx)
        return torch.mean(F.relu(positive_distances - negative_distances + weighted_margin))

    def regression_triplet_loss_v10(self, embedding_distance, actual_distance, source_index):

        embedding_distance = embedding_distance.pow(2)
        sorted_loss, sorted_indices = torch.sort(actual_distance, dim=1)
        #select the positive to be the closest by fitting loss
        positive_idx = sorted_indices[:, 0].unsqueeze(-1)
        negative_idx = sorted_indices[:, -self.num_negs:]

        negative_actual_distance = torch.gather(actual_distance, 1, negative_idx)
        # max = torch.max(negative_actual_distance[:, -1].unsqueeze(-1),  self.top2_max)
        # min = torch.min(negative_actual_distance[:, 0].unsqueeze(-1), self.top2_min)
        w = torch.div(negative_actual_distance - self.top2_min, (self.top2_max - self.top2_min))
        # w = torch.exp(w) - 1
        weighted_margin = self.hparams.margin + w * (self.hparams.margin_max - self.hparams.margin)
        weighted_margin = torch.where(weighted_margin > 4., torch.ones_like(weighted_margin) * 4., weighted_margin)
        #gather corresponding distances
        positive_distances = torch.gather(embedding_distance, 1, positive_idx)
        positive_distances = positive_distances.repeat(1, self.num_negs)

        negative_distances = torch.gather(embedding_distance, 1, negative_idx)
        return torch.mean(F.relu(positive_distances - negative_distances + weighted_margin))

    def regression_triplet_loss_v11(self, embedding_distance, actual_distance, obj_sigmas, eps=1e-8):
        embedding_distance = embedding_distance.pow(2)
        sorted_loss, sorted_indices = torch.sort(actual_distance, dim=1)
        #select the positive to be the closest by fitting loss
        positive_idx = sorted_indices[:, 0].unsqueeze(-1)
        negative_idx = sorted_indices[:, -self.num_negs:]

        negative_actual_distance = torch.gather(actual_distance, 1, negative_idx)
        max_val = negative_actual_distance[:, -1] + eps
        w = torch.div(negative_actual_distance, max_val.unsqueeze(-1)) 
        # w = torch.exp(w) - 1
        weighted_margin = self.margin + w * self.hparams.margin_range

        #gather corresponding distances
        positive_distances = torch.gather(embedding_distance, 1, positive_idx)
        positive_distances = positive_distances.repeat(1, self.num_negs)

        negative_distances = torch.gather(embedding_distance, 1, negative_idx)
        return torch.mean(F.relu(positive_distances - negative_distances + weighted_margin))

    def regression_triplet_loss_v12(self, embedding_distance, actual_distance, source_index, eps=1e-8):
        obj_top2 = self.source_top2[source_index, :]
        obj_max = self.source_sigmas[source_index, :]

        embedding_distance = embedding_distance.pow(2)
        sorted_loss, sorted_indices = torch.sort(actual_distance, dim=1)
        #select the positive to be the closest by fitting loss
        positive_idx = sorted_indices[:, 0].unsqueeze(-1)
        negative_idx = sorted_indices[:, -self.num_negs:]

        negative_actual_distance = torch.gather(actual_distance, 1, negative_idx)
        max = torch.max(negative_actual_distance[:, -1].unsqueeze(-1), obj_max)
        min = torch.min(negative_actual_distance[:, 0].unsqueeze(-1), obj_top2)
        w = torch.div(negative_actual_distance - min, (max - min))
        # w = torch.exp(w) - 1
        weighted_margin = self.hparams.margin + w * (self.hparams.margin_max - self.hparams.margin)

        #gather corresponding distances
        positive_distances = torch.gather(embedding_distance, 1, positive_idx)
        positive_distances = positive_distances.repeat(1, self.num_negs)

        negative_distances = torch.gather(embedding_distance, 1, negative_idx)
        return torch.mean(F.relu(positive_distances - negative_distances + weighted_margin))

    def regression_triplet_loss_v13(self, embedding_distance, actual_distance, source_index, eps=1e-8):

        embedding_distance = embedding_distance.pow(2)
        sorted_loss, sorted_indices = torch.sort(actual_distance, dim=1)
        #select the positive to be the closest by fitting loss
        positive_idx = sorted_indices[:, 0].unsqueeze(-1)
        negative_idx = sorted_indices[:, -self.num_negs:]

        negative_actual_distance = torch.gather(actual_distance, 1, negative_idx)
        max = torch.max(negative_actual_distance[:, -1].unsqueeze(-1), torch.tensor(eps).cuda())
        min = negative_actual_distance[:, 0].unsqueeze(-1)
        w = torch.div(negative_actual_distance - min, (max - min))
        # w = torch.exp(w) - 1
        weighted_margin = self.hparams.margin + w * (self.hparams.margin_max - self.hparams.margin)

        #gather corresponding distances
        positive_distances = torch.gather(embedding_distance, 1, positive_idx)
        positive_distances = positive_distances.repeat(1, self.num_negs)

        negative_distances = torch.gather(embedding_distance, 1, negative_idx)
        return torch.mean(F.relu(positive_distances - negative_distances + weighted_margin))

    def triplet_loss(self, embedding_distance, actual_distance, margin):
        embedding_distance = embedding_distance.pow(2)
        sorted_loss, sorted_indices = torch.sort(actual_distance, dim=1)
        #select the positive to be the closest by fitting loss
        positive_idx = sorted_indices[:, 0].unsqueeze(-1)
        negative_idx = sorted_indices[:, -self.num_negs:]

        #gather corresponding distances
        positive_distances = torch.gather(embedding_distance, 1, positive_idx)
        positive_distances = positive_distances.repeat(1, self.num_negs)

        negative_distances = torch.gather(embedding_distance, 1, negative_idx)
        return torch.mean(F.relu(positive_distances - negative_distances + margin))

    def triplet_loss_threshold(self):
        pass

    def deform_sketch_loss_v1(self, sketch_feat, shape_feat, sketch, shape, actual_distance, source_index):
        # d(D(s)+, t+) < d(D(s)-, t-) 
        B = sketch.shape[0]
        sorted_loss, sorted_indices = torch.sort(actual_distance, dim=1)
        negative_idx = sorted_indices[:, -1]
        negative_shapes = shape[negative_idx, :]
        target_shape = torch.cat([shape, negative_shapes])
        source_shape = torch.cat([sketch, sketch])
        if self.hparams.use_sketch_deformer:
            deformed_shape = self.sketch_deformer(source_shape, target_shape)['deformed'] # torch.Size([6, 1024, 3])
        else:
            deformed_shape = self.deformer(source_shape, target_shape)['deformed'] # torch.Size([6, 1024, 3])
        feat = self.encoder(deformed_shape.transpose(1, 2).detach().data) #torch.Size([8, 512])
        feat = F.normalize(feat, p=2, dim=1)

        p_feat, n_feat = torch.split(feat, [B, B])

        ap_deformed = self.pdist(p_feat, shape_feat).pow(2)
        an_deformed = self.pdist(n_feat, shape_feat[negative_idx, :]).pow(2)
        return torch.mean(F.relu(ap_deformed - an_deformed + self.hparams.deform_sketch_margin))



    def deform_sketch_loss_v2(self, sketch_feat, shape_feat, sketch, shape, actual_distance, source_index):
        # d(D(s)+, t+) < d(D(s)+, t-), visualization of deformation

        if self.hparams.use_sketch_deformer:
            deformed_shape = self.sketch_deformer(sketch, shape)['deformed'] # torch.Size([6, 1024, 3])
        else:
            deformed_shape = self.deformer(sketch, shape)['deformed'] # torch.Size([6, 1024, 3])
        deformed_feat = self.encoder(deformed_shape.transpose(1, 2).detach().data) #torch.Size([8, 512])
        deformed_feat = F.normalize(deformed_feat, p=2, dim=1)

        # d(D(s)+, t)
        embedding_distance = torch.cdist(deformed_feat, shape_feat)

        if self.hparams.loss_type == 'regression':
            return self.regression_loss(embedding_distance, actual_distance)
        elif self.hparams.loss_type == 'triplet':
            return self.triplet_loss(embedding_distance, actual_distance, self.hparams.deform_sketch_margin)
        elif self.hparams.loss_type == 'regression_triplet':
            method_to_call = getattr(self, self.hparams.regression_triplet_loss)
            return method_to_call(embedding_distance, actual_distance, source_index)
        else:
            NotImplementedError

    def deform_sketch_loss_v3(self, sketch_feat, shape_feat, sketch, shape, actual_distance, source_index):
        # d(D(s)+, t+) = d(s, t+) -> d(D(s)+ , s) decrease
        if self.hparams.use_sketch_deformer:
            deformed_shape = self.sketch_deformer(sketch, shape)['deformed'] # torch.Size([6, 1024, 3])
        else:
            deformed_shape = self.deformer(sketch, shape)['deformed'] # torch.Size([6, 1024, 3])
        deformed_feat = self.encoder(deformed_shape.transpose(1, 2).detach().data) #torch.Size([8, 512])
        deformed_feat = F.normalize(deformed_feat, p=2, dim=1)
        embedding_distance = self.pdist(deformed_feat, sketch_feat)
        return embedding_distance.mean()


    def deform_sketch_loss_v4(self, sketch_feat, shape_feat, sketch, shape, actual_distance, source_index):
        # d(D(s)+, t+) < d(s, t+)
        if self.hparams.use_sketch_deformer:
            deformed_shape = self.sketch_deformer(sketch, shape)['deformed'] # torch.Size([6, 1024, 3])
        else:
            deformed_shape = self.deformer(sketch, shape)['deformed'] # torch.Size([6, 1024, 3])
        deformed_feat = self.encoder(deformed_shape.transpose(1, 2).detach().data) #torch.Size([8, 512])
        deformed_feat = F.normalize(deformed_feat, p=2, dim=1)
        ap_deformed = self.pdist(deformed_feat, shape_feat).pow(2)
        ap = self.pdist(sketch_feat, shape_feat).pow(2)
        return torch.mean(F.relu(ap_deformed - ap + self.hparams.deform_sketch_margin))

    def deform_sketch_loss_v5(self, sketch_feat, shape_feat, sketch, shape, actual_distance, source_index):
        # d(D(s)+, t+) = d(s, t+) 
        if self.hparams.use_sketch_deformer:
            deformed_shape = self.sketch_deformer(sketch, shape)['deformed'] # torch.Size([6, 1024, 3])
        else:
            deformed_shape = self.deformer(sketch, shape)['deformed'] # torch.Size([6, 1024, 3])
        deformed_feat = self.encoder(deformed_shape.transpose(1, 2).detach().data) #torch.Size([8, 512])
        deformed_feat = F.normalize(deformed_feat, p=2, dim=1)
        ap_deformed = self.pdist(deformed_feat, shape_feat).pow(2)
        ap = self.pdist(sketch_feat, shape_feat).pow(2)
        return torch.mean(torch.abs(ap_deformed - ap))



    def training_step(self, batch, batch_idx):

        # torch.Size([6, 1024, 3])
        sketch, shape = self.prepare_batch(batch, mode='train')
        B, _, _ = shape.shape

        source_index = batch['index']    

        # Encode [source_sketch, target_shape]
        inputs = torch.cat([sketch, shape])#torch.Size([12, 1024, 3])
        features = self.encoder(inputs.transpose(1, 2)) #torch.Size([8, 512])
        
        # Losses
        losses = {}

        ## 1. embedding_distance
        if self.hparams.L2_norm:
            features = F.normalize(features, p=2, dim=1)
        sketch_feat, shape_feat = torch.split(features, [B, B])
        embedding_distance = torch.cdist(sketch_feat, shape_feat, p=2)
        ## 2. regression loss using [source_sketch, deformed_sketch, target_shape]
        # step 1: deform source shape towards target shape
        source_shape = torch.repeat_interleave(shape, torch.ones(B, dtype=torch.long).cuda()*B, dim=0)
        target_shape = shape.repeat(B, 1, 1)

        if self.hparams.fitting_gap == 'd1':
            deformed_shape = self.deformer(source_shape, target_shape)['deformed'] # torch.Size([6, 1024, 3])
            # step 2: compute distribution
            actual_distance = pytorch3d.loss.chamfer_distance(
                deformed_shape, target_shape, batch_reduction=None)[0].view([B, B]).detach().data
        elif self.hparams.fitting_gap == 'd2':
            deformed_shape = self.deformer(target_shape, source_shape)['deformed'] # torch.Size([6, 1024, 3])
            # step 2: compute distribution
            actual_distance = pytorch3d.loss.chamfer_distance(
                deformed_shape, source_shape, batch_reduction=None)[0].view([B, B]).detach().data

        elif self.hparams.fitting_gap == 'symmetric':
            deformed_shape = self.deformer(source_shape, target_shape)['deformed'] # torch.Size([6, 1024, 3])
            distance_1 = pytorch3d.loss.chamfer_distance(
                deformed_shape, target_shape, batch_reduction=None)[0].view([B, B]).detach().data
            deformed_shape = self.deformer(target_shape, source_shape)['deformed'] # torch.Size([6, 1024, 3])
            distance_2 = pytorch3d.loss.chamfer_distance(
                deformed_shape, source_shape, batch_reduction=None)[0].view([B, B]).detach().data
            actual_distance = (distance_1 + distance_2) / 2

        else: # Chamfer distance
            actual_distance = pytorch3d.loss.chamfer_distance(
                source_shape, target_shape, batch_reduction=None)[0].view([B, B]).detach().data

        if self.hparams.loss_type == 'regression':
            losses['regression_loss'] = self.regression_loss(embedding_distance, actual_distance, source_index)
        elif self.hparams.loss_type == 'triplet':
            losses['triplet_loss'] = self.triplet_loss(embedding_distance, actual_distance, self.hparams.margin)
        elif self.hparams.loss_type == 'regression_triplet':
            method_to_call = getattr(self, self.hparams.regression_triplet_loss)
            losses['regression_triplet_loss'] = method_to_call(embedding_distance, actual_distance, source_index)
        else:
            NotImplementedError
        
        if self.hparams.lambda_deform_sketch > 0:
            method_to_call = getattr(self, self.hparams.deform_sketch_loss)
            losses['deform_sketch'] = self.hparams.lambda_deform_sketch * method_to_call(sketch_feat, shape_feat, sketch, shape, actual_distance, source_index)

        loss = self.optimize(losses)
        for item in losses.keys():
            self.log('train/{}'.format(item), losses[item], on_step=False, on_epoch=True, logger=True)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss #tensor(0.2779, grad_fn=<AddBackward0>)

    def configure_optimizers(self):
        # params = [{"params": self.deformer.parameters()}]
        # deformer_optimizer = torch.optim.Adam(params, lr=self.hparams.lr)
        # Freeze deformer:
        for param in self.deformer.parameters():
            param.requires_grad = False
        if self.hparams.use_sketch_deformer:
            for param in self.sketch_deformer.parameters():
                param.requires_grad = False

        # deformer_optimizer.add_param_group({'params': self.influence_param, 'lr': 10 * self.hparams.lr})
        params = [{"params": self.encoder.parameters()}]
        if self.hparams.optimizer == 'adam':
            encoder_optimizer = torch.optim.Adam(params, lr=self.hparams.lr)
        else:
            encoder_optimizer = torch.optim.SGD(params, lr=self.hparams.lr, momentum=0.9)
        
        if self.hparams.loss_type == 'regression' and not self.hparams.fixed_sigma:
            encoder_optimizer.add_param_group({"params": self.source_sigmas, "lr": self.hparams.sigma_lr})
        if self.hparams.learn_margin:
            encoder_optimizer.add_param_group({"params": self.margin, "lr": self.hparams.sigma_lr})

        scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        return [encoder_optimizer], [scheduler]
    
    def optimize(self, losses):
        encoder_optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        encoder_optimizer.zero_grad()
        loss = self._sum_losses(losses, losses.keys())
        loss.backward()
        encoder_optimizer.step()
        # deformer_optimizer.step()
        if self.trainer.is_last_batch: 
            scheduler.step()
        return loss


class Deformer_Retrieval_fitting_gap_egocentric(Deformer_Retrieval_fitting_gap):
    def _build_model(self, *args, **kwargs):
        # Deformer network D
        self.deformer = Deformer_KP(*args, **kwargs)
        
        # Embedding network F
        the_module = importlib.import_module('src.architectures.{}'.format(self.hparams.encoder))
        self.encoder = getattr(the_module, 'get_model_egocentric')(feat_dim=self.hparams.feat_dim, normal_channel=False)
    
        #TODO: load pretrained Deformer and freeze it 
        # if self.hparams.lambda_deform_triplet > 0 or self.hparams.val_deform:
        import os
        from pytorch_lightning.utilities.cloud_io import load as pl_load

        checkpoint_file = os.path.join(self.hparams.work_dir, self.hparams.deformer_ckpt)
        if os.path.exists(checkpoint_file):
            ckpt = pl_load(checkpoint_file, map_location=lambda storage, loc: storage)  
            utils.log_string('Load deformer model: {}'.format(checkpoint_file))
            self.deformer.load_state_dict(ckpt['state_dict'])

        else:
            utils.log_string("Can't find the deformer ckpt: {}".format(checkpoint_file))
            return 0

        if self.hparams.use_sketch_deformer > 0:
            self.sketch_deformer = Deformer_KP(*args, **kwargs)
            checkpoint_file = os.path.join(self.hparams.work_dir, self.hparams.sketch_deformer_ckpt)
            if os.path.exists(checkpoint_file):
                ckpt = pl_load(checkpoint_file, map_location=lambda storage, loc: storage)  
                utils.log_string('Load deformer model: {}'.format(checkpoint_file))
                self.sketch_deformer.load_state_dict(ckpt['state_dict'])
            else:
                utils.log_string("Can't find the deformer ckpt: {}".format(checkpoint_file))
                return 0

    def training_step(self, batch, batch_idx):

        # torch.Size([6, 1024, 3])
        sketch, shape = self.prepare_batch(batch, mode='train')
        B, _, _ = shape.shape

        source_index = batch['index']    

        # Encode [source_sketch, target_shape]
        # inputs = torch.cat([sketch, shape])#torch.Size([12, 1024, 3])
        sketch_feat = self.encoder(sketch.transpose(1, 2)) #torch.Size([8, 512])
        shape_feat, matrix = self.encoder(shape.transpose(1, 2), matrix=True) #torch.Size([8, 512])

        # Losses
        losses = {}
        ## 1. embedding_distance
        if self.hparams.L2_norm:
            sketch_feat = F.normalize(sketch_feat, p=2, dim=1)
            shape_feat = F.normalize(shape_feat, p=2, dim=1)

        # sketch_feat, shape_feat = torch.split(features, [B, B])
        # embedding_distance = mahalanobis(sketch_feat, shape_feat, matrix)

        embedding_distance = egocentric(sketch_feat, shape_feat, matrix)
        ## 2. regression loss using [source_sketch, deformed_sketch, target_shape]
        # step 1: deform source shape towards target shape

        source_shape = torch.repeat_interleave(shape, torch.ones(B, dtype=torch.long).cuda()*B, dim=0)
        target_shape = shape.repeat(B, 1, 1)

        deformed_shape = self.deformer(source_shape, target_shape)['deformed'] # torch.Size([6, 1024, 3])
        # step 2: compute distribution

        # obj_sigmas = torch.gather(self.source_sigmas, 0, source_index.unsqueeze(-1).repeat(1,self.source_sigmas.shape[-1]))
        actual_distance = pytorch3d.loss.chamfer_distance(
            deformed_shape, target_shape, batch_reduction=None)[0].view([B, B]).detach().data

        if self.hparams.loss_type == 'regression':
            losses['regression_loss'] = self.regression_loss(embedding_distance, actual_distance, source_index)
        elif self.hparams.loss_type == 'triplet':
            losses['triplet_loss'] = self.triplet_loss(embedding_distance, actual_distance, self.hparams.margin)
        elif self.hparams.loss_type == 'regression_triplet':
            method_to_call = getattr(self, self.hparams.regression_triplet_loss)
            losses['regression_triplet_loss'] = method_to_call(embedding_distance, actual_distance, source_index)
        else:
            NotImplementedError
        
        if self.hparams.lambda_deform_sketch > 0:
            method_to_call = getattr(self, self.hparams.deform_sketch_loss)
            losses['deform_sketch'] = self.hparams.lambda_deform_sketch * method_to_call(sketch_feat, shape_feat, sketch, shape, actual_distance, source_index)

        loss = self.optimize(losses)
        for item in losses.keys():
            self.log('train/{}'.format(item), losses[item], on_step=False, on_epoch=True, logger=True)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss #tensor(0.2779, grad_fn=<AddBackward0>)

    def validation_epoch_end(self, outputs):
        if not self.trainer.running_sanity_check:
            sketch_features = []
            shape_features = []
            matrixes = []
            for output in outputs:
                sketch_features.append(output)

            for i, data in enumerate(self.val_shape_loader):
                shape = torch.tensor(data['shape']).cuda()
                feat, matrix = self.encoder(shape.transpose(1, 2), matrix=True) 
                shape_features.append(feat)
                matrixes.append(matrix)
            shape_features = torch.cat(shape_features, 0)
            sketch_features = torch.cat(sketch_features, 0)
            matrixes = torch.cat(matrixes, 0)
            if self.hparams.L2_norm:
                sketch_features = F.normalize(sketch_features, p=2, dim=1)
                shape_features = F.normalize(shape_features, p=2, dim=1)

            embedding_distance = egocentric(sketch_features, shape_features, matrixes)
            pair_sort = torch.argsort(embedding_distance, dim=1)
            # Top-k fitting_gap
            K = 5
            for metric in self.val_fitting_gap.keys():
                top_fitting_gap = torch.gather(self.val_fitting_gap[metric], 1, pair_sort[:, :K])
                for k in [1, 5]:
                    self.log(f'val_{metric}@{k}', top_fitting_gap[:, :k].mean() * 100, prog_bar=True)

    def test_epoch_end(self, outputs):
        sketch_features = []
        shape_features = []
        matrixes = []

        for output in outputs:
            sketch_features.append(output)
        for i, data in enumerate(self.test_shape_loader):
            shape = torch.tensor(data['shape']).cuda()
            feat, matrix = self.encoder(shape.transpose(1, 2), matrix=True) 
            shape_features.append(feat)
            matrixes.append(matrix)
        shape_features = torch.cat(shape_features, 0)
        sketch_features = torch.cat(sketch_features, 0)
        matrixes = torch.cat(matrixes, 0)
        if self.hparams.L2_norm:
            sketch_features = F.normalize(sketch_features, p=2, dim=1)
            shape_features = F.normalize(shape_features, p=2, dim=1)

        dis_mat = egocentric_iter(sketch_features, shape_features, matrixes).cpu().numpy()
        self.evaluate(dis_mat)
