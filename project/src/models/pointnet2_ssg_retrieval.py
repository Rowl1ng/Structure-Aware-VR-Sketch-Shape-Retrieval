import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_sched
# from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from src.models.pointnet2_utils import PointNetSetAbstraction
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms


# from pointnet2.data.ModelNet40Loader import ModelNet40Cls

from src.datamodules.TripletSampler import AllNegativeTripletSelector
from src.utils.custom_loss import OnlineTripletLoss
from src.utils.evaluation import compute_distance, compute_acc_at_k
def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(lr_sched.LambdaLR):
    def __init__(self, optimizer, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(type(model)._name_)
            )
        self.optimizer = optimizer
        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def state_dict(self):
        return dict(last_epoch=self.last_epoch)

    def load_state_dict(self, state):
        self.last_epoch = state["last_epoch"]
        self.step(self.last_epoch)


lr_clip = 1e-5
bnm_clip = 1e-2


class PointNet2RetrievalSSG(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.crt_tl = OnlineTripletLoss(self.hparams.margin, AllNegativeTripletSelector(symmetric=self.hparams.symmetric))

        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        in_channel = 0
        self.SA_modules.append(
            PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        )
        self.SA_modules.append(
            PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        )
        self.SA_modules.append(
            PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, self.hparams.feat_dim)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, xyz):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        # xyz, features = self._break_up_pc(pointcloud)
        xyz = xyz.transpose(2, 1)
        features = None
        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))

    def training_step(self, batch, batch_idx):
        shape = batch['shape']
        sketch = batch['sketch']
        labels = batch['label']
        pc = torch.cat([sketch, shape])
        feat = self.forward(pc)
        loss = self.crt_tl(feat)

        # loss = F.cross_entropy(logits, labels)
        # with torch.no_grad():
        #     acc = (torch.argmax(logits, dim=1) == labels).float().mean()

        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        shape = batch['shape']
        sketch = batch['sketch']
        pc = torch.cat([sketch, shape])
        feat = self.forward(pc)
        mini_batch = shape.shape[0]
        sketch_feat, shape_feat = torch.split(feat, [mini_batch, mini_batch], dim=0)

        return dict(sketch_feat=sketch_feat, shape_feat=shape_feat)

    def validation_epoch_end(self, outputs):
        if not self.trainer.running_sanity_check:
            sketch_features = []
            shape_features = []
            for output in outputs:
                sketch_features.append(output['sketch_feat'].data.cpu())
                shape_features.append(output['shape_feat'].data.cpu())

            shape_features = torch.cat(shape_features, 0).numpy()
            sketch_features = torch.cat(sketch_features, 0).numpy()
            d_feat_z = compute_distance(sketch_features.copy(), shape_features.copy(), l2=True)
            acc_at_k_feat_z = compute_acc_at_k(d_feat_z)
            self.log('val_acc', acc_at_k_feat_z[0], prog_bar=True)
            self.log('val/top1', acc_at_k_feat_z[0], prog_bar=True)
            self.log('val/top5', acc_at_k_feat_z[1], prog_bar=True)
            self.log('val/top10', acc_at_k_feat_z[2], prog_bar=True)

        # tensorboard_logs = {'val_loss': avg_loss, 'val_acc': val_acc, 'step': self.current_epoch}
        # return {'log': tensorboard_logs}
        # print(self.logger)
        # quit()
        #
        # self.logger.add_scalar('val/top-1', acc_at_k_feat_z[0], self.current_epoch)

        # return dict(val_acc=acc_at_k_feat_z[0])

    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.hparams.lr_decay
            ** (
                int(
                    self.global_step
                    * self.hparams.batch_size
                    / self.hparams.decay_step
                )
            ),
            lr_clip / self.hparams.lr,
        )
        bn_lbmd = lambda _: max(
            self.hparams.bn_momentum
            * self.hparams.bnm_decay
            ** (
                int(
                    self.global_step
                    * self.hparams.batch_size
                    / self.hparams.decay_step
                )
            ),
            bnm_clip,
        )

        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
        else:
            NotImplementedError

        if self.hparams.schedular == 'lambda':
            lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
            bnm_scheduler = BNMomentumScheduler(optimizer, self, bn_lambda=bn_lbmd)
            return [optimizer], [lr_scheduler, bnm_scheduler]
        elif self.hparams.schedular == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)

            return [optimizer], [scheduler]
        else:
            NotImplementedError





