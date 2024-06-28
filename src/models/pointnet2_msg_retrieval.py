import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG
from src.models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from src.models.pointnet2_ssg_retrieval import PointNet2RetrievalSSG


class PointNet2RetrievalMSG(PointNet2RetrievalSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        in_channel = 0
        self.SA_modules.append(
            PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
                                      [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        )
        self.SA_modules.append(
            PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        )
        self.SA_modules.append(
            PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(1024, self.hparams.feat_dim)
        )


    def _build_model_v2(self):
        super()._build_model()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[[0, 32, 32, 64], [0, 64, 64, 128], [0, 64, 96, 128]],
                use_xyz=self.hparams.use_xyz,
            )
        )

        input_channels = 64 + 128 + 128
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=self.hparams.use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[128 + 256 + 256, 256, 512, 1024],
                use_xyz=self.hparams.use_xyz,
            )
        )
