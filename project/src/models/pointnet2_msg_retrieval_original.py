import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG

from src.models.pointnet2_ssg_retrieval import PointNet2RetrievalSSG
from src.models.pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction

class PointNet2RetrievalMSG_Original(PointNet2RetrievalSSG):
    def _build_model(self):
        super()._build_model()

        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], 0, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, self.hparams.feat_dim)

    def forward(self, xyz):
        xyz = xyz.transpose(2, 1)
        B, _, _ = xyz.shape
        norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024) # [bs, 512]
        feat = self.fc1(x)
        return feat