import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools



# Modified from: https://github.com/yongheng1991/3D-point-capsule-networks/blob/master/models/pointcapsnet_ae.py

class PointGenCon(nn.Module):
    def __init__(self, nlatent=128, bottleneck_size=0, outdim=3, config=None):
        
        self.bottleneck_size = nlatent if bottleneck_size == 0 else bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(nlatent, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, int(self.bottleneck_size/2), 1)
        self.conv3 = torch.nn.Conv1d(int(self.bottleneck_size/2), int(self.bottleneck_size/4), 1)
        self.conv4 = torch.nn.Conv1d(int(self.bottleneck_size/4), outdim, 1)
        self.th = torch.nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(int(self.bottleneck_size/2))
        self.bn3 = torch.nn.BatchNorm1d(int(self.bottleneck_size/4))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x

class KpDecoder(nn.Module):
    def __init__(self, num_kp, desc_dim, num_points, config, grid_dim=2):
        super(KpDecoder, self).__init__()
        grid_dim = config.grid_dim
        self.grid_dim = grid_dim 
        self.num_kp = num_kp
        self.num_points = num_points
        self.config = config

        if config.decoder_grid == "learnable":
            grid = torch.nn.Parameter(
                torch.FloatTensor(1, num_kp, num_points//num_kp, grid_dim))
            grid.data.uniform_(0,1)
            self.register_parameter("grid", grid)
            self.grid = grid
        elif config.decoder_grid == "nonlearnable":
            self.grid = None
        else:
            self.grid = None
        
        self.decoder = nn.ModuleList(
            [PointGenCon(nlatent=desc_dim+grid_dim, bottleneck_size=config.decoder_bottleneck_size, outdim=config.indim, config=config) for i in range(0, num_kp)])
        
    def forward(self, x, return_splits=False):
        """ 
        x: B*num_kp*desc_dim
        out: B*num_pts*3
        """
        batch_size = x.shape[0]
        t_dim = self.config.indim
        
        if self.config.pose_code in ["nl-noR_T"]:
            t = x[:, :, :t_dim]
            x = x[:, :, t_dim:]

        outs = []
        if self.grid is None:
            # non-learnable random grid
            rand_grid = torch.FloatTensor(
                x.size(0), self.num_kp, self.num_points//self.num_kp,
                self.grid_dim).to(device=x.device)
            rand_grid.data.uniform_(0,1)
        else:
            rand_grid = self.grid.expand(
                batch_size, -1, -1, -1)

        x = x[:, :, None].expand(-1, -1, rand_grid.shape[2], -1)
        y = torch.cat([rand_grid, x], dim=-1) # B*num_kp*num_pts_per_grid*C

        for i in range(self.num_kp):
            y_ = y[:, i].transpose(2, 1)
            out_ = self.decoder[i](y_) # B*3*num_pts_per_grid
            if self.config.pose_code in ["nl-noR_T"]:
                out_ = out_ + t[:, i][..., None]
            outs.append(out_)
        if return_splits:
            return outs
        else:
            out = torch.cat(outs, dim=2).transpose(2, 1)
            return out

class MlpPointsFC(nn.Module):
    def __init__(self, desc_dim, num_points, out_dim, config):
        super(MlpPointsFC, self).__init__()
        self.out_dim = out_dim
        self.num_points = num_points

        self.layer = nn.Sequential()
        self.layer.add_module(
            "fc1", nn.Linear(desc_dim, desc_dim))
        self.layer.add_module(
            "relu", nn.ReLU(inplace=True))
        self.layer.add_module(
            "fc2", nn.Linear(desc_dim, out_dim * num_points))

    def forward(self, x):
        # x: BxC
        # pts; Bx3xM
        pts = self.layer(x).reshape(-1, self.out_dim, self.num_points)
        return pts
# decoder_utils.py ends here