import numpy as np
import torch


def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()

def apply_random_scale_xyz(pc, scale=[0.9, 1.1]):
    B, N, dim = pc.shape
    scale = torch.rand(B, dim,dtype=torch.double).to(pc) * (scale[1] - scale[0]) + scale[0]
    # (B, N, 3) mul (B, 3, 3) -> (B, N, 3)
    scale = scale.view(B, 1, dim).repeat(1, N, 1)
    pc_scaled = pc * scale

    return pc_scaled

class PointcloudScale(object):
    def __init__(self, lo=0.8, hi=1.25):
        self.lo, self.hi = lo, hi

    def __call__(self, batch_data):
        # scaler = np.random.uniform(self.lo, self.hi)
        # points[:, 0:3] *= scaler
        # return points
        B, N, C = batch_data.shape
        # scales = np.random.uniform(self.lo, self.hi, B)
        scales = torch.FloatTensor(B).uniform_(self.lo, self.hi).to(batch_data.device)

        for batch_index in range(B):
            batch_data[batch_index,:,:] *= scales[batch_index]
        return batch_data



class PointcloudRotate(object):
    def __init__(self, axis=np.array([0.0, 1.0, 0.0])):
        self.axis = axis

    def __call__(self, points):
        rotation_angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = angle_axis(rotation_angle, self.axis)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip

    def _get_angles(self):
        angles = np.clip(
            self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
        )

        return angles

    def __call__(self, points):
        angles = self._get_angles()
        Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, points):
        jittered_data = (
            points.new(points.size(0), 3)
            .normal_(mean=0.0, std=self.std)
            .clamp_(-self.clip, self.clip)
        )
        points[:, 0:3] += jittered_data
        return points


class PointcloudTranslate(object):
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range

    def __call__(self, batch_data):
        # translation = np.random.uniform(-self.translate_range, self.translate_range)
        # points[:, 0:3] += translation
        # return points
        B, N, C = batch_data.shape
        # shifts = np.random.uniform(-self.translate_range, self.translate_range, (B,3))
        # r1, r2 = [-self.translate_range, self.translate_range]
        # shifts = (r1 - r2) * torch.rand(B, 3) + r2
        shifts = torch.FloatTensor(B, 3).uniform_(-self.translate_range, self.translate_range).to(batch_data.device)

        for batch_index in range(B):
            batch_data[batch_index,:,:] += shifts[batch_index,:]
        return batch_data



class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()


class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.875):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, batch_pc):
        ''' batch_pc: BxNx3 '''
        B, N, _ = batch_pc.shape
        dropout_ratio =  np.random.random(B)*self.max_dropout_ratio # 0~0.875

        # dropout_ratio = torch.rand(B).to(batch_pc.device) * self.max_dropout_ratio  # 0~0.875
        for b in range(B):
        #     batch_pc[b] = torch.where(torch.rand(N).to(batch_pc.device) <= dropout_ratio[b], batch_pc[b][0], batch_pc[b])
            # if len(drop_idx) > 0:
            #     pc[drop_idx] = pc[0]  # set to the first point
            drop_idx = np.where(np.random.random(N)<=dropout_ratio[b])[0]
            if len(drop_idx)>0:
                batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point

        return batch_pc
