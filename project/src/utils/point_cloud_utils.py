import numpy as np
import torch
# import cv2
# from pointnet2_ops import _ext

import numpy as np
import pytorch3d.io
import torch
from einops import repeat



def load_pc(path, npoints):
    pc = np.load(path).astype(np.float32) #[n, 3]
    pc = torch.tensor(pc).unsqueeze(0).transpose(1,2) #[1, 3, n]
    pc = sample_farthest_points(pc, npoints) #[1, 3, npoints]
    pc, _, _ = normalize_to_box(pc) #[1, 3, npoints]
    return pc

def normalize(x, method='unit_box'):
    if method == 'unit_box':
        pc, center, scale = normalize_to_box(x)
    else:
        raise ValueError()
    return pc, center, scale

def sample_farthest_points(points, num_samples, return_index=False):
    b, c, n = points.shape
    sampled = torch.zeros((b, 3, num_samples), device=points.device, dtype=points.dtype)
    indexes = torch.zeros((b, num_samples), device=points.device, dtype=torch.int64)
    
    index = torch.randint(n, [b], device=points.device)
    
    gather_index = repeat(index, 'b -> b c 1', c=c)
    sampled[:, :, 0] = torch.gather(points, 2, gather_index)[:, :, 0]
    indexes[:, 0] = index
    dists = torch.norm(sampled[:, :, 0][:, :, None] - points, dim=1)

    # iteratively sample farthest points
    for i in range(1, num_samples):
        _, index = torch.max(dists, dim=1)
        gather_index = repeat(index, 'b -> b c 1', c=c)
        sampled[:, :, i] = torch.gather(points, 2, gather_index)[:, :, 0]
        indexes[:, i] = index
        dists = torch.min(dists, torch.norm(sampled[:, :, i][:, :, None] - points, dim=1))

    if return_index:
        return sampled, indexes
    else:
        return sampled


def resample_mesh(mesh, n_points):
    points, normals = pytorch3d.ops.sample_points_from_meshes(mesh, n_points, return_normals=True)
    points = torch.cat([points[0], normals[0]], dim=-1)
    return points

def rotate_point_cloud(batch_data): # torch.Size([1024, 3])
    rotation_angle = 0.25 * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = torch.tensor([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]]).float()
    rotated_data = torch.mm(batch_data, rotation_matrix)
    return rotated_data # torch.Size([1024, 3])

def normalize_to_box(input):
    """
    normalize point cloud to unit bounding box
    center = (max - min)/2
    scale = max(abs(x))
    input: pc [N, P, dim] or [P, dim]
    output: pc, centroid, furthest_distance

    From https://github.com/yifita/pytorch_points
    """
    if len(input.shape) == 2:
        axis = 0
        P = input.shape[0]
        D = input.shape[1]
    elif len(input.shape) == 3:
        axis = 1
        P = input.shape[1]
        D = input.shape[2]
    else:
        raise ValueError()
    
    if isinstance(input, np.ndarray):
        maxP = np.amax(input, axis=axis, keepdims=True)
        minP = np.amin(input, axis=axis, keepdims=True)
        centroid = (maxP+minP)/2
        input = input - centroid
        furthest_distance = np.amax(np.abs(input), axis=(axis, -1), keepdims=True)
        input = input / furthest_distance
    elif isinstance(input, torch.Tensor):
        maxP = torch.max(input, dim=axis, keepdim=True)[0]
        minP = torch.min(input, dim=axis, keepdim=True)[0]
        centroid = (maxP+minP)/2
        input = input - centroid
        in_shape = list(input.shape[:axis])+[P*D]
        furthest_distance = torch.max(torch.abs(input).reshape(in_shape), dim=axis, keepdim=True)[0]
        furthest_distance = furthest_distance.unsqueeze(-1)
        input = input / furthest_distance
    else:
        raise ValueError()

    return input, centroid, furthest_distance

def write_points_off(fname, points, colors=None):

    with open(fname, 'w') as f:

        num = points.shape[0]
        f.write('COFF\n')
        f.write('{0} 0 0\n'.format(num))
        for i in range(0, num):
            if colors is not None:
                f.write('{0} {1} {2} {3} {4} {5}\n'.format(points[i, 0], points[i, 1], points[i, 2], int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])))
            else:
                f.write('{0} {1} {2}\n'.format(points[i, 0], points[i, 1], points[i, 2]))


def write_points_obj(fname, points, colors=None):

    with open(fname, 'w') as f:

        num = points.shape[0]
        for i in range(0, num):
            if colors is not None:
                f.write('v {0} {1} {2} {3} {4} {5}\n'.format(points[i, 0], points[i, 1], points[i, 2], int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])))
            else:
                f.write('v {0} {1} {2}\n'.format(points[i, 0], points[i, 1], points[i, 2]))


def compute_pca(points):
    mean, eigvec = cv2.PCACompute(points, mean=None)
    if np.dot(np.cross(eigvec[0], eigvec[1]), eigvec[2])<0:
        eigvec[2] = -eigvec[2]

    eigvec[0] = eigvec[0] / np.linalg.norm(eigvec[0])
    eigvec[1] = eigvec[1] / np.linalg.norm(eigvec[1])
    eigvec[2] = eigvec[2] / np.linalg.norm(eigvec[2])

    return eigvec

def query_KNN(points, query_pts, k, return_dis=True):
    '''

    :param points: n x 3
    :param query_pts: m x 3
    :param k: num of neighbors
    :return: m x k  ids, sorted_dis
    '''

    diff = query_pts[:, None, :] - points[None, :, :]
    dis = np.sqrt(np.sum(diff * diff, axis=2))# m x n
    sorted_idx = np.argsort(dis, axis=1)
    sorted_idx = sorted_idx[:, :k]

    if return_dis:
        sorted_dis = dis[None, 0, sorted_idx[0, :]]
        for i in range(1, query_pts.shape[0]):
            sorted_dis = np.concatenate((sorted_dis, dis[None, i, sorted_idx[i, :]]), axis=0)

        return sorted_idx, sorted_dis
    else:
        return sorted_idx


def query_KNN_tensor(points, query_pts, k):
    '''

    :param points: n x 3
    :param query_pts: m x 3
    :param k: num of neighbors
    :return: m x k  ids, sorted_dis
    '''

    diff = query_pts[:, None, :] - points[None, :, :]
    dis = torch.sqrt(torch.sum(diff * diff, dim=2))# m x n
    sorted_idx = torch.argsort(dis, dim=1)
    sorted_idx = sorted_idx[:, :k]

    sorted_dis = dis[None, 0, sorted_idx[0, :]]
    for i in range(1, query_pts.shape[0]):
        sorted_dis = torch.cat((sorted_dis, dis[None, i, sorted_idx[i, :]]), dim=0)

    return sorted_idx, sorted_dis



def read_pointcloud_obj(fname):
    vertices = []
    try:
        f = open(fname)

        for line in f:
            if line[:2] == "v ":
                strs = line.split(' ')
                v0 = float(strs[1])
                v1 = float(strs[2])
                v2 = float(strs[3])
                vertex = [v0, v1, v2]
                vertices.append(vertex)

        f.close()
    except IOError:
        print(".obj file not found.")

    vertices = np.array(vertices)


    return vertices


def read_points_off(fname, read_color=False):
    vertices = []
    colors = []

    try:
        f = open(fname)
        head = f.readline()
        strline = f.readline()
        strs = strline.split(' ')
        vnum = int(strs[0])
        fnum = int(strs[1])
        for i in range(0, vnum):
            strline = f.readline()
            strs = strline.split(' ')
            v0 = float(strs[0])
            v1 = float(strs[1])
            v2 = float(strs[2])
            vertex = [v0, v1, v2]
            vertices.append(vertex)

            if len(strs) > 3:
                c0 = float(strs[3])
                c1 = float(strs[4])
                c2 = float(strs[5])
                color = [c0, c1, c2]
                colors.append(color)




        f.close()
    except IOError:
        print(".off file not found.")

    pts = np.array(vertices).astype(np.float32)

    if len(colors) > 0 and read_color == True:
        colors = np.array(colors).astype(np.float32)
        return pts, colors
    else:
        return pts

def trans_pointcloud(rot_mat, trans_mat, points):
    '''

    :param rot_mat: 3 x 3
    :param trans_mat: 3
    :param points: n x 3
    :return: n x 3
    '''
    tmp_points = np.matmul(rot_mat, np.transpose(points, (1, 0)))
    tmp_points = tmp_points + trans_mat[:, None]
    tmp_points = np.transpose(tmp_points, (1, 0))
    return tmp_points




def farthest_pts_sampling_tensor(pts, num_samples, return_sampled_idx=False):
    '''

    :param pts: bn, n, 3
    :param num_samples:
    :return:
    '''
    sampled_pts_idx = _ext.furthest_point_sampling(pts, num_samples)
    sampled_pts_idx_viewed = sampled_pts_idx.view(sampled_pts_idx.shape[0]*sampled_pts_idx.shape[1]).cuda().type(torch.LongTensor)
    batch_idxs = torch.tensor(range(pts.shape[0])).type(torch.LongTensor)
    batch_idxs_viewed = batch_idxs[:, None].repeat(1, sampled_pts_idx.shape[1]).view(batch_idxs.shape[0]*sampled_pts_idx.shape[1])
    sampled_pts = pts[batch_idxs_viewed, sampled_pts_idx_viewed, :]
    sampled_pts = sampled_pts.view(pts.shape[0], num_samples, 3)

    if return_sampled_idx == False:
        return sampled_pts
    else:
        return sampled_pts, sampled_pts_idx























