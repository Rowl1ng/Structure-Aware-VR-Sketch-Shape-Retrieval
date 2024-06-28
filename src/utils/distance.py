import torch
# from tools.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
# from tools.ChamferDistancePytorch.fscore import fscore
# chamLoss = dist_chamfer_3D.chamfer_3DDist()
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss.chamfer import _handle_pointcloud_input

def chamfer_single(x, y, batch_reduction=None):
    x, x_lengths, x_normals = _handle_pointcloud_input(x, None, None)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, None, None)

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)
    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()

    return cham_x, cham_y

def fscore(dist1, dist2, threshold=0.001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2

def f_score(x, y, radius=0.01, eps=1e-8):
    r"""Computes the f-score of two sets of points, with a hit defined by two point existing within a defined radius of each other

    Args:
        gt_points (torch.Tensor): ground truth pointclouds of shape (B, N, 3)
        pred_points (torch.Tensor): predicted points pointclouds of shape (B, M, 3)
        radius (float): radius from a point to define a hit
                        Default: 0.01
        eps (float): epsilon used to calculate f score.

    Returns:
        (torch.Tensor): computed f-score tensor of shape (B), which has the same dtype as input pred_points.

    Example:
        >>> p1 = torch.tensor([[[8.8977, 4.1709, 1.2839],
        ...                     [8.5640, 7.7767, 9.4214]],
        ...                    [[0.5431, 6.4495, 11.4914],
        ...                     [3.2126, 8.0865, 3.1018]]], device='cuda', dtype=torch.float)
        >>> p2 = torch.tensor([[[9.4863, 4.2249, 0.1712],
        ...                     [8.1783, 8.5310, 8.5119]],
        ...                    [[-0.0020699, 6.4429, 12.3],
        ...                     [3.8386, 8.3585, 4.7662]]], device='cuda', dtype=torch.float)
        >>> f_score(p1, p2, radius=1)
        tensor([0.5000, 0.0000], device='cuda:0')
        >>> f_score(p1, p2, radius=1.5)
        tensor([1.0000, 0.5000], device='cuda:0')
    """
    # pred_distances = torch.sqrt(sided_distance(gt_points, pred_points)[0])
    # gt_distances = torch.sqrt(sided_distance(pred_points, gt_points)[0])

    # data_type = gt_points.dtype

    # fn = torch.sum(pred_distances > radius, dim=1).type(data_type)
    # fp = torch.sum(gt_distances > radius, dim=1).type(data_type)
    # tp = (gt_distances.shape[1] - fp).type(data_type)

    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    x, x_lengths, x_normals = _handle_pointcloud_input(x, None, None)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, None, None)

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    # dist1, dist2, idx1, idx2 = chamLoss(gt_points, pred_points)
    f_score, precision, recall = fscore(cham_x, cham_y, threshold=radius)
    # f_score = 2 * (precision * recall) / (precision + recall + eps)
    return f_score