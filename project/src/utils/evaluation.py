import numpy as np
import torch
import torch.nn.functional as F

def compute_acc_at_k(pair_sort):
    # pair_sort = np.argsort(dist)
    count_1 = 0
    count_5 = 0
    count_10 = 0
    query_num = pair_sort.shape[0]
    for idx1 in range(0, query_num):
        if idx1 in pair_sort[idx1, 0:1]:
            count_1 = count_1 + 1
        if idx1 in pair_sort[idx1, 0:5]:
            count_5 = count_5 + 1
        if idx1 in pair_sort[idx1, 0:10]:
            count_10 = count_10 + 1
    acc_1 = count_1 / float(query_num)
    acc_5 = count_5 / float(query_num)
    acc_10 = count_10 / float(query_num)
    return [acc_1, acc_5, acc_10]

# def l2_normalize(features):
#     # features: num * ndim
#     features_c = features.copy()
#     features_c /= np.sqrt((features_c * features_c).sum(axis=1))[:, None]
#     return features_c
# def compute_distance(a, b, l2=True):
#     if l2:
#         a = l2_normalize(a)
#         b = l2_normalize(b)

#     """cdist (squared euclidean) with pytorch"""
#     a = torch.from_numpy(a)
#     b = torch.from_numpy(b)
#     a_norm = (a**2).sum(1).view(-1, 1)
#     b_t = b.permute(1, 0).contiguous()
#     b_norm = (b**2).sum(1).view(1, -1)
#     dist = a_norm + b_norm - 2.0 * torch.matmul(a, b_t)
#     dist[dist != dist] = 0
#     return torch.clamp(dist, 0.0, np.inf).numpy()

def compute_distance(a, b, l2=True):
    if l2:
        a = F.normalize(a, p=2, dim=1)
        b = F.normalize(b, p=2, dim=1)
    
    distance = torch.cdist(a, b, p=2)
    return distance
