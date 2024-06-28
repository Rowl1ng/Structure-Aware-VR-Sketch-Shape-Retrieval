import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.nn.functional as F
import pytorch3d.loss
# from tools.chamfer_python import chamfer_distance
################################################################
## Triplet related loss
################################################################

def egocentric(sketch_feat, shape_feat, matrix):
    B1 = sketch_feat.shape[0]
    B2 = shape_feat.shape[0]
    u = torch.repeat_interleave(sketch_feat, torch.ones(B1, dtype=torch.long).cuda()*B2, dim=0)
    v = shape_feat.repeat(B1, 1)
    cov = matrix.repeat(B1, 1)
    delta = u - v
    m = torch.matmul(delta.pow(2).unsqueeze(1), cov.unsqueeze(-1)) 
    return torch.sqrt(m).view([B1, B2])

def egocentric_iter(sketch_feat, shape_feat, matrix):
    B1 = sketch_feat.shape[0]
    B2 = shape_feat.shape[0]
    dist = []
    for i in range(B1):
        u = sketch_feat[i].repeat(B2, 1)
        # for j in range(B2):
        delta = u - shape_feat
        m = torch.matmul(delta.pow(2).unsqueeze(1), matrix.unsqueeze(-1)) 
        dist.append(m)
    dist = torch.cat(dist, 0)
    return torch.sqrt(dist).view([B1, B2])

# Modified from: https://github.com/XuyangBai/PPF-FoldNet/blob/master/loss/chamfer_loss.py 
class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, gts, preds, atten=None, side="both", reduce=True):
        # atten: BMN?
        P = self.batch_pairwise_dist(gts, preds)
        if atten is not None:
            P = P / (atten + 1e-3)
        if reduce:
            mins, _ = torch.min(P, 1)
            loss_1 = torch.mean(mins)
            mins, _ = torch.min(P, 2)
            loss_2 = torch.mean(mins)
            if side == "both":
                return loss_1 + loss_2
            elif side == "left":
                return loss_1
            elif side == "right":
                return loss_2
        else:
            mins, _ = torch.min(P, 1)
            loss_1 = torch.mean(mins, dim=1)
            mins, _ = torch.min(P, 2)
            loss_2 = torch.mean(mins, dim=1)
            if side == "both":
                return loss_1 + loss_2
            elif side == "left":
                return loss_1
            elif side == "right":
                return loss_2

    def batch_pairwise_dist(self, x, y):
        x = x.float()
        y = y.float()
        bs, num_points_x, points_dim = x.size()
        xx = torch.sum(x ** 2, dim=2, keepdim=True)
        yy = torch.sum(y ** 2, dim=2)[:, None]
        xy = torch.matmul(x, y.transpose(2, 1))
        P = (xx + yy - 2*xy)
        return P

    def batch_pairwise_dist_deprecated(self, x, y):
        x = x.float()
        y = y.float()
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        # brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

class Chamfer_dist(nn.Module):
    def __init__(self):
        super(Chamfer_dist, self).__init__()
    def forward(self, gts, preds, reduce=True):
        return pytorch3d.loss.chamfer_distance(
                        gts, preds)[0]

class ordered_l2(nn.Module): 
    def __init__(self):
        super(ordered_l2, self).__init__()
        self.pdist = nn.PairwiseDistance(p=2)
        self.use_cuda = torch.cuda.is_available()

    def forward(self, gts, preds, reduce=True):
        bs, num_point, dim = gts.shape
        gts = gts.reshape((bs * num_point, -1))
        preds = preds.reshape((bs * num_point, -1))
        output = self.pdist(gts, preds).view(bs, -1)
        output = output.mean(-1)
        if reduce:
            return output.mean()
        return output

class ordered_l2_weighted(nn.Module):
    def __init__(self):
        super(ordered_l2_weighted, self).__init__()
        self.pdist = nn.PairwiseDistance(p=2)
        self.use_cuda = torch.cuda.is_available()

    def normalize(self, x):
        return (x - x.min()) / (x.max() - x.min())
    def forward(self, gts, preds, reduce=True):
        bs, num_point, dim = gts.shape
        gts = gts.reshape((bs * num_point, -1))
        preds = preds.reshape((bs * num_point, -1))
        gt_norm = torch.norm(gts, p=2, dim=1)
        weights = torch.exp(self.normalize(gt_norm))
        output = (self.pdist(gts, preds) * weights).mean()
        return output

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class ContrastiveLoss_1_aug(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(ContrastiveLoss_1_aug, self).__init__()
        self.lsoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.anchor_num = 2

    def forward(self, embeddings, xyz):
        device = (torch.device('cuda')
                  if embeddings.is_cuda
                  else torch.device('cpu'))
        embeddings = F.normalize(embeddings, p=2, dim=1)
        B = embeddings.shape[0]
        mini_batch = int(B/(self.anchor_num+1))
        anchor_num = self.anchor_num * mini_batch
        sketchs, shapes = torch.split(embeddings, [anchor_num, mini_batch], dim=0)
        total = torch.div(torch.mm(sketchs, torch.transpose(shapes, 0, 1)), self.temperature) # e.g. size 8*8
        # for numerical stability
        logits_max, _ = torch.max(total, dim=1, keepdim=True)
        logits = total - logits_max.detach()
        logits = logits.repeat(1, self.anchor_num)

        logits_mask = (torch.ones([mini_batch, mini_batch]) - torch.eye(mini_batch)).repeat(self.anchor_num, self.anchor_num).to(device)
        exp_logits = torch.exp(logits) * logits_mask

        # Sketch x Sketch
        ss_total = torch.div(torch.mm(sketchs, torch.transpose(sketchs, 0, 1)), self.temperature)  # e.g. size 8*8
        logits_max_ss, _ = torch.max(ss_total, dim=1, keepdim=True)
        logits_ss = ss_total - logits_max_ss.detach()
        ss_logits_mask = (torch.ones([anchor_num, anchor_num]) - torch.eye(anchor_num)).to(device)
        exp_logits_ss = torch.exp(logits_ss) * ss_logits_mask

        s_z = torch.diag(logits).view([-1, 1])
        log_prob_prob_pos = s_z + torch.diag(logits_ss).view([-1, 1]) - torch.log(exp_logits.sum(1, keepdim=True) / self.anchor_num + exp_logits_ss.sum(1, keepdim=True))
        mean_log_prob_pos = log_prob_prob_pos.mean(0)
        nce = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return nce

class ContrastiveLoss_1(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(ContrastiveLoss_1, self).__init__()
        self.lsoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()
        self.temperature = temperature
        self.base_temperature = base_temperature
    def forward(self, embeddings, xyz):
        device = (torch.device('cuda')
                  if embeddings.is_cuda
                  else torch.device('cpu'))
        embeddings = F.normalize(embeddings, p=2, dim=1)
        B = embeddings.shape[0]
        mini_batch = int(B/2)
        sketchs, shapes = torch.split(embeddings, [mini_batch, mini_batch], dim=0)
        total = torch.div(torch.mm(sketchs, torch.transpose(shapes, 0, 1)), self.temperature) # e.g. size 8*8
        # for numerical stability
        logits_max, _ = torch.max(total, dim=1, keepdim=True)
        logits = total - logits_max.detach()
        logits_mask = (torch.ones([mini_batch, mini_batch]) - torch.eye(mini_batch)).to(device)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob_prob_pos = torch.diag(logits).view([-1, 1]) - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = log_prob_prob_pos.mean(0)
        nce = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return nce

class ContrastiveLoss_2(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(ContrastiveLoss_2, self).__init__()
        self.lsoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()
        self.temperature = temperature
        self.base_temperature = base_temperature
    def forward(self, embeddings, xyz):
        device = (torch.device('cuda')
                  if embeddings.is_cuda
                  else torch.device('cpu'))
        embeddings = F.normalize(embeddings, p=2, dim=1)
        B = embeddings.shape[0]
        mini_batch = int(B/2)
        sketchs, shapes = torch.split(embeddings, [mini_batch, mini_batch], dim=0)
        total = torch.div(torch.mm(sketchs, torch.transpose(shapes, 0, 1)), self.temperature) # e.g. size 8*8
        # for numerical stability
        logits_max, _ = torch.max(total, dim=1, keepdim=True)
        logits = total - logits_max.detach()
        logits_mask = (torch.ones([mini_batch, mini_batch]) - torch.eye(mini_batch)).to(device)
        exp_logits = torch.exp(logits) * logits_mask

        ss_total = torch.div(torch.mm(sketchs, torch.transpose(sketchs, 0, 1)), self.temperature) # e.g. size 8*8
        logits_max_ss, _ = torch.max(ss_total, dim=1, keepdim=True)
        logits_ss = ss_total - logits_max_ss.detach()
        exp_logits_ss = torch.exp(logits_ss) * logits_mask

        log_prob_prob_pos = torch.diag(logits).view([-1, 1]) - torch.log(exp_logits.sum(1, keepdim=True) + exp_logits_ss.sum(1, keepdim=True))
        mean_log_prob_pos = log_prob_prob_pos.mean(0)
        nce = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return nce

class ContrastiveLoss_3(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(ContrastiveLoss_3, self).__init__()
        self.lsoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()
        self.temperature = temperature
        self.base_temperature = base_temperature
    def forward(self, embeddings, xyz):
        device = (torch.device('cuda')
                  if embeddings.is_cuda
                  else torch.device('cpu'))
        embeddings = F.normalize(embeddings, p=2, dim=1)
        B = embeddings.shape[0]
        mini_batch = int(B/2)
        sketchs, shapes = torch.split(embeddings, [mini_batch, mini_batch], dim=0)
        total = torch.div(torch.mm(sketchs, torch.transpose(shapes, 0, 1)), self.temperature) # e.g. size 8*8
        # for numerical stability
        logits_max, _ = torch.max(total, dim=1, keepdim=True)
        logits = total - logits_max.detach()
        logits_mask = (torch.ones([mini_batch, mini_batch]) - torch.eye(mini_batch)).to(device)
        exp_logits = torch.exp(logits) * logits_mask

        ss_total = torch.div(torch.mm(sketchs, torch.transpose(sketchs, 0, 1)), self.temperature) # e.g. size 8*8
        logits_max_ss, _ = torch.max(ss_total, dim=1, keepdim=True)
        logits_ss = ss_total - logits_max_ss.detach()
        exp_logits_ss = torch.exp(logits_ss) * logits_mask

        zz_total = torch.div(torch.mm(shapes, torch.transpose(shapes, 0, 1)), self.temperature) # e.g. size 8*8
        logits_max_zz, _ = torch.max(zz_total, dim=1, keepdim=True)
        logits_zz = zz_total - logits_max_zz.detach()
        exp_logits_zz = torch.exp(logits_zz) * logits_mask

        log_prob_prob_pos = torch.diag(logits).view([-1, 1]) - torch.log(exp_logits.sum(1, keepdim=True) + exp_logits_ss.sum(1, keepdim=True) + exp_logits_zz.sum(1, keepdim=True))
        mean_log_prob_pos = log_prob_prob_pos.mean(0)
        nce = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return nce
class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector, beta=1):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector
        self.softplus = nn.Softplus(beta=beta)

    def forward(self, anchors,  positives, softplus=False):
        anchors = F.normalize(anchors, p=2, dim=1)
        positives = F.normalize(positives, p=2, dim=1)
        triplets = self.triplet_selector.get_triplets(anchors,  positives)
        if anchors.is_cuda:
            triplets = triplets.cuda()
        ap_distances = (anchors[triplets[:, 0]] - positives[triplets[:, 0]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (anchors[triplets[:, 0]] - positives[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        if softplus:
            losses = self.softplus(ap_distances - an_distances + self.margin)
        else:
            losses = F.relu(ap_distances - an_distances + self.margin)
        return losses.mean()

def pdist(A, squared=False, eps=1e-4):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    return res if squared else (res + eps).sqrt() + eps


class RegressionLoss(nn.Module):
    def __init__(self, args):
        super(RegressionLoss, self).__init__()
        self.sigma = args.m * 0.997/3
        self.use_softmax = args.use_softmax
        self.use_KL = args.use_KL
        self.softmax = nn.Softmax(dim=1)
        self.KL = nn.KLDivLoss()
    def forward(self, embeddings, xyz):
        device = (torch.device('cuda')
                  if embeddings.is_cuda
                  else torch.device('cpu'))
        embeddings = F.normalize(embeddings, p=2, dim=1)

        B = embeddings.shape[0]
        mini_batch = int(B/2)
        sketchs, shapes = torch.split(xyz, [mini_batch, mini_batch], dim=0)
        sketchs_feat, shapes_feat = torch.split(embeddings, [mini_batch, mini_batch], dim=0)
        embeddings_inner_dot = torch.mm(sketchs_feat, torch.transpose(shapes_feat, 0, 1))

        x = torch.repeat_interleave(shapes, torch.ones(mini_batch, dtype=torch.long).to(device)*mini_batch, dim=0)
        y = shapes.repeat(mini_batch, 1, 1)
        real_dist = torch.div(chamfer_distance(x, y).view([mini_batch, mini_batch]).pow(2), 2 *(self.sigma ** 2))

        p = self.softmax(-real_dist)  # [b, b]
        if self.use_softmax:
            p_hat = self.softmax(embeddings_inner_dot) # [b, b]
        else:
            p_hat = torch.div(embeddings_inner_dot + 1, (embeddings_inner_dot + 1).sum(1, keepdim=True))
        if not self.use_KL:
            reg = torch.mean(torch.abs(p_hat - p))
        else:
            # reg = torch.mean(torch.square(p_hat - p))
            reg = self.KL(p_hat, p)
        return reg.mean()



class TripletCenterLoss(nn.Module):
    def __init__(self, margin=0, center_embed=10, num_classes=10, l2norm=False):
        super(TripletCenterLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.centers = nn.Parameter(torch.randn(num_classes, center_embed))
        self.l2norm = l2norm

    def forward(self, inputs, targets):
        batch_size = inputs.shape[0]
        targets_expand = targets.view(batch_size, 1).expand(batch_size, inputs.shape[1])
        centers_batch = self.centers.gather(0, targets_expand)  # centers batch

        # compute pairwise distances between input features and corresponding centers
        centers_batch_bz = torch.stack([centers_batch] * batch_size)
        inputs_bz = torch.stack([inputs] * batch_size).transpose(0, 1)
        if self.l2norm:
            centers_batch_bz = F.normalize(centers_batch_bz, p=2, dim=1)
            inputs_bz = F.normalize(inputs_bz, p=2, dim=1)

        dist = torch.sum((centers_batch_bz - inputs_bz) ** 2, 2).squeeze()
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # for each anchor, find the hardest positive and negative
        mask = targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())
        dist_ap, dist_an = [], []
        for i in range(batch_size):  # for each sample, we compute distance
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # mask[i]: positive samples of sample i
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # mask[i]==0: negative samples of sample i

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # generate a new label y
        # compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        # y_i = 1, means dist_an > dist_ap + margin will casuse loss be zero
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # prec = (dist_an.data > dist_ap.data).sum().to(dtype=torch.float)/ y.size(0) # normalize data by batch size
        prec = (dist_an.data - dist_ap.data).sum().to(dtype=torch.float) / y.size(0)
        triplet_num = y.shape[0]
        return loss, prec#, triplet_num

if __name__ == "__main__":
    from dataset.TripletSampler import RandomNegativeTripletSelector
    margin = 1.
    loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
    batch_size = 12
    embeddings = torch.randn(batch_size, 10)
    target = torch.Tensor([1,1,2,2,3,3,1,1,2,2,3,3]).view(batch_size)
    loss, triplet_num = loss_fn(embeddings, target)
    print(loss, triplet_num)