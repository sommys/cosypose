import torch
from cosypose.lib3d.transform_ops import transform_pts


def dists_add(TXO_pred, TXO_gt, points):
    TXO_pred_points = transform_pts(TXO_pred, points)
    TXO_gt_points = transform_pts(TXO_gt, points)
    dists = TXO_gt_points - TXO_pred_points
    return dists

def dists_add_symmetric(TXO_pred, TXO_gt, points):
    TXO_pred_points = transform_pts(TXO_pred, points)
    TXO_gt_points = transform_pts(TXO_gt, points)
    distances = torch.cdist(TXO_gt_points, TXO_pred_points,
                            p=2, compute_mode='donot_use_mm_for_euclid_dist')
    closest_points_idx = torch.argmin(distances, dim=2).squeeze()
    TXO_pred_closest_to_gt = torch.index_select(TXO_pred_points, 1, closest_points_idx)
    min_translations = TXO_gt_points - TXO_pred_closest_to_gt
    return min_translations
