import torch
from typing import Dict, Optional, Tuple
from protein_utils.affine_utils import T
from protein_utils.rigid_utils import Rigid, Rotation
from Loss.distance_loss import compute_fape, compute_fape_dis
from Loss.clash_loss import clash_avoidance_loss


def backbone_loss(
    backbone_rigid_tensor: torch.Tensor,
    backbone_rigid_mask: torch.Tensor,
    traj: torch.Tensor,
    pair_mask: Optional[torch.Tensor] = None,
    use_clamped_fape: Optional[torch.Tensor] = None,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    eps: float = 1e-4,
    dist_gt=None,
    dist_upper_bound_gt=None,
    dist_lower_bound_gt=None,
    mask_window=None,
    using_fafe: bool = False,
    **kwargs,
) -> torch.Tensor:
    ### need to check if the traj belongs to 4*4 matrix or a tensor_7
    if traj.shape[-1] == 7:
        pred_aff = Rigid.from_tensor_7(traj)
    elif traj.shape[-1] == 4:
        pred_aff = Rigid.from_tensor_4x4(traj)

    pred_aff = Rigid(
        Rotation(rot_mats=pred_aff.get_rots().get_rot_mats(), quats=None),
        pred_aff.get_trans(),
    )
    # DISCREPANCY: DeepMind somehow gets a hold of a tensor_7 version of
    # backbone tensor, normalizes it, and then turns it back to a rotation
    # matrix. To avoid a potentially numerically unstable rotation matrix
    # to quaternion conversion, we just use the original rotation matrix
    # outright. This one hasn't been composed a bunch of times, though, so
    # it might be fine.

    # pred_aff = T.from_tensor(traj)
    # gt_aff = T.from_tensor(backbone_affine_tensor)
    gt_aff = Rigid.from_tensor_4x4(backbone_rigid_tensor)

    if using_fafe:
        from Loss.distance_loss import compute_fafe

        losses = compute_fafe(
            R_pred=pred_aff.get_rots().get_rot_mats(),
            t_pred=pred_aff.get_trans(),
            R_gt=gt_aff[None].get_rots().get_rot_mats(),
            t_gt=gt_aff[None].get_trans(),
            frame_mask=backbone_rigid_mask,
            rotate_scale=1.0,
            axis_scale=loss_unit_distance,
            eps_r3=eps,
            dist_clamp=clamp_distance,
            pair_mask=pair_mask,
            mask_window=mask_window,
            dist_gt=dist_gt,
            dist_upper_bound_gt=dist_upper_bound_gt,
            dist_lower_bound_gt=dist_lower_bound_gt,
        )
        loss = losses["fafe"]
    else:
        loss = compute_fape(
            pred_aff,
            gt_aff[None],
            backbone_rigid_mask[None],
            pred_aff.get_trans(),
            gt_aff[None].get_trans(),
            backbone_rigid_mask[None],
            pair_mask=pair_mask,
            l1_clamp_distance=clamp_distance,
            length_scale=loss_unit_distance,
            eps=eps,
            dist_gt=dist_gt,
            dist_upper_bound_gt=dist_upper_bound_gt,
            dist_lower_bound_gt=dist_lower_bound_gt,
            mask_window=mask_window,
        )

    if use_clamped_fape is not None:
        unclamped_fape_loss = compute_fape(
            pred_aff,
            gt_aff[None],
            backbone_rigid_mask[None],
            pred_aff.get_trans(),
            gt_aff[None].get_trans(),
            backbone_rigid_mask[None],
            pair_mask=pair_mask,
            l1_clamp_distance=None,
            length_scale=loss_unit_distance,
            eps=eps,
            dist_gt=dist_gt,
            dist_upper_bound_gt=dist_upper_bound_gt,
            dist_lower_bound_gt=dist_lower_bound_gt,
            mask_window=mask_window,
        )

        loss = loss * use_clamped_fape + unclamped_fape_loss * (1 - use_clamped_fape)

    # Average over the batch dimension
    loss = torch.mean(loss)
    # return fape_loss
    dis_loss = compute_fape_dis(
        pred_frames=pred_aff,
        pred_positions=pred_aff.get_trans(),
        length_scale=loss_unit_distance,
        eps=eps,
        dist_gt=dist_gt,
        dist_upper_bound_gt=dist_upper_bound_gt,
        dist_lower_bound_gt=dist_lower_bound_gt,
    )

    clash_loss = clash_avoidance_loss(
        pred_frames=pred_aff,
        pred_positions=pred_aff.get_trans(),
        eps=eps,
        mask_window=mask_window,
    )

    return loss, dis_loss, clash_loss
