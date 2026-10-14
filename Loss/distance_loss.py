import torch
from protein_utils.affine_utils import T
from protein_utils.rigid_utils import Rigid, Rotation
from typing import Dict, Optional, Tuple


def compute_fape_dis(
    pred_frames: Rigid,
    pred_positions: torch.Tensor,
    length_scale: float,
    eps=1e-8,
    pair_mask: Optional[torch.Tensor] = None,
    dist_gt=None,
    dist_upper_bound_gt: torch.Tensor = None,
    dist_lower_bound_gt: torch.Tensor = None,
) -> torch.Tensor:

    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    if dist_gt is not None:
        dis_mask = torch.where(dist_gt != 0, 1, 0)
        if pair_mask is not None:
            effective_mask = dis_mask * pair_mask
        else:
            effective_mask = dis_mask
        local_pred_pos_pair = local_pred_pos * effective_mask[None, None, ..., None]

        error_dist_pair = 0.5 * torch.square(
            (
                dist_gt[None, None, ...]
                - torch.sqrt(torch.sum(local_pred_pos_pair**2, dim=-1) + eps)
            )
            * effective_mask[None, None, ...]
        )

        error_dist_pair = error_dist_pair / length_scale

        loss = torch.sum(error_dist_pair) / (
            len(torch.nonzero(effective_mask)) * pred_positions.size(0)
        )
        return loss

    if dist_upper_bound_gt is not None and dist_lower_bound_gt is not None:
        dis_mask = torch.where(
            (dist_lower_bound_gt != 0) | (dist_upper_bound_gt != 0), 1, 0
        )
        if pair_mask is not None:
            effective_mask = dis_mask * pair_mask
        else:
            effective_mask = dis_mask

        local_pred_pos_pair = local_pred_pos * effective_mask[None, None, ..., None]

        # calculate the distance
        pred_dist = torch.sqrt(torch.sum(local_pred_pos_pair**2, dim=-1) + eps)

        # if the distance is within the bounds, the error is 0
        error_dist_pair = torch.where(
            (pred_dist >= dist_lower_bound_gt[None, None, ...])
            & (pred_dist <= dist_upper_bound_gt[None, None, ...]),
            torch.zeros_like(pred_dist),
            torch.minimum(
                torch.abs(pred_dist - dist_lower_bound_gt[None, None, ...]),
                torch.abs(pred_dist - dist_upper_bound_gt[None, None, ...]),
            ),
        )

        # apply mask
        error_dist_pair = 0.5 * torch.square(
            error_dist_pair * effective_mask[None, None, ...]
        )

        # normalize by length scale
        error_dist_pair = error_dist_pair / length_scale

        # calculate the loss
        loss = torch.sum(error_dist_pair) / (
            len(torch.nonzero(effective_mask)) * pred_positions.size(0)
        )

        return loss
    else:
        raise ValueError("distance constraints must be provided.")


def compute_sidechain_dis(
    pred_frames: T,
    target_frames: T,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    pair_mask: Optional[torch.Tensor] = None,
    l1_clamp_distance: Optional[float] = None,
    eps=1e-8,
    dist_gt=None,
    dist_upper_bound_gt: Optional[torch.Tensor] = None,
    dist_lower_bound_gt: Optional[torch.Tensor] = None,
    domain_window=None,
) -> torch.Tensor:

    # [*, N_frames, N_pts, 3]

    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    error_dist = None

    if dist_gt is not None or (
        dist_lower_bound_gt is not None and dist_upper_bound_gt is not None
    ):
        if domain_window is not None:
            domain_window_sidechain = domain_window.repeat_interleave(
                14, dim=1
            ).repeat_interleave(8, dim=0)
            local_pred_pos = local_pred_pos * domain_window_sidechain[None, ..., None]
            local_target_pos = (
                local_target_pos * domain_window_sidechain[None, ..., None]
            )

        error_dist = torch.sqrt(
            torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
        )

    if error_dist is None:
        raise ValueError("distance constraints must be provided.")

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale

    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]
    # FP16-friendly averaging. Roughly equivalent to:
    #
    # norm_factor = (
    #     torch.sum(frames_mask, dim=-1) *
    #     torch.sum(positions_mask, dim=-1)
    # )
    # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
    #
    # ("roughly" because eps is necessarily duplicated in the latter
    # atom14_mask = positions_mask.squeeze(0).repeat_interleave(frames_mask.shape[1]).reshape(-1, positions_mask.shape[1])
    # group8_mask = frames_mask.squeeze(0).repeat_interleave(positions_mask.shape[1]).reshape(-1, frames_mask.shape[1]).transpose_(0,1)

    if pair_mask is not None:
        normed_error = normed_error * pair_mask
        normed_error = torch.sum(normed_error, dim=(-1, -2))

        mask = frames_mask[..., None] * positions_mask[..., None, :] * pair_mask
        norm_factor = torch.sum(mask, dim=(-2, -1))

        normed_error = normed_error / (eps + norm_factor)
    else:
        normed_error = torch.sum(normed_error, dim=-1)
        normed_error = normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
        normed_error = torch.sum(normed_error, dim=-1)
        normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error


def compute_fape(
    pred_frames: Rigid,
    target_frames: Rigid,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    pair_mask: Optional[torch.Tensor] = None,
    l1_clamp_distance: Optional[float] = None,
    eps=1e-8,
    dist_gt=None,
    dist_lower_bound_gt: Optional[torch.Tensor] = None,
    dist_upper_bound_gt: Optional[torch.Tensor] = None,
    mask_window=None,
) -> torch.Tensor:

    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if dist_gt is not None or (
        dist_lower_bound_gt is not None and dist_upper_bound_gt is not None
    ):
        if mask_window is not None:
            error_dist = error_dist * mask_window[None, None, ...]

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)
        # error_dist = l1_clamp_distance * torch.tanh(error_dist / l1_clamp_distance)

    normed_error = error_dist / length_scale

    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]

    if pair_mask is not None:
        normed_error = normed_error * pair_mask
        normed_error = torch.sum(normed_error, dim=(-1, -2))

        mask = frames_mask[..., None] * positions_mask[..., None, :] * pair_mask
        norm_factor = torch.sum(mask, dim=(-2, -1))

        normed_error = normed_error / (eps + norm_factor)
    else:
        normed_error = torch.sum(normed_error, dim=-1)
        normed_error = normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
        normed_error = torch.sum(normed_error, dim=-1)
        normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error


from typing import Union


def compute_fafe(
    R_pred: torch.Tensor,
    t_pred: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
    frame_mask: torch.Tensor,
    rotate_scale: float = 1.0,
    axis_scale: float = 20.0,
    eps_so3: float = 1e-7,
    eps_r3: float = 1e-4,
    dist_clamp: Union[float, None] = None,
    pair_mask: Union[torch.Tensor, None] = None,
    dist_gt: Union[torch.Tensor, None] = None,
    dist_upper_bound_gt: Union[torch.Tensor, None] = None,
    dist_lower_bound_gt: Union[torch.Tensor, None] = None,
    mask_window: Union[torch.Tensor, None] = None,
):
    """Compute frame aligned frame error loss with double geodesic metric.

    Args:
        R_pred: Predicted rotation matrices of frames, (..., N, 3, 3).
        t_pred: Predicted translations of frames, (..., N, 3).
        R_gt: Ground truth rotation matrices of frames, (..., N, 3, 3).
        t_gt: Ground truth translations of frames, (..., N, 3).
        frame_mask: Existing masks of ground truth frames, (..., N).
        axis_scale: Scale by which the R^3 part of loss is divided.
        eps_so3: Small number for numeric safety for arccos.
        eps_r3: Small number for numeric safety for sqrt.
        dist_clamp: Cutoff above which distance errors are disregarded.
        pair_mask: Additional pair masks of pairs which should be calculated, (..., N, M) or None.
            pair_mask=True, the FAPE loss is calculated; vice not calculated.
            If None, all pairs are calculated.

    Returns:
        Dict of (B) FAFE losses. Contains "fafe", "fafe_so3", "fafe_r3".
    """
    import einops
    from Loss.fafe.fafe_loss import (
        compute_double_geodesic_error,
        invert_rigid,
        compose_rotation_and_translation,
        node2pair,
    )

    N = R_pred.shape[-3]

    def _diff_frame(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        R_inv, t_inv = invert_rigid(
            R=einops.repeat(R, "... i r1 r2 -> ... (i j) r1 r2", j=N),
            t=einops.repeat(t, "... i t -> ... (i j) t", j=N),
        )
        R_j = einops.repeat(R, "... j r1 r2 -> ... (i j) r1 r2", i=N)
        t_j = einops.repeat(t, "... j t -> ... (i j) t", i=N)

        return compose_rotation_and_translation(R_inv, t_inv, R_j, t_j)

    frame_mask = node2pair(frame_mask, frame_mask, -1, torch.logical_and)
    if pair_mask is not None:
        frame_mask = pair_mask * frame_mask
    if dist_gt is not None or (
        dist_lower_bound_gt is not None and dist_upper_bound_gt is not None
    ):
        if mask_window is not None:
            frame_mask = frame_mask * mask_window
    frame_mask = einops.rearrange(frame_mask, "... i j -> ... (i j)")

    losses = compute_double_geodesic_error(
        *_diff_frame(R_pred, t_pred),
        *_diff_frame(R_gt, t_gt),
        frame_mask=frame_mask,
        rotate_scale=rotate_scale,
        axis_scale=axis_scale,
        dist_clamp=dist_clamp,
        eps_so3=eps_so3,
        eps_r3=eps_r3,
    )
    return losses
