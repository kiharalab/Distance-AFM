import torch
from protein_utils.affine_utils import T
from protein_utils.rigid_utils import Rigid, Rotation
from typing import Dict, Optional, Tuple


def clash_avoidance_loss(
    pred_frames: Rigid,
    pred_positions: torch.Tensor,
    eps=1e-8,
    mask_window=None,
    clash_limit=200,
    clash_dis=5.0,
    alpha=0.1,
    beta=0.01,
    gamma=1,
) -> torch.Tensor:
    """
    Calculate clash avoidance loss.

    dist_matrix: Tensor of shape [batch_size, num_atoms, num_atoms]
    clash_limit: Threshold for the number of clashes
    penalty_weight: Weight for the penalty

    Returns the clash avoidance loss.
    """
    device = pred_positions.device

    # local_pred_pos = pred_frames.invert()[..., None].apply(
    #     pred_positions[..., None, :, :],
    # )

    # Calculate pairwise distances within local_pred_pos
    diff = pred_positions[..., :, None, :] - pred_positions[..., None, :, :]
    intra_structure_distances = torch.sqrt(torch.sum(diff**2, dim=-1) + eps)

    if mask_window is not None:
        effective_mask = torch.where(mask_window == 0, 1, 0)
        intra_structure_distances = (
            intra_structure_distances * effective_mask[None, None, ...]
        )

    # Count the number of clashes
    num_clash = (
        torch.sum(
            (intra_structure_distances < clash_dis) & (intra_structure_distances > eps)
        )
        / intra_structure_distances.size(0)
        / intra_structure_distances.size(1)
    )

    # Calculate the penalty based on the number of clashes
    penalty = torch.where(
        num_clash <= clash_limit,
        alpha * torch.log(1 + num_clash),
        beta * (num_clash - clash_limit) + gamma * torch.log(1 + num_clash),
    )

    loss = torch.mean(penalty)

    return loss
