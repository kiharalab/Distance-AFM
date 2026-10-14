from Loss.distance_loss import compute_sidechain_dis
import torch
from protein_utils.affine_utils import T
from protein_utils.rigid_utils import Rigid

def sidechain_loss(
    sidechain_frames: torch.Tensor,
    sidechain_atom_pos: torch.Tensor,
    rigidgroups_gt_frames: torch.Tensor,
    rigidgroups_alt_gt_frames: torch.Tensor,
    rigidgroups_gt_exists: torch.Tensor,
    renamed_atom14_gt_positions: torch.Tensor,
    renamed_atom14_gt_exists: torch.Tensor,
    alt_naming_is_better: torch.Tensor,
    dist_gt: torch.Tensor,
    dist_upper_bound_gt: torch.Tensor,
    dist_lower_bound_gt: torch.Tensor,
    mask_window: torch.Tensor,
    clamp_distance: float = 10.0,
    length_scale: float = 10.0,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:
    renamed_gt_frames = (
                            1.0 - alt_naming_is_better[..., None, None, None]
                        ) * rigidgroups_gt_frames + alt_naming_is_better[
                            ..., None, None, None
                        ] * rigidgroups_alt_gt_frames

    # Steamroll the inputs
    sidechain_frames = sidechain_frames[-1]
    batch_dims = sidechain_frames.shape[:-4]
    sidechain_frames = sidechain_frames.view(*batch_dims, -1, 4, 4)
    sidechain_frames = Rigid.from_tensor_4x4(sidechain_frames)
    renamed_gt_frames = renamed_gt_frames.view(*batch_dims, -1, 4, 4)
    renamed_gt_frames = Rigid.from_tensor_4x4(renamed_gt_frames)
    rigidgroups_gt_exists = rigidgroups_gt_exists.reshape(*batch_dims, -1)
    sidechain_atom_pos = sidechain_atom_pos[-1]
    sidechain_atom_pos = sidechain_atom_pos.view(*batch_dims, -1, 3)
    renamed_atom14_gt_positions = renamed_atom14_gt_positions.view(
        *batch_dims, -1, 3
    )
    renamed_atom14_gt_exists = renamed_atom14_gt_exists.view(*batch_dims, -1)

    fape = compute_sidechain_dis(
        sidechain_frames,
        renamed_gt_frames,
        rigidgroups_gt_exists,
        sidechain_atom_pos,
        renamed_atom14_gt_positions,
        renamed_atom14_gt_exists,
        pair_mask=None,
        l1_clamp_distance=clamp_distance,
        length_scale=length_scale,
        eps=eps,
        dist_gt=dist_gt,
        dist_upper_bound_gt=dist_upper_bound_gt,
        dist_lower_bound_gt=dist_lower_bound_gt,
        domain_window=mask_window
    )

    return fape