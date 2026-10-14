import torch
import numpy as np
from typing import Dict, Optional, Tuple
from protein_utils.affine_utils import T
from protein_utils.rigid_utils import Rigid, Rotation

def frame_loss(
    pred_rigid_tensor: torch.Tensor,  # predicted rigid tensor [N, 4, 4] or [N, 7]
    ref_rots: np.ndarray,       # reference rotations [N, 3, 3]
    ref_trans: np.ndarray,   # reference translations [N, 3]
    rigid_mask: torch.Tensor,         # mask for valid positions [N]
    clamp_distance: float = 10.0,     # maximum distance threshold
    loss_unit_distance: float = 10.0, # unit distance for loss
    eps: float = 1e-4,               # small value for numerical stability
    **kwargs,
) -> torch.Tensor:
    """Calculate the loss between predicted frames and reference frames"""
    # Convert predicted tensor to Rigid object
    if pred_rigid_tensor.shape[-1] == 7:
        pred_aff = Rigid.from_tensor_7(pred_rigid_tensor)
    elif pred_rigid_tensor.shape[-1] == 4:
        pred_aff = Rigid.from_tensor_4x4(pred_rigid_tensor)
    
    device = pred_rigid_tensor.device
    
    ref_rots = torch.tensor(ref_rots, 
                          dtype=torch.float32, 
                          device=device)    # [N, 3, 3]
    ref_trans = torch.tensor(ref_trans, 
                           dtype=torch.float32, 
                           device=device)   # [N, 3]
    
    # Convert rigid_mask to boolean if it's not already
    if rigid_mask.dtype != torch.bool:
        rigid_mask = rigid_mask.bool()
    
    # Calculate rotation loss
    rot_loss = torch.sum(
        torch.square(pred_aff.get_rots().get_rot_mats() - ref_rots),
        dim=(-1, -2)
    )
    
    # Calculate translation loss
    trans_diff = pred_aff.get_trans() - ref_trans
    trans_loss = torch.sum(torch.square(trans_diff), dim=-1)
    
    if clamp_distance is not None:
        trans_loss = torch.clamp(
            trans_loss,
            max=clamp_distance ** 2
        )
    
    # Combine losses and apply mask
    loss = (rot_loss + trans_loss / (loss_unit_distance ** 2)) * rigid_mask
    
    # Average over valid positions
    loss = torch.sum(loss) / (torch.sum(rigid_mask) + eps)
    
    return loss

def frame_distance_loss(
    pred_frames: Rigid,
    ref_frames: Rigid,
    length_scale: float = 10.0,
    eps: float = 1e-4,
    **kwargs
) -> torch.Tensor:
    """Calculate the distance loss between predicted frames and reference frames
    
    Args:
        pred_frames: predicted Rigid object
        ref_frames: reference Rigid object
        length_scale: length scale
        eps: small value for numerical stability
        
    Returns:
        distance_loss: scalar loss value
    """
    # calculate the difference of positions
    diff = pred_frames.get_trans() - ref_frames.get_trans()
    distance = torch.sqrt(torch.sum(torch.square(diff), dim=-1) + eps)
    
    # normalize the distance
    normalized_distance = distance / length_scale
    
    # calculate the loss
    loss = torch.mean(normalized_distance)
    
    return loss
