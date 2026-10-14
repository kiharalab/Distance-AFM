from Loss.backbone_loss import backbone_loss
from Loss.sidechain_loss import sidechain_loss
from Loss.clash_loss import clash_avoidance_loss
from typing import Dict
import torch
import ml_collections
from Loss.openfold_loss import supervised_chi_loss, violation_loss
from Loss.frame_loss import frame_loss


def fape_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    traj: torch.Tensor,
    domain_window: torch.Tensor,
    config: ml_collections.ConfigDict,
    dist=0,
    dist_range=0,
    dist_weight=0,
    dist_gt=None,
    dist_upper_bound_gt=None,
    dist_lower_bound_gt=None,
    loose_dist=0,
    seq_len=0,
    crop_len=0,
    using_fafe=0,
    fake_gt_frames=None,
) -> torch.Tensor:
    asym_id = batch.get("asym_id")

    if fake_gt_frames is not None:
        # release the domain constraint to let the model learn
        fake_domain_window = torch.ones_like(domain_window)

        fake_gt_bb_loss, fake_gt_dis_loss, fake_gt_clash_loss = backbone_loss(
            backbone_rigid_tensor=fake_gt_frames["rigidgroups_gt_frames"][..., 0, :, :],
            backbone_rigid_mask=fake_gt_frames["rigidgroups_gt_exists"][..., 0],
            traj=traj,
            dist_gt=dist_gt,
            dist_upper_bound_gt=dist_upper_bound_gt,
            dist_lower_bound_gt=dist_lower_bound_gt,
            mask_window=fake_domain_window,
            using_fafe=using_fafe,
            clamp_distance=None,
            loss_unit_distance=5.0,
            # **{**batch, **config.fape.backbone},
        )
        fake_gt_weighted_bb_loss = (
            fake_gt_bb_loss * config.fape.intra_chain_backbone.weight
        )

        fake_gt_sc_loss = sidechain_loss(
            out["sidechain_frames"],
            out["positions"],
            dist_gt=dist_gt,
            dist_upper_bound_gt=dist_upper_bound_gt,
            dist_lower_bound_gt=dist_lower_bound_gt,
            mask_window=fake_domain_window,
            **{**fake_gt_frames, **config.fape.sidechain},
        )

        fake_gt_angle_loss = supervised_chi_loss(
            out["angles"],
            out["unnormalized_angles"],
            **{**fake_gt_frames, **config.supervised_chi},
            dist=dist,
        )

        fake_gt_violation_loss_ = violation_loss(
            out["violation"],
            **{**fake_gt_frames, **config.violation},
        )

        fake_gt_violation_loss = torch.mean(fake_gt_violation_loss_)

        if fake_gt_weighted_bb_loss > 10.0:
            loss = (
                12 * fake_gt_weighted_bb_loss
                + fake_gt_sc_loss
                + fake_gt_violation_loss
                + fake_gt_angle_loss
            )
        elif fake_gt_weighted_bb_loss > 5.0 and fake_gt_weighted_bb_loss < 10.0:
            loss = (
                48 * fake_gt_weighted_bb_loss
                + fake_gt_sc_loss
                + fake_gt_violation_loss
                + fake_gt_angle_loss
            )
        else:
            loss = (
                96 * fake_gt_weighted_bb_loss
                + fake_gt_sc_loss
                + fake_gt_violation_loss
                + fake_gt_angle_loss
            )
        return (
            loss,
            fake_gt_weighted_bb_loss,
            fake_gt_dis_loss,
            fake_gt_clash_loss,
            fake_gt_sc_loss,
            fake_gt_violation_loss,
            fake_gt_angle_loss,
        )

    else:
        bb_loss, dis_loss, clash_loss = backbone_loss(
            backbone_rigid_tensor=batch["rigidgroups_gt_frames"][..., 0, :, :],
            backbone_rigid_mask=batch["rigidgroups_gt_exists"][..., 0],
            traj=traj,
            dist_gt=dist_gt,
            dist_upper_bound_gt=dist_upper_bound_gt,
            dist_lower_bound_gt=dist_lower_bound_gt,
            mask_window=domain_window,
            using_fafe=using_fafe,
            # **{**batch, **config.fape.backbone},
        )
        # weighted_bb_loss = bb_loss * config.fape.backbone.weight
        weighted_bb_loss = bb_loss * config.fape.intra_chain_backbone.weight
        # weighted_dis_loss = dis_loss * config.fape.distance.weight
        weighted_dis_loss = dis_loss
        weighted_clash_loss = clash_loss

        sc_loss = sidechain_loss(
            out["sidechain_frames"],
            out["positions"],
            dist_gt=dist_gt,
            dist_upper_bound_gt=dist_upper_bound_gt,
            dist_lower_bound_gt=dist_lower_bound_gt,
            mask_window=domain_window,
            **{**batch, **config.fape.sidechain},
        )

        angle_loss = supervised_chi_loss(
            out["angles"],
            out["unnormalized_angles"],
            **{**batch, **config.supervised_chi},
            dist=dist,
        )

        violation_loss_ = violation_loss(
            out["violation"],
            **{**batch, **config.violation},
        )
        # Average over the batch dimension
        vio_loss = torch.mean(violation_loss_)

        if weighted_dis_loss > 10.0:
            loss = 12 * weighted_dis_loss * dist_weight + (
                weighted_bb_loss + sc_loss + vio_loss + angle_loss
            ) * torch.sqrt(min(seq_len, crop_len))
        elif weighted_dis_loss > 5.0 and weighted_dis_loss < 10.0:
            loss = 24 * weighted_dis_loss * dist_weight + (
                weighted_bb_loss + sc_loss + vio_loss + angle_loss
            ) * torch.sqrt(min(seq_len, crop_len))
        else:
            if loose_dist and weighted_dis_loss < 1.0:
                loss = 12 * weighted_dis_loss * dist_weight + (
                    weighted_bb_loss + sc_loss + vio_loss + angle_loss
                ) * torch.sqrt(min(seq_len, crop_len))
            else:
                loss = 48 * weighted_dis_loss * dist_weight + (
                    weighted_bb_loss + sc_loss + vio_loss + angle_loss
                ) * torch.sqrt(min(seq_len, crop_len))

        return (
            loss,
            weighted_bb_loss,
            weighted_dis_loss,
            weighted_clash_loss,
            sc_loss,
            vio_loss,
            angle_loss,
        )
