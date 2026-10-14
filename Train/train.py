from torch.utils.data import DataLoader
from Dataset.dataset import DistAF_MultimerDataset
import torch.optim as optim
import torch
import os
from Loss.fape_loss import fape_loss
from Loss.openfold_loss import (
    compute_renamed_ground_truth,
    find_structural_violations,
    chain_center_of_mass_loss,
)
from train_utils.feats import atom14_to_atom37
from protein_utils import protein
from utils import rmsd
import numpy as np
import functools
from utils.set_seed import set_seed
from train_utils.collate import collate_fn
import gc
from utils.model_checkpoint import checkpoint_model as ckpt
from config import model_config
from train_utils.generate_start_structures import SurfaceDistanceAnalyzer


def load_pretrained_weights(model, weight_path, device_id, strict=False):
    """
    Load pretrained weights from a checkpoint file
    Args:
        model: The model to load weights into
        weight_path: Path to the weight file
        device_id: Device to load weights to
        strict: Whether to strictly enforce that the keys in state_dict match the keys returned by model's state_dict() function
    """
    print(f"Loading pretrained weights from {weight_path}")
    if weight_path.endswith(".pth"):
        # Load from checkpoint file
        checkpoint = torch.load(weight_path, map_location=device_id)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        # Load from state dict file
        state_dict = torch.load(weight_path, map_location=device_id)

    # Handle key mapping for compatibility
    key_mapping = {
        "structure_module.ipa.linear_q_points.weight": "structure_module.ipa.linear_q_points.linear.weight",
        "structure_module.ipa.linear_q_points.bias": "structure_module.ipa.linear_q_points.linear.bias",
        "structure_module.ipa.linear_kv_points.weight": "structure_module.ipa.linear_kv_points.linear.weight",
        "structure_module.ipa.linear_kv_points.bias": "structure_module.ipa.linear_kv_points.linear.bias",
    }

    # Rename keys in state dict
    for old_key, new_key in key_mapping.items():
        if old_key in state_dict:
            state_dict[new_key] = state_dict.pop(old_key)

    # Load state dict with strict=False to allow partial loading
    model.load_state_dict(state_dict, strict=strict)
    print("Pretrained weights loaded successfully")


def train(args):
    skip_weights = False

    set_seed(args)
    with open(args.target_file, "r") as f:
        target_name = f.read().splitlines()[0]
    target_output_dir = os.path.join(args.output_dir, target_name)
    if not os.path.exists(target_output_dir):
        os.makedirs(target_output_dir)
    # multimer dataset usage
    # load config
    config = model_config(
        args.config_preset,
        train=True,
        # low_prec=(str(args.precision) == "16")
    )
    if not config.globals.is_multimer:
        raise ValueError(
            f"config_preset '{args.config_preset}' is not a multimer preset; "
            "this release supports multimer presets only."
        )
    from model.alphafold_finetune_multimer import AlphaFold_Multimer

    # train_dataset = DistAF_MultimerDataset(args, transform=augment_data_with_rotation)
    train_dataset = DistAF_MultimerDataset(args)
    # train_dataset = AlphaFold_Multimer(args)
    args.training_examples = len(train_dataset)
    collate = functools.partial(collate_fn, args=args)
    # set num_workers to 0 to avoid issues with multiprocessing, need further improvement for jax
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )
    model = AlphaFold_Multimer(args)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    model.to(args.device_id)
    if args.model_dir:
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        if os.path.exists(f"{args.model_dir}/checkpoint.pth"):
            checkpoint = torch.load(
                f"{args.model_dir}/checkpoint.pth", map_location=args.device_id
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            # Skip loading optimizer state if parameter groups don't match
            try:
                optimizer_state_dict = checkpoint["optimizer_state_dict"]
                optimizer.load_state_dict(optimizer_state_dict)
                starting_epoch = checkpoint["epoch"]
            except ValueError:
                print(
                    "Warning: Optimizer state dict size mismatch. Using default optimizer state."
                )
                starting_epoch = 0
        else:
            starting_epoch = 0
            # Load weights using the new function if skip_weights is False
            if not skip_weights:
                load_pretrained_weights(
                    model, f"{args.model_dir}/model_state_dict.pt", args.device_id
                )
                # Skip loading optimizer state if parameter groups don't match
                try:
                    optimizer_state_dict = torch.load(
                        f"{args.model_dir}/optimizer.pt", map_location=args.device_id
                    )
                    for key in optimizer_state_dict.keys():
                        optimizer_state_dict[key] = (
                            optimizer_state_dict[key].to(args.device_id)
                            if isinstance(optimizer_state_dict[key], torch.Tensor)
                            else optimizer_state_dict[key]
                        )
                    optimizer.load_state_dict(optimizer_state_dict)
                except ValueError:
                    print(
                        "Warning: Optimizer state dict size mismatch. Using default optimizer state."
                    )
            else:
                print(
                    "Skipping loading pretrained weights. Using random initialization."
                )
        print(f"Checkpoints (model and optimizer) loaded from {args.model_dir}")
    else:
        starting_epoch = 0
        if args.weight_file and not skip_weights:
            load_pretrained_weights(model, args.weight_file, args.device_id)
        elif not skip_weights:
            load_pretrained_weights(
                model,
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "model_dir",
                    "multimer",
                    "model_state_dict.pt",
                ),
                args.device_id,
            )
        else:
            print("Skipping loading pretrained weights. Using random initialization.")

    print("----------------- Starting Training ---------------")
    print("  Num examples = %d" % (int(args.training_examples)))
    print("  Num Epochs = %d" % (int(args.epochs)))
    print("  Batch Size = %d" % (int(args.batch)))

    model.train()
    # casp_results = test(args, model, test_dataloader)
    # results = val(args, model, val_dataloader)
    gt_keys = [
        "all_atom_positions",
        "all_atom_mask",
        "atom14_atom_exists",
        "atom14_gt_exists",
        "atom14_gt_positions",
        "residx_atom14_to_atom37",
        "residx_atom37_to_atom14",
        "atom37_atom_exists",
        "atom14_alt_gt_positions",
        "atom14_alt_gt_exists",
        "atom14_atom_is_ambiguous",
        "residue_index",
    ]
    gt_frames_keys = [
        "atom14_atom_exists",
        "rigidgroups_gt_frames",
        "rigidgroups_gt_exists",
        "rigidgroups_group_exists",
        "rigidgroups_group_is_ambiguous",
        "rigidgroups_alt_gt_frames",
        "torsion_angles_sin_cos",
        "alt_torsion_angles_sin_cos",
        "torsion_angles_mask",
        "chi_angles_sin_cos",
        "chi_mask",
        "seq_mask",
    ]

    best_loss = float("inf")
    patience = 1000  # Set patience value
    window_size = 10  # Size of the sliding window for loss averaging
    loss_history = []
    no_improvement_count = 0
    coarse_tuning_epochs = args.coarse_tuning_epochs
    fine_tuning_start_idx = (
        coarse_tuning_epochs  # Initialize to avoid UnboundLocalError
    )
    improvement_threshold = (
        0.001  # Threshold for considering an improvement significant
    )

    loss_log_file = os.path.join(target_output_dir, "loss_log.txt")
    # if os.path.exists(loss_log_file):
    #     os.remove(loss_log_file)

    for epoch in range(starting_epoch + 1, args.epochs + 1):
        epoch_loss = 0
        num_batches = 0

        for step, (batch, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            embedding = batch["embed"]
            domain_window = batch["domain_window"].squeeze(0)
            if args.dist_range:
                dist_constraint_lower_bound = batch[
                    "dist_constraint_lower_bound"
                ].squeeze(0)
                dist_constraint_upper_bound = batch[
                    "dist_constraint_upper_bound"
                ].squeeze(0)
            else:
                dist_constraint = batch["dist_constraint"].squeeze(0)
            single_repr_batch = batch["single_representation"]

            aatype_batch = batch["aatype"]
            batch_gt = {key: batch[key] for key in gt_keys}
            batch_gt_frames = {key: batch[key] for key in gt_frames_keys}
            # coarse tuning with fake gt frames, TODO: make it more official
            if (
                args.start_pdb is not None
                and epoch <= coarse_tuning_epochs
                and batch["fake_gt_frames"] is not None
            ):
                print(f"Coarse tuning towards the start structure at epoch {epoch}")
                fake_gt_frames = {
                    key: batch["fake_gt_frames"][key] for key in gt_frames_keys
                }
            else:
                fake_gt_frames = None
            initial_pdb = args.initial_pdb
            with open(initial_pdb, "r") as f:
                initial_pdb_str = f.read()
            prot_initial = protein.from_pdb_string(initial_pdb_str)
            if "asym_id" not in batch_gt_frames.keys():
                batch_gt_frames.update({"asym_id": prot_initial.chain_index})
            if "asym_id" not in batch_gt.keys():
                batch_gt.update({"asym_id": prot_initial.chain_index})
            if fake_gt_frames is not None:
                for key in fake_gt_frames.keys():
                    if isinstance(fake_gt_frames[key], np.ndarray):
                        fake_gt_frames[key] = torch.from_numpy(fake_gt_frames[key])
                    fake_gt_frames[key] = fake_gt_frames[key].to(args.device_id)

            batch_gt.update({"seq_length": batch["seq_length"]})
            batch_gt.update({"aatype": aatype_batch})
            batch_gt_frames.update({"aatype": aatype_batch})
            if fake_gt_frames is not None:
                fake_gt_frames.update({"aatype": aatype_batch})
            resolution = batch["resolution"]
            representation = None

            # TODO: gt_features extracted to used in permutation align
            # if "gt_features" not in batch.keys() and "multimer" in args.config_preset:
            # batch_gt = prepare_ground_truth_features(batch)
            # prepare ground truth features that are only needed for loss calculation during training
            # gt_tensors = prepare_ground_truth_features(batch)

            if args.cuda:
                embedding = embedding.to(args.device_id)
                resolution = resolution.to(args.device_id)
                for key in batch_gt.keys():
                    if isinstance(batch_gt[key], np.ndarray):
                        batch_gt[key] = torch.from_numpy(batch_gt[key])
                    batch_gt[key] = batch_gt[key].to(args.device_id)
                for key in batch_gt_frames.keys():
                    if isinstance(batch_gt_frames[key], np.ndarray):
                        batch_gt_frames[key] = torch.from_numpy(batch_gt_frames[key])
                    batch_gt_frames[key] = batch_gt_frames[key].to(args.device_id)
                if fake_gt_frames is not None:
                    for key in fake_gt_frames.keys():
                        if isinstance(fake_gt_frames[key], np.ndarray):
                            fake_gt_frames[key] = torch.from_numpy(fake_gt_frames[key])
                    fake_gt_frames[key] = fake_gt_frames[key].to(args.device_id)
                single_repr_batch = single_repr_batch.to(args.device_id)
                # coords_batch = coords_batch.cuda(args.device_id)
                # masks_batch = masks_batch.cuda(args.device_id)
                aatype_batch = aatype_batch.to(args.device_id)
                domain_window = domain_window.to(args.device_id)
                if args.dist_range:
                    dist_constraint_lower_bound = dist_constraint_lower_bound.to(
                        args.device_id
                    )
                    dist_constraint_upper_bound = dist_constraint_upper_bound.to(
                        args.device_id
                    )
                else:
                    dist_constraint = dist_constraint.to(args.device_id)
            dummy = torch.Tensor(1)
            dummy.requires_grad = True

            # # Apply random rotation to each chain
            # for chain in batch_gt_frames.keys():
            #     rotation_matrix = random_rotation_matrix()
            #     batch_gt_frames[chain] = rotate_coordinates(batch_gt_frames[chain], rotation_matrix)

            if args.use_checkpoint:

                def run_ckpt(
                    model,
                    embedding,
                    single_repr_batch,
                    aatype_batch,
                    batch_gt_frames,
                    dummy,
                ):
                    # return model(embedding, single_repr_batch, aatype_batch, batch_gt_frames)
                    return model(
                        embedding,
                        single_repr_batch,
                        aatype_batch,
                        batch_gt,
                        batch_gt_frames,
                        resolution,
                        representation=representation,
                    )

                outputs, pred_frames = ckpt(
                    run_ckpt,
                    model,
                    embedding,
                    single_repr_batch,
                    aatype_batch,
                    batch_gt_frames,
                    dummy,
                )
            else:
                outputs, pred_frames = model(
                    embedding, single_repr_batch, aatype_batch, batch_gt_frames
                )

            if "renamed_atom14_gt_positions" not in outputs.keys():
                batch_gt_frames.update(
                    compute_renamed_ground_truth(
                        batch_gt,
                        outputs["positions"][-1],
                    )
                )
                if fake_gt_frames is not None:
                    fake_gt_frames.update(
                        compute_renamed_ground_truth(
                            batch_gt,
                            torch.from_numpy(
                                batch["fake_gt_frames"]["atom14_gt_positions"]
                            ).to(args.device_id),
                        )
                    )
            if "violation" not in outputs.keys():
                outputs["violation"] = find_structural_violations(
                    batch_gt,
                    outputs["positions"][-1],
                    **config.loss.violation,
                )

            # compute all needed loss
            seq_len = torch.mean(batch_gt["seq_length"].float())
            crop_len = torch.tensor(aatype_batch.shape[-1]).to(
                device=aatype_batch.device
            )

            # fape/fafe loss
            if args.dist_range:
                (
                    fape,
                    weighted_bb_loss,
                    weighted_dis_loss,
                    weighted_clash_loss,
                    sc_loss,
                    vio_loss,
                    angle_loss,
                ) = fape_loss(
                    out=outputs,
                    batch=batch_gt_frames,
                    traj=pred_frames,
                    dist_lower_bound_gt=dist_constraint_lower_bound,
                    dist_upper_bound_gt=dist_constraint_upper_bound,
                    domain_window=domain_window,
                    config=config.loss,
                    dist=args.dist,
                    dist_range=args.dist_range,
                    dist_weight=args.dist_weight,
                    loose_dist=args.loose_dist,
                    seq_len=seq_len,
                    crop_len=crop_len,
                    using_fafe=args.using_fafe,
                    fake_gt_frames=fake_gt_frames,
                )
            else:
                (
                    fape,
                    weighted_bb_loss,
                    weighted_dis_loss,
                    weighted_clash_loss,
                    sc_loss,
                    vio_loss,
                    angle_loss,
                ) = fape_loss(
                    out=outputs,
                    batch=batch_gt_frames,
                    traj=pred_frames,
                    dist_gt=dist_constraint,
                    domain_window=domain_window,
                    config=config.loss,
                    dist=args.dist,
                    dist_range=args.dist_range,
                    dist_weight=args.dist_weight,
                    loose_dist=args.loose_dist,
                    seq_len=seq_len,
                    crop_len=crop_len,
                    using_fafe=args.using_fafe,
                    fake_gt_frames=fake_gt_frames,
                )

            # chain center of mass loss
            chain_center_of_mass = 0
            if config.loss.chain_center_of_mass.enabled:
                chain_center_of_mass = chain_center_of_mass_loss(
                    all_atom_pred_pos=outputs["positions"][-1],
                    **{**batch_gt, **config.loss.chain_center_of_mass},
                )
                fape += (
                    chain_center_of_mass
                    * config.loss.chain_center_of_mass.weight
                    * torch.sqrt(min(seq_len, crop_len))
                )

            positions = outputs["positions"][-1]
            del outputs
            del pred_frames

            gc.collect()
            torch.cuda.empty_cache()
            print(f"Epoch:{epoch}, loss:{fape.item()}")
            # Record loss to file
            with open(loss_log_file, "a") as log_file:
                log_file.write(f"Epoch:{epoch}, loss:{fape.item()}\n")
                log_file.write(
                    f"Epoch:{epoch}, weighted_bb_loss:{weighted_bb_loss.item()}\n"
                )
                log_file.write(
                    f"Epoch:{epoch}, weighted_dis_loss:{weighted_dis_loss.item()}\n"
                )
                log_file.write(
                    f"Epoch:{epoch}, weighted_clash_loss:{weighted_clash_loss.item()}\n"
                )
                log_file.write(f"Epoch:{epoch}, sc_loss:{sc_loss.item()}\n")
                log_file.write(f"Epoch:{epoch}, vio_loss:{vio_loss.item()}\n")
                log_file.write(f"Epoch:{epoch}, angle_loss:{angle_loss.item()}\n")
                log_file.write(
                    f"Epoch:{epoch}, chain_center_of_mass:{chain_center_of_mass.item()}\n"
                )
            loss = fape.item()

            # check the coarse tuning result
            if (
                "fake_gt_frames" in batch.keys()
                and epoch == coarse_tuning_epochs
                and args.start_pdb is not None
            ):
                if loss >= 100:
                    print(f"Coarse tuning failed at epoch {epoch}")
                    exit()

            epoch_loss += loss
            num_batches += 1

            # fape.backward()
            fape.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()
            # save model checkpoint
            if args.val_epochs > 0 and epoch % args.val_epochs == 0 and epoch > 0:
                epoch_output_dir = os.path.join(
                    target_output_dir, f"checkpoint-{epoch}-{epoch}"
                )
                if not os.path.exists(epoch_output_dir):
                    os.makedirs(epoch_output_dir)
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "rng_state": torch.get_rng_state(),
                }
                torch.save(checkpoint, os.path.join(epoch_output_dir, "checkpoint.pth"))
                torch.save(
                    model.state_dict(),
                    os.path.join(epoch_output_dir, "model_state_dict.pt"),
                )
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(epoch_output_dir, "optimizer.pt"),
                )
                # save predicted pdb for each evaluted epoch
                final_pos = atom14_to_atom37(positions, batch_gt)
                final_atom_mask = batch_gt["atom37_atom_exists"]

                pred_prot = protein.Protein(
                    aatype=aatype_batch.squeeze(0).cpu().numpy(),
                    atom_positions=final_pos.squeeze(0).detach().cpu().numpy(),
                    atom_mask=final_atom_mask.squeeze(0).cpu().numpy(),
                    residue_index=prot_initial.residue_index,
                    b_factors=prot_initial.b_factors,
                    chain_index=prot_initial.chain_index,
                )
                pred_pdb_lines = protein.to_pdb(pred_prot)

                output_dir_pred = os.path.join(
                    epoch_output_dir, f"{target_name}_pred_epoch{epoch}.pdb"
                )
                with open(output_dir_pred, "w") as f:
                    f.write(pred_pdb_lines)

                # compute RMSD between initial and epoch predicted pdb
                gt = prot_initial.atom_positions[None, ...][
                    prot_initial.atom_mask[None, ...].astype(bool)
                ]
                pred = pred_prot.atom_positions[None, ...][
                    prot_initial.atom_mask[None, ...].astype(bool)
                ]
                gt = gt.reshape(-1, 3)
                pred = pred.reshape(-1, 3)
                gt -= rmsd.centroid(gt)
                pred -= rmsd.centroid(pred)
                U = rmsd.kabsch(gt, pred)
                A = np.dot(gt, U)
                rmsd_value = rmsd.rmsd(A, pred)
                with open(os.path.join(epoch_output_dir, f"rmsd.txt"), "w") as file:
                    file.write(f"Epoch: {epoch}\n")
                    file.write(f"RMSD: {rmsd_value}\n")
                    file.write(f"Loss: {loss}\n")

        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / num_batches
        loss_history.append(avg_epoch_loss)

        # If we have enough history, calculate the moving average
        if len(loss_history) >= window_size:
            # After epoch coarse_tuning_epochs, only consider losses from fine tuning phase
            if epoch > coarse_tuning_epochs:
                recent_losses = loss_history[fine_tuning_start_idx:]
                if len(recent_losses) >= window_size:
                    current_avg = sum(recent_losses[-window_size:]) / window_size
                    if len(recent_losses) > window_size:
                        previous_avg = (
                            sum(recent_losses[-window_size - 1 : -1]) / window_size
                        )

                        # Check if there's a significant improvement
                        if (
                            previous_avg - current_avg
                        ) / previous_avg > improvement_threshold:
                            no_improvement_count = 0
                        else:
                            no_improvement_count += 1
                else:
                    # Not enough recent losses for fine tuning, use current epoch loss
                    current_avg = avg_epoch_loss
            else:
                current_avg = sum(loss_history[-window_size:]) / window_size
                if len(loss_history) > window_size:
                    previous_avg = (
                        sum(loss_history[-window_size - 1 : -1]) / window_size
                    )

                    # Check if there's a significant improvement
                    if (
                        previous_avg - current_avg
                    ) / previous_avg > improvement_threshold:
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1

            # Update best loss
            if current_avg < best_loss:
                best_loss = current_avg
                best_epoch = epoch

                # Save best model checkpoint
                best_checkpoint_dir = os.path.join(target_output_dir, "best_checkpoint")
                if not os.path.exists(best_checkpoint_dir):
                    os.makedirs(best_checkpoint_dir)

                # Save checkpoint
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": best_loss,
                }
                torch.save(
                    checkpoint, os.path.join(best_checkpoint_dir, "best_checkpoint.pth")
                )
                torch.save(
                    model.state_dict(),
                    os.path.join(best_checkpoint_dir, "model_state_dict.pt"),
                )
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(best_checkpoint_dir, "optimizer.pt"),
                )

                # Save predicted pdb
                final_pos = atom14_to_atom37(positions, batch_gt)
                final_atom_mask = batch_gt["atom37_atom_exists"]

                pred_prot = protein.Protein(
                    aatype=aatype_batch.squeeze(0).cpu().numpy(),
                    atom_positions=final_pos.squeeze(0).detach().cpu().numpy(),
                    atom_mask=final_atom_mask.squeeze(0).cpu().numpy(),
                    residue_index=prot_initial.residue_index,
                    b_factors=prot_initial.b_factors,
                    chain_index=prot_initial.chain_index,
                )
                pred_pdb_lines = protein.to_pdb(pred_prot)

                output_dir_pred = os.path.join(
                    best_checkpoint_dir, f"{target_name}_best_pred.pdb"
                )
                with open(output_dir_pred, "w") as f:
                    f.write(pred_pdb_lines)

                # Compute RMSD
                gt = prot_initial.atom_positions[None, ...][
                    prot_initial.atom_mask[None, ...].astype(bool)
                ]
                pred = pred_prot.atom_positions[None, ...][
                    prot_initial.atom_mask[None, ...].astype(bool)
                ]
                gt = gt.reshape(-1, 3)
                pred = pred.reshape(-1, 3)
                gt -= rmsd.centroid(gt)
                pred -= rmsd.centroid(pred)
                U = rmsd.kabsch(gt, pred)
                A = np.dot(gt, U)
                rmsd_value = rmsd.rmsd(A, pred)
                with open(
                    os.path.join(best_checkpoint_dir, "best_results.txt"), "w"
                ) as file:
                    file.write(f"Best Epoch: {epoch}\n")
                    file.write(f"Best RMSD: {rmsd_value}\n")
                    file.write(f"Best Loss: {best_loss}\n")

        print(f"Epoch: {epoch}, Average Loss: {avg_epoch_loss}, Best Loss: {best_loss}")

        # Check if early stopping should be triggered
        if (
            no_improvement_count >= patience
            and fake_gt_frames is None
            and epoch > coarse_tuning_epochs
        ):
            print(
                f"Early stopping triggered. No significant improvement for {patience} epochs."
            )
            break

        if epoch == coarse_tuning_epochs and args.start_pdb is not None:
            # coarse tuning finished, reset loss tracking while preserving history
            print(
                f"Coarse tuning finished at epoch {coarse_tuning_epochs}. Resetting loss tracking for fine tuning phase."
            )
            best_loss = float("inf")
            patience = 1000
            no_improvement_count = 0
            # We keep loss_history intact but will use a separate window for fine tuning
            # fine_tuning_start_idx already initialized earlier to mark where fine tuning begins

    print(f"Best model saved at epoch {best_epoch} with loss {best_loss}")

    # Create DONE file
    with open(os.path.join(target_output_dir, "DONE.out"), "w") as done_file:
        done_file.write(
            f"Training completed. Best epoch: {best_epoch}, Best loss: {best_loss}"
        )

    return
