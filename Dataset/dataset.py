from torch.utils.data.dataset import Dataset
import numpy as np
import torch
import os
import random
from utils.read_pdb_info import *
from protein_utils import protein, all_atom, data_transforms
from protein_utils.data_transforms import atom37_to_torsion_angles
import math


# multimer dataset definition, main change: chain index is added
class DistAF_MultimerDataset(Dataset):
    def __init__(self, args=None, transform=None):
        self.train_file = args.target_file

        with open(self.train_file, "r") as f:
            self.targets = f.read().splitlines()

        self.max_len = args.max_len
        self.embedding_file = args.emd_file
        self.fasta_file = args.fasta_file
        self.initial_pdb = args.initial_pdb
        self.start_pdb = args.start_pdb
        self.window_info = args.window_info
        self.dist_constraint_file = args.dist_info
        self.output_dir_parent = args.output_dir
        self.output_dir = args.output_dir
        self.args = args
        self.target_seq_len = 0
        self.start_position = 0
        self.end_position = self.start_position + self.target_seq_len
        self.transform = transform
        self.dist_range = args.dist_range
        if args.selected_chains:
            self.selected_chains = list(args.selected_chains)
        else:
            self.selected_chains = None

    def __len__(self):
        return len(self.targets)

    def process_pdb_file(self, pdb_path, selected_chains=None):
        """
        Process PDB file to get protein object and related dictionaries
        Args:
            pdb_path: path to PDB file
            selected_chains: list of chains to process, if None process all chains
        Returns:
            prot: protein object
            all_dict: combined dictionary with both protein and frame features
        """
        # Read PDB file
        if pdb_path is None:
            return None, None

        with open(pdb_path, "r") as f:
            pdb_str = f.read()

        # Create protein object
        prot = protein.from_pdb_string(pdb_str, selected_chains)

        # Create protein dictionary
        all_dict = {
            "aatype": np.asarray(prot.aatype),
            "all_atom_positions": np.asarray(prot.atom_positions),
            "all_atom_mask": np.asarray(prot.atom_mask),
            "chain_index": np.asarray(prot.chain_index),
        }
        all_dict.update(all_atom.make_atom14_positions(all_dict))

        # Process frames and update dictionary
        frames_dict = data_transforms.atom37_to_frames(
            {
                "aatype": torch.tensor(prot.aatype),
                "all_atom_positions": torch.tensor(prot.atom_positions),
                "all_atom_mask": torch.tensor(prot.atom_mask),
            }
        )
        frames_dict = data_transforms.atom37_to_torsion_angles(frames_dict)

        residue_index = torch.tensor(prot.residue_index)
        frames_dict.update(
            {
                "residue_index": torch.as_tensor(residue_index).clone().detach(),
                "chi_angles_sin_cos": frames_dict["torsion_angles_sin_cos"][..., 3:, :],
                "chi_mask": frames_dict["torsion_angles_mask"][..., 3:],
                "seq_mask": torch.ones(
                    frames_dict["aatype"].shape, dtype=torch.float32
                ),
            }
        )

        # Merge all features
        all_dict.update(
            {
                k: v.clone().detach() if isinstance(v, torch.Tensor) else v
                for k, v in frames_dict.items()
            }
        )

        return prot, all_dict

    def __getitem__(self, index):
        target = self.targets[index]
        self.output_dir = os.path.join(self.output_dir_parent, target)

        # Process initial and fake ground truth structures

        prot, all_dict = self.process_pdb_file(self.initial_pdb, self.selected_chains)
        fake_gt_prot, fake_gt_dict = self.process_pdb_file(
            self.start_pdb, self.selected_chains
        )

        self.target_seq_len = len(prot.aatype)
        self.end_position = self.target_seq_len

        data = {}
        data["target"] = target

        if self.embedding_file.endswith(".npz"):
            emd = np.load(self.embedding_file)
        else:
            import pickle

            with open(self.embedding_file, "rb") as f:
                emd = pickle.load(f)["representations"]

        pair = emd["pair"]
        single = emd["single"]

        single = torch.tensor(single)
        pair = torch.tensor(pair)

        resolution = 1

        data["resolution"] = torch.tensor([resolution])

        coords = read_pdb_info(self.initial_pdb, self.target_seq_len)
        mask = np.ones(self.target_seq_len)

        data["single_representation"] = single
        single_emd = single.unsqueeze(0)
        pair_emd = pair.unsqueeze(0)

        data["single_representation"] = single_emd
        data["embed"] = pair_emd

        data["coords"] = coords
        data["mask"] = mask

        """all atom attribute load using DeepMind's utils"""
        data["aatype"] = torch.tensor(np.asarray(prot.aatype))
        for key in all_dict.keys():
            data[key] = all_dict[key]

        # Add fake ground truth data
        data["fake_gt_frames"] = fake_gt_dict

        data["seq_length"] = self.target_seq_len

        data["aatype_start"] = 0
        data["domain_window"] = None
        data["dist_constraint_lower_bound"] = None
        data["dist_constraint_upper_bound"] = None
        data["dist_constraint"] = None

        # variables needed in distance af defined below
        if os.path.exists(
            os.path.join(self.output_dir, f"{target}_constraint.pt")
        ) or os.path.exists(
            os.path.join(self.output_dir, f"{target}_constraint_lower_bound.pt")
        ):
            data["domain_window"] = torch.load(
                os.path.join(self.output_dir, f"{target}_domain_window.pt")
            )
            if self.dist_range:
                data["dist_constraint_lower_bound"] = torch.load(
                    os.path.join(self.output_dir, f"{target}_constraint_lower_bound.pt")
                )
                data["dist_constraint_upper_bound"] = torch.load(
                    os.path.join(self.output_dir, f"{target}_constraint_upper_bound.pt")
                )
            else:
                data["dist_constraint"] = torch.load(
                    os.path.join(self.output_dir, f"{target}_constraint.pt")
                )

        else:
            domain_window = torch.zeros((self.target_seq_len, self.target_seq_len))
            with open(self.window_info, "r") as file:
                lines = file.readlines()
            for line in lines:
                line = line.strip()
                start = int(line.split(",")[0]) - 1
                end = int(line.split(",")[1]) + 1
                domain_window[start:end, start:end] = 1
            data["domain_window"] = domain_window

            if self.dist_range:
                dist_constraint_lower_bound = torch.zeros(
                    (self.target_seq_len, self.target_seq_len)
                )
                dist_constraint_upper_bound = torch.zeros(
                    (self.target_seq_len, self.target_seq_len)
                )
            else:
                dist_constraint = torch.zeros(
                    (self.target_seq_len, self.target_seq_len)
                )
            with open(self.dist_constraint_file, "r") as file:
                lines = file.readlines()
            for line in lines:
                line = line.strip()
                first_resi = int(line.split(",")[0]) - 1
                second_resi = int(line.split(",")[1]) - 1
                if self.dist_range:
                    # if use the distance range constraint
                    if line.split(",")[2] == "-inf" or line.split(",")[2] == "-INF":
                        dist_cons_lower = 0.0
                    else:
                        dist_cons_lower = float(line.split(",")[2])

                    if line.split(",")[3] == "inf" or line.split(",")[3] == "INF":
                        dist_cons_upper = float("inf")
                    else:
                        dist_cons_upper = float(line.split(",")[3])
                    dist_constraint_upper_bound[first_resi, second_resi] = (
                        dist_cons_upper
                    )
                    dist_constraint_upper_bound[second_resi, first_resi] = (
                        dist_cons_upper
                    )
                    dist_constraint_lower_bound[second_resi, first_resi] = (
                        dist_cons_lower
                    )
                    dist_constraint_lower_bound[first_resi, second_resi] = (
                        dist_cons_lower
                    )
                else:
                    dist_cons = float(line.split(",")[2])
                    dist_constraint[first_resi, second_resi] = dist_cons
                    dist_constraint[second_resi, first_resi] = dist_cons

            if self.dist_range:
                print(self.output_dir)
                data["dist_constraint_lower_bound"] = dist_constraint_lower_bound
                data["dist_constraint_upper_bound"] = dist_constraint_upper_bound
                torch.save(
                    data["dist_constraint_lower_bound"],
                    os.path.join(
                        self.output_dir, f"{target}_constraint_lower_bound.pt"
                    ),
                )
                torch.save(
                    data["dist_constraint_upper_bound"],
                    os.path.join(
                        self.output_dir, f"{target}_constraint_upper_bound.pt"
                    ),
                )
            else:
                data["dist_constraint"] = dist_constraint
                torch.save(
                    data["dist_constraint"],
                    os.path.join(self.output_dir, f"{target}_constraint.pt"),
                )
            torch.save(
                data["domain_window"],
                os.path.join(self.output_dir, f"{target}_domain_window.pt"),
            )

        return data

    def generate_intermediate_pdbs(self, initial_pdb, target_pdb, output_dir):
        """
        Generate intermediate PDBs between initial and target structures.
        Automatically aligns first chains and rotates second chains.

        Args:
            initial_pdb: path to initial PDB file
            target_pdb: path to target PDB file
            output_dir: directory to save intermediate PDBs

        Returns:
            List of paths to generated intermediate PDBs
        """
        from pymol import cmd
        from train_utils.orientation import angle_between_domains
        import math
        import re

        # Initialize PyMOL and load structures
        cmd.reinitialize()
        cmd.load(initial_pdb, "initial")
        cmd.load(target_pdb, "target")

        # Get chain IDs for both structures
        initial_chains = cmd.get_chains("initial")
        target_chains = cmd.get_chains("target")

        if len(initial_chains) < 2 or len(target_chains) < 2:
            raise ValueError("Both structures must have at least 2 chains")

        # Use first chains for alignment
        chain_to_align_initial = initial_chains[0]
        chain_to_align_target = target_chains[0]

        # Use second chains for rotation
        chain_to_rotate_initial = initial_chains[1]
        chain_to_rotate_target = target_chains[1]

        # Print chain information
        print(
            f"Aligning initial chain {chain_to_align_initial} to target chain {chain_to_align_target}"
        )
        print(
            f"Calculating rotation for initial chain {chain_to_rotate_initial} to target chain {chain_to_rotate_target}"
        )

        # Align the first chains
        cmd.align(
            f"initial and chain {chain_to_align_initial}",
            f"target and chain {chain_to_align_target}",
        )

        # Calculate rotation angle using orientation module
        result = angle_between_domains(
            f"initial and chain {chain_to_rotate_initial}",
            f"target and chain {chain_to_rotate_target}",
            method="align",
            visualize=1,
            quiet=0,
        )

        angle = result[0]
        displacement = result[1]
        rotation_axis = result[2]
        print("angle", angle)
        print("displacement", displacement)
        print("rotation_axis", rotation_axis[0], rotation_axis[1], rotation_axis[2])

        cmd.select("rotating_chain", f"initial and chain {chain_to_rotate_initial}")
        center = cmd.centerofmass("rotating_chain")

        # Generate two intermediate states (1/3 and 2/3)
        intermediate_pdbs = []
        for fraction in [1 / 3, 2 / 3]:
            # Create intermediate structure
            intermediate_name = f"intermediate_{int(fraction * 100)}"
            cmd.create(intermediate_name, "initial")

            # Calculate intermediate angle
            intermediate_angle = angle * fraction

            # Apply rotation around the orientation axis
            cmd.rotate(
                axis=[
                    float(rotation_axis[0]),
                    float(rotation_axis[1]),
                    float(rotation_axis[2]),
                ],  # First three elements represent the axis vector
                # axis=[0, 0, 1],
                angle=intermediate_angle,
                selection=f"{intermediate_name} and chain {chain_to_rotate_initial}",
                camera=0,
                origin=center,
            )

            # # Apply displacement after rotation
            # cmd.translate(
            #     [d * fraction for d in displacement],
            #     f"{intermediate_name} and chain {chain_to_rotate_initial}"
            # )

            # Save intermediate PDB
            output_path = os.path.join(output_dir, f"{intermediate_name}.pdb")
            cmd.save(output_path, intermediate_name)
            intermediate_pdbs.append(output_path)

        cmd.reinitialize()
        return intermediate_pdbs
