import Bio.PDB
# import freesasa
import numpy as np
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import argparse
import os
import __main__
__main__.pymol_argv = ['pymol', '-qc']  # Quiet and no GUI
import sys
import pymol
from pymol import cmd
from Bio.PDB import Superimposer
import os
import shutil

def calculate_rmsd(pdb1, pdb2):
    # Initialize PyMOL in headless mode
    pymol.finish_launching()
    
    # Clear all objects in PyMOL
    cmd.delete('all')
    
    # Load structures
    cmd.load(pdb1, "struct1")
    cmd.load(pdb2, "struct2")
    
    # Align structures and get RMSD
    align_result = cmd.align("struct1", "struct2")
    rmsd = align_result[0]  # The first value in the tuple is the RMSD
    
    # Clean up
    cmd.delete("all")
    
    return rmsd

def get_sequence_window(chain, center_index, first_chain_length):
    """Get sequence window around a residue."""
    sequence = []
    start = center_index - first_chain_length
    end = center_index + first_chain_length + 1
    
    for i in range(start, end):
        try:
            residue = chain[(' ', i, ' ')]
            sequence.append(residue.resname)
        except KeyError:
            sequence.append(None)
    return sequence

def calculate_sequence_match(seq1, seq2):
    """Calculate matching score between two sequence windows."""
    if len(seq1) != len(seq2):
        return 0.0
    
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b and a is not None and b is not None)
    total = sum(1 for a, b in zip(seq1, seq2) if a is not None and b is not None)
    
    return matches / total if total > 0 else 0.0

def find_matching_residue(native_residue, predict_chain, native_chain, first_chain_length=10):
    native_index = native_residue.id[1]
    native_seq = get_sequence_window(native_chain, native_index, first_chain_length)
    
    for predict_residue in predict_chain:
        if predict_residue.id[0] != ' ' or predict_residue.resname != native_residue.resname:
            continue
            
        predict_index = predict_residue.id[1]
        predict_seq = get_sequence_window(predict_chain, predict_index, first_chain_length)
        
        match_score = calculate_sequence_match(native_seq, predict_seq)
        
        if match_score == 1.0:  # Perfect match
            return predict_residue
            
    return None

def rotation_matrix_x(theta):
    return np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])

def rotation_matrix_y(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])

def rotation_matrix_z(theta): 
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0], 
                    [0, 0, 1]])

class SurfaceDistanceAnalyzer:
    def __init__(self, pdb_file: str, dist_constraints: List[Tuple[int, int, float]]):
        """
        Initialize the analyzer with PDB file and distance constraints.
        
        Args:
            pdb_file: Path to PDB file
            dist_constraints: List of tuples (res1, res2, distance)
        """
        self.pdb_file = pdb_file
        self.dist_constraints = dist_constraints
        self.structure = None
        self.surface_residues = set()
        self.ca_coords = {}
        
        # Load structure
        parser = Bio.PDB.PDBParser(QUIET=True)
        self.structure = parser.get_structure('protein', pdb_file)
        
        # Get CA coordinates
        self._get_ca_coords()
        
        # Calculate surface residues
        self._calculate_surface_residues()
    
    def _get_ca_coords(self):
        """Extract CA coordinates for all residues."""
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        res_id = (chain.id, residue.id[1])
                        self.ca_coords[res_id] = residue['CA'].get_coord()

    def _calculate_surface_residues(self):
        """Calculate surface residues using neighbor count method."""
        NEIGHBOR_CUTOFF = 10.0  # Keep at 10.0 Angstroms
        SURFACE_NEIGHBOR_THRESHOLD = 18 
        
        neighbor_counts = {}  # Store neighbor counts for debugging
        
        for res1_id, coord1 in self.ca_coords.items():
            # Count neighbors within cutoff
            neighbor_count = 0
            for res2_id, coord2 in self.ca_coords.items():
                if res1_id != res2_id:
                    dist = np.linalg.norm(coord1 - coord2)
                    if dist < NEIGHBOR_CUTOFF:
                        neighbor_count += 1
            
            neighbor_counts[res1_id] = neighbor_count
            # If residue has few neighbors, it's likely on the surface
            if neighbor_count < SURFACE_NEIGHBOR_THRESHOLD:
                
                self.surface_residues.add(res1_id)
        
        # Add detailed debug information
        print(f"Total residues: {len(self.ca_coords)}")
        print(f"Surface residues found: {len(self.surface_residues)}")
        print("\nNeighbor counts for constraint residues:")
        
        # Print neighbor counts for constraint residues with error handling
        # for res1, res2, dist in self.dist_constraints:
        #     # Check if residues exist in structure
        #     if res1 not in neighbor_counts:
        #         print(f"Warning: Residue {res1} not found in structure")
        #     else:
        #         print(f"Residue {res1} has {neighbor_counts[res1]} neighbors")
                
        #     if res2 not in neighbor_counts:
        #         print(f"Warning: Residue {res2} not found in structure")
        #     else:
        #         print(f"Residue {res2} has {neighbor_counts[res2]} neighbors")
        #     print("-" * 40)
        
        # Print all available residue IDs for debugging
        # print("\nAvailable residue IDs in structure:")
        # print(sorted(list(self.ca_coords.keys()))[:10], "...")
    
    def filter_surface_constraints(self) -> List[Tuple[int, int, float]]:
        """Filter constraints to only include those between surface residues."""
        surface_constraints = []
        for res1, res2, dist in self.dist_constraints:
            # Add debug information
            # if res1 not in self.surface_residues:
            #     print(f"Residue {res1} not on surface")
            # if res2 not in self.surface_residues:
            #     print(f"Residue {res2} not on surface")
            
            if (res1 in self.surface_residues and res2 in self.surface_residues):
                surface_constraints.append((res1, res2, dist))
        
        # Add debug information
        print(f"Total constraints: {len(self.dist_constraints)}")
        print(f"Surface constraints found: {len(surface_constraints)}")
        
        # Sort by distance
        return sorted(surface_constraints, key=lambda x: x[2])

    # def check_clash(self, domain1_residues: List[int], domain2_residues: List[int]) -> bool:
    #     """
    #     Check if two domains clash using CA-CA distance threshold.
        
    #     Args:
    #         domain1_residues: List of residue IDs in domain 1
    #         domain2_residues: List of residue IDs in domain 2
            
    #     Returns:
    #         True if there is a clash, False otherwise
    #     """
    #     CLASH_THRESHOLD = 8.0  # Angstroms
        
    #     for res1 in domain1_residues:
    #         coord1 = self.ca_coords[res1]
    #         for res2 in domain2_residues:
    #             coord2 = self.ca_coords[res2]
    #             dist = np.linalg.norm(coord1 - coord2)
    #             if dist < CLASH_THRESHOLD:
    #                 return True
    #     return False

    def find_rotation_transform(self, constraint: Tuple[Tuple[str, int], Tuple[str, int], float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find rotation and translation to satisfy the distance constraint.
        
        Args:
            constraint: (res1, res2, target_distance) where res1 and res2 are (chain, resid) tuples
        
        Returns:
            translation_vector: 3D translation vector
            pivot_point: Rotation center point (coordinates of fixed chain residue)
        """
        fixed_res, moving_res, target_dist = constraint
        
        # Get current coordinates
        pivot_point = self.ca_coords[fixed_res]  # Fixed point (B chain)
        moving_point = self.ca_coords[moving_res]  # Point to move (C chain)
        
        # Calculate current distance
        current_dist = np.linalg.norm(moving_point - pivot_point)
        
        # First translate moving point to match the target distance
        direction = moving_point - pivot_point
        direction = direction / np.linalg.norm(direction)  # Normalize
        new_moving_point = pivot_point + direction * target_dist
        
        # Translation vector to move point2 to new_point2
        translation = new_moving_point - moving_point
        
        print(f"Current distance: {current_dist:.2f}Å")
        print(f"Target distance: {target_dist:.2f}Å")
        print(f"Translation vector: {translation}")
        print(f"Pivot point (from chain B): {pivot_point}")
        
        return translation, pivot_point

    def get_neighbors(self, angles, angle_step, angle_results):
        """Get all possible neighbors by changing one angle by one step"""
        x, y, z = angles
        neighbors = []
        # Try changing x
        if x + angle_step < 2*np.pi:
            neighbors.append((x + angle_step, y, z))
        if x - angle_step >= 0:
            neighbors.append((x - angle_step, y, z))
        # Try changing y
        if y + angle_step < 2*np.pi:
            neighbors.append((x, y + angle_step, z))
        if y - angle_step >= 0:
            neighbors.append((x, y - angle_step, z))
        # Try changing z
        if z + angle_step < 2*np.pi:
            neighbors.append((x, y, z + angle_step))
        if z - angle_step >= 0:
            neighbors.append((x, y, z - angle_step))
        return [n for n in neighbors if n in angle_results]

    def find_valid_orientation(self, translation: np.ndarray,
                            pivot_point: np.ndarray,
                            fixed_domain_chains: List[str],
                            moving_domain_chains: List[str],
                            search_step_degree: int = 60,
                            candidates_per_iteration: int = 5,
                            output_extra_rotation_pdbs: bool = False) -> Tuple[Optional[List[np.ndarray]], Optional[Dict]]:
        """
        Find valid orientation by systematically sampling rotation angles
        Returns rotation matrix instead of euler angles
        
        Args:
            translation: Translation vector
            pivot_point: Pivot point for rotation
            fixed_domain_chains: List of chain IDs in the fixed domain
            moving_domain_chains: List of chain IDs in the moving domain
            search_step_degree: Search step in degrees
            candidates_per_iteration: Number of candidate structures to keep in each iteration
        """
        # Get residues and their coordinates for each domain
        moving_residues = [(res_id, self.ca_coords[res_id]) 
                          for res_id in self.ca_coords.keys() 
                          if res_id[0] in moving_domain_chains]
        fixed_residues = [(res_id, self.ca_coords[res_id]) 
                          for res_id in self.ca_coords.keys() 
                          if res_id[0] in fixed_domain_chains]
        
        CLASH_THRESHOLD = 12.0
        
        print(f"\nDebug Information:")
        print(f"Number of moving domain residues: {len(moving_residues)}")
        print(f"Number of fixed domain residues: {len(fixed_residues)}")
        print(f"Translation vector: {translation}")
        print(f"Pivot point: {pivot_point}")
        
        # Test identity rotation (no rotation) first
        print("\nTesting no rotation case (identity matrix):")
        identity_matrix = np.eye(3)
        identity_clash_count = 0
        identity_clash_pairs = []
        
        # Check clashes between different chains using original coordinates
        all_residues = fixed_residues + moving_residues
        for res1_id, res1_coord in all_residues:
            for res2_id, res2_coord in all_residues:
                if res1_id[0] != res2_id[0]:  # Different chains
                    dist = np.linalg.norm(res1_coord - res2_coord)
                    if dist < CLASH_THRESHOLD:
                        identity_clash_count += 1
                        identity_clash_pairs.append((res1_id, res2_id, dist))
                
        print(f"Number of clashes with no rotation: {identity_clash_count}")
        print("Sample of clash pairs (first 5):")
        for i, (res1, res2, dist) in enumerate(identity_clash_pairs[:5]):
            print(f"  Clash {i+1}: Residue {res1} - Residue {res2}, Distance: {dist:.2f}Å")
        
        # Try different MAX_CLASHES values
        max_clashes_attempts = [
            identity_clash_count+50,
            min(len(moving_residues)*len(fixed_residues) * 0.01, identity_clash_count + len(fixed_residues)),
            max(len(moving_residues)*len(fixed_residues) * 0.01, identity_clash_count + len(fixed_residues)),   # Most relaxed
        ]
        
        angle_step = search_step_degree * np.pi / 180
        angles_x = np.arange(0, 2*np.pi, angle_step)
        angles_y = np.arange(0, 2*np.pi, angle_step)
        angles_z = np.arange(0, 2*np.pi, angle_step)
        
        n_samples = len(angles_x) * len(angles_y) * len(angles_z)
        for MAX_CLASHES in max_clashes_attempts:
            print(f"\nSearching through {n_samples} orientations with MAX_CLASHES: {MAX_CLASHES}...")
            print(f"Angle step: {search_step_degree} degrees")
            print(f"Try with MAX_CLASHES: {MAX_CLASHES}")
            print(f"Clash threshold: {CLASH_THRESHOLD} Å")
            
            # Store valid rotations with their angles
            valid_rotations = []  # (R, clash_count, (x,y,z))
            
            # Store clash counts for all rotations to analyze stability
            angle_results = {}  # (x,y,z) -> clash_count
            
            min_clashes = float('inf')
            min_clash_angles = None
            
            # Pre-translate moving residue coordinates for rotation search
            translated_coords = {res_id: coord + translation 
                               for res_id, coord in moving_residues}
            
            for x_angle in tqdm(angles_x, desc="X angle"):
                for y_angle in angles_y:
                    for z_angle in angles_z:
                        # Generate rotation matrix from angles
                        Rx = rotation_matrix_x(x_angle)
                        Ry = rotation_matrix_y(y_angle)
                        Rz = rotation_matrix_z(z_angle)
                        R = Rz @ Ry @ Rx
                        
                        # Count clashes
                        clash_count = 0
                        clash_pairs = []
                        early_termination = False
                        
                        for moving_res_id, _ in moving_residues:
                            centered_coord = translated_coords[moving_res_id] - pivot_point
                            rotated_coord = np.dot(R, centered_coord) + pivot_point
                            
                            # Check clashes with all other residues
                            for res_id, res_coord in all_residues:
                                if moving_res_id[0] != res_id[0]:  # Different chains
                                    dist = np.linalg.norm(rotated_coord - res_coord)
                                    if dist < CLASH_THRESHOLD:
                                        clash_count += 1
                                        clash_pairs.append((moving_res_id, res_id, dist))
                                        if clash_count > MAX_CLASHES:
                                            early_termination = True
                                            break
                            
                            if early_termination:
                                break
                        
                        # Store result for all rotations
                        angles = (x_angle, y_angle, z_angle)
                        angle_results[angles] = clash_count
                        
                        if clash_count < min_clashes and (x_angle,y_angle,z_angle) != (0,0,0):
                            min_clashes = clash_count
                            min_clash_angles = angles
                        
                        if clash_count <= MAX_CLASHES and (x_angle,y_angle,z_angle) != (0,0,0):
                            valid_rotations.append((R, clash_count, angles))
            
            if not valid_rotations:
                print("\nNo valid orientations found!")
                continue
            else:
                print(f"\nFound {len(valid_rotations)} valid orientations with MAX_CLASHES: {MAX_CLASHES}")
                break

        def find_cluster(start_angles):
            """Find all connected valid rotations starting from given angles"""
            cluster = set()
            to_visit = {start_angles}
            visited = set()
            
            while to_visit:
                current = to_visit.pop()
                if current in visited:
                    continue
                    
                visited.add(current)
                if angle_results[current] <= MAX_CLASHES:
                    cluster.add(current)
                    # Add valid neighbors to visit
                    neighbors = self.get_neighbors(current, angle_step, angle_results)
                    for neighbor in neighbors:
                        if neighbor not in visited and angle_results[neighbor] <= MAX_CLASHES:
                            to_visit.add(neighbor)
            return cluster

        # Find all clusters of valid rotations
        valid_angles = {angles for angles, clashes in angle_results.items() 
                       if clashes <= MAX_CLASHES}
        clusters = []
        unclustered = set(valid_angles)
        
        while unclustered:
            start = unclustered.pop()
            cluster = find_cluster(start)
            if cluster:
                clusters.append(cluster)
                unclustered -= cluster
        
        print(f"\nFound {len(clusters)} clusters of valid rotations:")
        for i, cluster in enumerate(clusters):
            print(f"Cluster {i+1}: {len(cluster)} rotations")
        
        # Get top largest clusters
        if clusters:
            top_clusters = sorted(clusters, key=len, reverse=True)[:2*candidates_per_iteration]
            refined_clusters = []
            
            print(f"\nRefining top {len(top_clusters)} clusters...")
            for i, cluster in enumerate(top_clusters):
                print(f"Refining cluster {i+1} ({len(cluster)} rotations)...")
                refined = self.refine_cluster(cluster, angle_step, angle_results, translated_coords, fixed_residues, pivot_point, 15, MAX_CLASHES, CLASH_THRESHOLD)
                refined_clusters.append(refined)
                print(f"After refinement: {len(refined)} rotations")
            
            # Find center rotations for each refined cluster
            center_rotations = []
            for cluster in refined_clusters:
                # Calculate mean angles
                mean_x = sum(x for x, _, _ in cluster) / len(cluster)
                mean_y = sum(y for _, y, _ in cluster) / len(cluster)
                mean_z = sum(z for _, _, z in cluster) / len(cluster)
                
                # Find the rotation closest to the mean
                center_rotation = min(cluster, 
                                   key=lambda a: sum((a[i] - mean)**2 for i, mean 
                                                   in enumerate([mean_x, mean_y, mean_z])))
                
                x, y, z = center_rotation
                Rx = rotation_matrix_x(x)
                Ry = rotation_matrix_y(y)
                Rz = rotation_matrix_z(z)
                R = Rz @ Ry @ Rx
                center_rotations.append((R, angle_results[center_rotation], center_rotation))
            
            # Filter similar rotations
            def are_rotations_similar(R1, R2, angle_threshold=15):
                """Check if two rotations are similar by comparing their angles"""
                # Convert rotation matrices to Euler angles
                def matrix_to_euler(R):
                    # Extract Euler angles from rotation matrix
                    # Using the method from: https://www.geometrictools.com/Documentation/EulerAngles.pdf
                    if abs(R[2,0]) < 1:
                        x = np.arctan2(-R[2,1], R[2,2])
                        y = np.arcsin(R[2,0])
                        z = np.arctan2(-R[1,0], R[0,0])
                    else:
                        x = np.arctan2(R[1,2], R[1,1])
                        y = np.arcsin(R[2,0])
                        z = 0
                    return np.array([x, y, z])
                
                angles1 = matrix_to_euler(R1)
                angles2 = matrix_to_euler(R2)
                
                # Convert to degrees and normalize to [0, 360]
                angles1 = np.degrees(angles1) % 360
                angles2 = np.degrees(angles2) % 360
                
                # Calculate angular differences
                diff = np.abs(angles1 - angles2)
                diff = np.minimum(diff, 360 - diff)  # Handle wrap-around
                
                return np.all(diff < angle_threshold)
            
            # Filter similar rotations
            filtered_rotations = []
            for R, clashes, angles in center_rotations:
                # Check if this rotation is similar to any already accepted rotation
                is_similar = False
                for existing_R, _, _ in filtered_rotations:
                    if are_rotations_similar(R, existing_R):
                        is_similar = True
                        break
                
                if not is_similar:
                    filtered_rotations.append((R, clashes, angles))
                    if len(filtered_rotations) >= candidates_per_iteration:
                        break
            
            print(f"\nAfter filtering similar rotations: {len(filtered_rotations)} unique rotations")
            
            print("\nSelected rotations:")
            for i, (R, clashes, angles) in enumerate(filtered_rotations):
                print(f"\nRotation {i+1}:")
                print(f"Angles (degrees): ({angles[0]*180/np.pi:.1f}, "
                      f"{angles[1]*180/np.pi:.1f}, {angles[2]*180/np.pi:.1f})")
                print(f"Clashes: {clashes}")
            
            pocket_Rs = [R for R, _, _ in filtered_rotations]
            extra_dict = None
            if output_extra_rotation_pdbs:
                # Fixed 8 rotations (degrees): (180,0,0), (0,180,0), (0,0,180), (180,180,0), (180,0,180), (0,180,180), (0,0,0), (180,180,180)
                fixed_angles_deg = [(180, 0, 0), (0, 180, 0), (0, 0, 180), (180, 180, 0), (180, 0, 180), (0, 180, 180), (0, 0, 0), (180, 180, 180)]
                fixed_list = []
                for (dx, dy, dz) in fixed_angles_deg:
                    x = dx * np.pi / 180
                    y = dy * np.pi / 180
                    z = dz * np.pi / 180
                    Rx = rotation_matrix_x(x)
                    Ry = rotation_matrix_y(y)
                    Rz = rotation_matrix_z(z)
                    R = Rz @ Ry @ Rx
                    fixed_list.append((f"{dx},{dy},{dz}", R))
                # Top 3 clusters, each with 3 rotation (angle_tuple, R) for PDB output
                top3_refined = refined_clusters[:3]
                cluster_list = []
                for c_idx, cluster in enumerate(top3_refined):
                    center_angles = center_rotations[c_idx][2]  # (x,y,z) in radians
                    sorted_by_clash = sorted(cluster, key=lambda a: angle_results[a])
                    chosen = [center_angles]
                    for a in sorted_by_clash:
                        if a not in chosen and len(chosen) < 3:
                            chosen.append(a)
                    cluster_rotations = []
                    for r_idx, angles in enumerate(chosen):
                        x, y, z = angles
                        Rx = rotation_matrix_x(x)
                        Ry = rotation_matrix_y(y)
                        Rz = rotation_matrix_z(z)
                        R = Rz @ Ry @ Rx
                        label = f"{angles[0]*180/np.pi:.0f},{angles[1]*180/np.pi:.0f},{angles[2]*180/np.pi:.0f}"
                        cluster_rotations.append((label, R))
                    cluster_list.append(cluster_rotations)
                extra_dict = {"fixed": fixed_list, "clusters": cluster_list}
            return (pocket_Rs, extra_dict)
        
        return (None, None)

    def refine_cluster(self, cluster, angle_step,angle_results, translated_coords, fixed_residues, pivot_point, angle_step_fine=5, MAX_CLASHES=100, CLASH_THRESHOLD=12.0):
        """Refine cluster by expanding boundaries and filling gaps"""
        angle_step_fine_rad = angle_step_fine * np.pi / 180
        expanded = True
        refined_cluster = set(cluster)
        
        while expanded:
            expanded = False
            boundary = set()
            
            # Find boundary points
            for angles in refined_cluster:
                for neighbor in self.get_neighbors(angles, angle_step_fine_rad, angle_results):
                    if neighbor not in refined_cluster:
                        boundary.add(neighbor)
            
            # Try to expand boundary
            for angles in tqdm(boundary, desc="Expanding boundary", leave=False):
                if angles not in angle_results:
                    # Calculate clashes for new angle
                    x, y, z = angles
                    Rx = rotation_matrix_x(x)
                    Ry = rotation_matrix_y(y)
                    Rz = rotation_matrix_z(z)
                    R = Rz @ Ry @ Rx
                    
                    clash_count = self.count_clashes(R, translated_coords, fixed_residues, 
                                                     pivot_point, CLASH_THRESHOLD, MAX_CLASHES)
                    angle_results[angles] = clash_count
                
                if angle_results[angles] <= MAX_CLASHES:
                    refined_cluster.add(angles)
                    expanded = True
        
        # Fill gaps with fine step size
        filled_cluster = set(refined_cluster)
        for x, y, z in refined_cluster:
            # Generate intermediate points
            for dx in np.arange(-angle_step/2, angle_step/2, angle_step_fine_rad):
                for dy in np.arange(-angle_step/2, angle_step/2, angle_step_fine_rad):
                    for dz in np.arange(-angle_step/2, angle_step/2, angle_step_fine_rad):
                        new_angles = (x + dx, y + dy, z + dz)
                        if new_angles not in angle_results:
                            # Calculate clashes for new angles
                            Rx = rotation_matrix_x(new_angles[0])
                            Ry = rotation_matrix_y(new_angles[1])
                            Rz = rotation_matrix_z(new_angles[2])
                            R = Rz @ Ry @ Rx
                            clash_count = self.count_clashes(R, translated_coords, fixed_residues,
                                                     pivot_point, CLASH_THRESHOLD, MAX_CLASHES)
                            angle_results[new_angles] = clash_count
                            
                        if angle_results[new_angles] <= MAX_CLASHES:
                            filled_cluster.add(new_angles)
        
        return filled_cluster

    def count_clashes(self, R, translated_coords, fixed_residues, pivot_point, CLASH_THRESHOLD, MAX_CLASHES):
        """Helper function to count clashes for a given rotation matrix"""
        clash_count = 0
        for moving_res_id, _ in translated_coords.items():
            centered_coord = translated_coords[moving_res_id] - pivot_point
            rotated_coord = np.dot(R, centered_coord) + pivot_point
            
            for fixed_res_id, fixed_coord in fixed_residues:
                dist = np.linalg.norm(rotated_coord - fixed_coord)
                if dist < CLASH_THRESHOLD:
                    clash_count += 1
                    if clash_count > MAX_CLASHES:
                        return clash_count
        return clash_count

    def calculate_ground_truth_transform(self, ground_truth_pdb: str,
                                   translation: np.ndarray,
                                   pred_pivot_point: np.ndarray,
                                   moving_chain: str = 'C',
                                   gt_chain: str = 'B') -> np.ndarray:
        """
        Calculate ground truth rotation matrix using sequence-based residue matching.
        The ground truth structure is superimposed onto the prediction structure before applying translation.
        """
        parser = Bio.PDB.PDBParser(QUIET=True)
        ground_truth = parser.get_structure('ground_truth', ground_truth_pdb)
        
        # Find pivot residue in predicted structure
        min_dist = float('inf')
        pivot_residue = None
        pred_chain_B = self.structure[0]['B']
        for residue in pred_chain_B:
            if 'CA' in residue:
                dist = np.linalg.norm(residue['CA'].get_coord() - pred_pivot_point)
                if dist < min_dist:
                    min_dist = dist
                    pivot_residue = residue

        if pivot_residue is None:
            raise RuntimeError("No pivot residue found in predicted structure")

        # Find matching pivot residue in ground truth structure
        gt_chain_B = ground_truth[0]['B']
        gt_pivot_residue = find_matching_residue(pivot_residue, gt_chain_B, pred_chain_B)

        if gt_pivot_residue is None or 'CA' not in gt_pivot_residue:
            raise RuntimeError("Could not find matching pivot residue in ground truth structure")

        gt_pivot_point = gt_pivot_residue['CA'].get_coord()

        print(f"\nPivot point matching:")
        print(f"Predicted structure pivot residue: Chain B, {pivot_residue.id[1]}, {pivot_residue.resname}")
        print(f"Ground truth structure pivot residue: Chain B, {gt_pivot_residue.id[1]}, {gt_pivot_residue.resname}")
        print(f"Distance between pivot points: {np.linalg.norm(pred_pivot_point - gt_pivot_point):.2f}Å")

        # Get matching residue pairs for alignment
        pred_chain_C = self.structure[0][moving_chain]
        gt_chain_match = ground_truth[0][gt_chain]

        matching_pairs = []
        for pred_residue in pred_chain_C:
            if 'CA' not in pred_residue:
                continue
                
            gt_residue = find_matching_residue(pred_residue, gt_chain_match, pred_chain_C)
            if gt_residue and 'CA' in gt_residue:
                matching_pairs.append((pred_residue, gt_residue))

        print(f"Found {len(matching_pairs)} matching residue pairs for superposition")

        if not matching_pairs:
            raise RuntimeError("No matching residue pairs found for superposition")

        # Use Superimposer for alignment
        sup = Superimposer()
        sup.set_atoms([pair[0]['CA'] for pair in matching_pairs],
                     [pair[1]['CA'] for pair in matching_pairs])
        sup.apply([pair[1]['CA'] for pair in matching_pairs])  # Apply rotation and translation to ground truth CA atoms

        R = sup.rotran[0]  # Rotation matrix
        t = sup.rotran[1]  # Translation vector

        # If Q_transformed needs to be calculated manually, it can be defined as:
        # Q_transformed = np.array([pair[1]['CA'].get_coord() for pair in matching_pairs])

        print(f"Calculated rotation matrix:\n{R}")
        print(f"Calculated translation vector:\n{t}")

        return R

    def apply_transformation(self, output_pdb: str, translation: np.ndarray,
                            pivot_point: np.ndarray, 
                            rotation_matrix: np.ndarray,
                            moving_chains: List[str] = ['C']):
        """
        Apply transformation using rotation matrix.
        The pivot_point should be from the same structure being transformed.
        """
        print(f"\nApplying transformation:")
        print(f"Translation vector: {translation}")
        print(f"Pivot point: {pivot_point}")
        print(f"Rotation matrix:\n{rotation_matrix}")
        print(f"Moving chains: {moving_chains}")

        # Get the constraint residues before transformation
        constraint_res1 = None
        constraint_res2 = None
        target_dist = None
        for constraint in self.dist_constraints:
            (chain1, idx1), (chain2, idx2), dist = constraint
            if (chain1 in moving_chains or chain2 in moving_chains):
                # Find the residues in the structure
                for chain in self.structure[0]:
                    if chain.id == chain1:
                        try:
                            constraint_res1 = chain[(' ', idx1, ' ')]
                        except:
                            pass
                    if chain.id == chain2:
                        try:
                            constraint_res2 = chain[(' ', idx2, ' ')]
                        except:
                            pass
                if constraint_res1 and constraint_res2:
                    target_dist = dist
                    break

        if constraint_res1 and constraint_res2:
            # Calculate distance before transformation
            res1_coord = constraint_res1['CA'].get_coord()
            res2_coord = constraint_res2['CA'].get_coord()
            original_dist = np.linalg.norm(res1_coord - res2_coord)
            print(f"\nDistance between constrained residues before transformation:")
            print(f"Residue 1: Chain {constraint_res1.parent.id}, {constraint_res1.resname}{constraint_res1.id[1]}")
            print(f"Residue 2: Chain {constraint_res2.parent.id}, {constraint_res2.resname}{constraint_res2.id[1]}")
            print(f"Distance: {original_dist:.2f}Å")
            print(f"Target distance: {target_dist:.2f}Å")

        new_structure = Bio.PDB.Structure.Structure('transformed')
        new_model = Bio.PDB.Model.Model(0)
        new_structure.add(new_model)
        
        # Store original and transformed coordinates for logging
        coord_changes = []
        
        for chain in self.structure[0]:
            new_chain = Bio.PDB.Chain.Chain(chain.id)
            new_model.add(new_chain)
            
            for residue in chain:
                new_residue = residue.copy()
                new_chain.add(new_residue)
                
                if chain.id in moving_chains:
                    for atom in new_residue:
                        original_coord = atom.get_coord()
                        # Apply translation first
                        translated_coord = original_coord + translation
                        # Center around pivot point and rotate
                        centered_coord = translated_coord - pivot_point
                        rotated_coord = np.dot(rotation_matrix, centered_coord) + pivot_point
                        # Set final coordinate
                        final_coord = rotated_coord
                        atom.set_coord(final_coord)
                        
                        # Log coordinate changes for first few atoms
                        if len(coord_changes) < 5:
                            coord_changes.append({
                                'residue': f"{residue.id[1]}{residue.resname}",
                                'atom': atom.name,
                                'original': original_coord,
                                'translated': translated_coord,
                                'centered': centered_coord,
                                'rotated': rotated_coord,
                                'final': final_coord
                            })
        
        # # Print coordinate changes for first few atoms
        # print("\nCoordinate changes for first few atoms:")
        # for change in coord_changes:
        #     print(f"\nResidue {change['residue']}, Atom {change['atom']}:")
        #     print(f"Original: {change['original']}")
        #     print(f"After translation: {change['translated']}")
        #     print(f"After centering: {change['centered']}")
        #     print(f"After rotation: {change['rotated']}")
        #     print(f"Final: {change['final']}")
        
        io = Bio.PDB.PDBIO()
        io.set_structure(new_structure)
        io.save(output_pdb)
        print(f"\nSaved transformed structure to: {output_pdb}")

        # Calculate distance after transformation
        if constraint_res1 and constraint_res2:
            # Find the residues in the new structure
            new_res1 = None
            new_res2 = None
            for chain in new_structure[0]:
                if chain.id == constraint_res1.parent.id:
                    try:
                        new_res1 = chain[(' ', constraint_res1.id[1], ' ')]
                    except:
                        pass
                if chain.id == constraint_res2.parent.id:
                    try:
                        new_res2 = chain[(' ', constraint_res2.id[1], ' ')]
                    except:
                        pass
            
            if new_res1 and new_res2:
                new_res1_coord = new_res1['CA'].get_coord()
                new_res2_coord = new_res2['CA'].get_coord()
                new_dist = np.linalg.norm(new_res1_coord - new_res2_coord)
                print(f"\nDistance between constrained residues after transformation:")
                print(f"Distance: {new_dist:.2f}Å")
                print(f"Target distance: {target_dist:.2f}Å")
                print(f"Difference from target: {abs(new_dist - target_dist):.2f}Å")

    def verify_ground_truth_transform(self, ground_truth_pdb: str,
                                    translation: np.ndarray,
                                    pivot_point: np.ndarray,
                                    theta: float,
                                    phi: float,
                                    moving_chains: List[str] = ['C'],
                                    gt_chain: str = 'B'):
        """
        Verify the calculated transformation by trying different rotation combinations.
        """
        # Load ground truth structure
        parser = Bio.PDB.PDBParser(QUIET=True)
        ground_truth = parser.get_structure('ground_truth', ground_truth_pdb)
        
        # Get ground truth coordinates
        gt_coords = {}
        for chain in ground_truth[0]:
            if chain.id == gt_chain:
                for residue in chain:
                    if 'CA' in residue:
                        gt_coords[residue.id[1]] = residue['CA'].get_coord()
        
        # Create temporary PDB file for transformed structure
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp:
            tmp_pdb = tmp.name
        
        # Apply transformation using existing function
        self.apply_transformation(tmp_pdb, translation, pivot_point, (theta, phi), moving_chains)
        
        # Load transformed structure
        transformed = parser.get_structure('transformed', tmp_pdb)
        transformed_coords = {}
        for chain in transformed[0]:
            if chain.id in moving_chains:
                for residue in chain:
                    if 'CA' in residue:
                        transformed_coords[residue.id[1]] = residue['CA'].get_coord()
        
        # Calculate RMSD using calculate_rmsd
        gt_rmsd = calculate_rmsd(ground_truth_pdb, tmp_pdb)
        
        print(f"\nGround truth transformation RMSD: {gt_rmsd:.2f}Å")
        print(f"Number of common residues used for RMSD: {len(transformed_coords)}")
        
        # Clean up temporary file
        import os
        os.unlink(tmp_pdb)

def parse_dist_constraints(dist_constraints_file: str, pdb_file: str) -> list:
    """
    Parse distance constraints from file and convert to required format.
    
    Args:
        dist_constraints_file: Path to constraints file
        pdb_file: Path to PDB file to get chain lengths
        
    Returns:
        List of tuples in format: (('chain1', index1), ('chain2', local_index2), distance)
        where local_index2 is the residue index relative to the start of chain2
        
    Example input file format:
        396,942,9.78  # where 396 is in chain A, 942 is in chain C
        328,1014,5.48
        ...
    """
    # Get chain lengths from PDB file
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    # Calculate chain lengths and cumulative lengths
    chain_lengths = {}
    cumulative_lengths = {}
    total_length = 0
    
    # First pass: calculate individual chain lengths
    for chain in structure[0]:
        chain_lengths[chain.id] = len([res for res in chain if 'CA' in res])
    
    # Second pass: calculate cumulative lengths
    for chain in structure[0]:
        cumulative_lengths[chain.id] = total_length
        total_length += chain_lengths[chain.id]
    
    print("Chain information:")
    for chain in structure[0]:
        print(f"Chain {chain.id}: length={chain_lengths[chain.id]}, cumulative_length={cumulative_lengths[chain.id]}")
    
    constraints = []
    
    with open(dist_constraints_file, 'r') as f:
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Parse line
            try:
                idx1, idx2, dist = line.strip().split(',')
                # Convert to proper types
                idx1 = int(idx1)
                idx2 = int(idx2)
                dist = float(dist)
                
                # Determine which chains these residues belong to
                chain1 = None
                chain2 = None
                
                # Constraint indices are 1-based (residues are numbered from 1
                # consecutively over all chains), so a chain
                # owns (cumulative, cumulative + length] -- NOT the half-open
                # [cumulative, ...), which mapped each chain's boundary residue to
                # the next chain at local index 0 and raised KeyError: (chain, 0).

                # Find chain1 (first residue)
                for chain in structure[0]:
                    if cumulative_lengths[chain.id] < idx1 <= cumulative_lengths[chain.id] + chain_lengths[chain.id]:
                        chain1 = chain.id
                        break
                
                # Find chain2 (second residue)
                for chain in structure[0]:
                    if cumulative_lengths[chain.id] < idx2 <= cumulative_lengths[chain.id] + chain_lengths[chain.id]:
                        chain2 = chain.id
                        break
                
                if chain1 is None or chain2 is None:
                    print(f"Warning: Could not determine chains for residues {idx1} and {idx2}")
                    continue
                
                # Calculate local indices
                local_idx1 = idx1 - cumulative_lengths[chain1]
                local_idx2 = idx2 - cumulative_lengths[chain2]
                
                # Create constraint tuple
                constraint = ((chain1, local_idx1), (chain2, local_idx2), dist)
                constraints.append(constraint)
                
            except ValueError as e:
                print(f"Warning: Skipping invalid line: {line.strip()}")
                print(f"Error: {e}")
                continue
    
    print(f"Loaded {len(constraints)} distance constraints")
    
    # Sort by distance (optional)
    constraints.sort(key=lambda x: x[2])
    
    return constraints

def rename_chains(input_pdb: str, output_pdb: str, start_chain: str = 'A') -> None:
    """
    Rename chains in PDB file sequentially starting from a specified chain name.
    
    Args:
        input_pdb: Path to input PDB file
        output_pdb: Path to output PDB file
        start_chain: Starting chain name (default: 'A')
        
    Example:
        If start_chain='A', chains will be renamed to A, B, C, ...
        If start_chain='X', chains will be renamed to X, Y, Z, ...
    """
    # Initialize PDB parser
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', input_pdb)
    
    # Get starting ASCII value for chain naming
    start_ascii = ord(start_chain)
    
    # Count number of chains
    num_chains = len([chain for chain in structure[0]])
    if num_chains > (90 - start_ascii + 1):  # 90 is ASCII for 'Z'
        raise ValueError(f"Too many chains ({num_chains}) to rename starting from '{start_chain}'")
    
    # Create new structure for renamed chains
    new_structure = Bio.PDB.Structure.Structure('renamed')
    new_model = Bio.PDB.Model.Model(0)
    new_structure.add(new_model)
    
    # Rename chains
    for i, chain in enumerate(structure[0]):
        new_chain = Bio.PDB.Chain.Chain(chr(start_ascii + i))
        new_model.add(new_chain)
        
        # Copy all residues to new chain
        for residue in chain:
            new_residue = residue.copy()
            new_chain.add(new_residue)
    
    # Save renamed structure
    io = Bio.PDB.PDBIO()
    io.set_structure(new_structure)
    io.save(output_pdb)
    
    print(f"Renamed {num_chains} chains starting from '{start_chain}'")
    print(f"Original chains renamed to: {', '.join(chr(start_ascii + i) for i in range(num_chains))}")

def process_multi_chain_iterative(analyzer, dist_constraints, fixed_chains, moving_chains, search_step_degree, use_multiple_constraints, output_dir, target, ground_truth_pdb=None, candidates_per_iteration=5, used_chain_pairs=None, output_extra_rotation_pdbs=False):
    """
    Process multi-chain structure iteratively by treating two groups of chains at a time.
    Returns:
        tuple: (results, updated_used_chain_pairs) where results is the list of valid transformations
    """
    if used_chain_pairs is None:
        used_chain_pairs = set()
        
    print(f"\nProcessing chains: {fixed_chains} -> {moving_chains}")
    
    # Get the last fixed chain and first moving chain
    last_fixed_chain = fixed_chains[-1]
    first_moving_chain = moving_chains[0]
    
    # First try: Filter constraints between last fixed chain and first moving chain
    relevant_constraints = []
    for constraint in dist_constraints:
        (chain1, idx1), (chain2, idx2), dist = constraint
        chain_pair = tuple(sorted([chain1, chain2]))
        
        # Only include constraints between last fixed chain and first moving chain
        if (chain1 == last_fixed_chain and chain2 == first_moving_chain) or \
           (chain1 == first_moving_chain and chain2 == last_fixed_chain):
            # Skip constraints that have been used before
            if chain_pair in used_chain_pairs:
                continue
            relevant_constraints.append(constraint)
    
    # If no constraints found between specified chains, try fallback
    if not relevant_constraints:
        print(f"No unused constraints found between chains {last_fixed_chain} and {first_moving_chain}")
        print("Trying fallback: finding shortest unused constraint between fixed and moving domains")
        
        # Fallback: Find shortest unused constraint between any fixed and moving chain
        fallback_constraints = []
        for constraint in dist_constraints:
            (chain1, idx1), (chain2, idx2), dist = constraint
            chain_pair = tuple(sorted([chain1, chain2]))
            
            # Check if constraint is between fixed and moving domains
            if ((chain1 in fixed_chains and chain2 in moving_chains) or \
                (chain1 in moving_chains and chain2 in fixed_chains)) and \
               chain_pair not in used_chain_pairs:
                fallback_constraints.append(constraint)
        
        if fallback_constraints:
            # Sort by distance and take the shortest
            fallback_constraints.sort(key=lambda x: x[2])
            relevant_constraints = [fallback_constraints[0]]
            print(f"Found fallback constraint: {relevant_constraints[0]}")
        else:
            print("No unused constraints found between fixed and moving domains")
            return None, used_chain_pairs
    
    # Sort constraints by distance
    relevant_constraints.sort(key=lambda x: x[2])
    
    # Select constraints to use
    if use_multiple_constraints:
        constraints_to_use = relevant_constraints[:3]  # Use top 3 smallest constraints
    else:
        constraints_to_use = [relevant_constraints[0]]  # Use only the smallest constraint
    
    # Update used_chain_pairs with the selected constraints
    for constraint in constraints_to_use:
        (chain1, idx1), (chain2, idx2), dist = constraint
        chain_pair = tuple(sorted([chain1, chain2]))
        used_chain_pairs.add(chain_pair)
        print(f"Using constraint: {constraint}")
    
    all_pocket_Rs = []
    for i, constraint in enumerate(constraints_to_use):
        print(f"\nProcessing constraint {i+1} of {len(constraints_to_use)}\n")
        # Find translation and pivot point
        translation, pivot_point = analyzer.find_rotation_transform(constraint)
        
        # Find valid orientations
        pocket_Rs, extra_rotations = analyzer.find_valid_orientation(
            translation,
            pivot_point,
            fixed_chains,
            moving_chains,
            search_step_degree=search_step_degree,
            candidates_per_iteration=candidates_per_iteration,
            output_extra_rotation_pdbs=output_extra_rotation_pdbs
        )
        
        if pocket_Rs:
            all_pocket_Rs.append((pocket_Rs, translation, pivot_point, extra_rotations))
    
    return all_pocket_Rs, used_chain_pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--pdb_file", type=str, required=True)
    parser.add_argument("--dist_constraints_file", type=str, required=True)
    parser.add_argument("--search_step_degree", type=int, default=30)
    parser.add_argument("--ground_truth_pdb", type=str)
    parser.add_argument("--output_dir", type=str, default='./output')
    parser.add_argument("--use_multiple_constraints", action="store_true", help="Use top 3 smallest constraints instead of just the smallest one")
    parser.add_argument("--candidates_per_iteration", type=int, default=3, help="Number of candidate structures to keep in each iteration")
    parser.add_argument("--no_translation", action="store_true", help="Do not translate the structure")
    parser.add_argument("--output_rotation_pdbs", action="store_true", help="When True, also output PDBs for 8 fixed rotations (180,0,0 etc.) and top 3 clusters with 3 rotations each")
    args = parser.parse_args()
    
    print(f"\nProcessing target: {args.target}")
    
    pdb_file = args.pdb_file
    import tempfile
    import shutil

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Intermediate PDBs must NOT land in the default $TMPDIR: on the compute nodes
    # that resolved to the shared /dev/shm tmpfs, where files written by one job
    # were observed to vanish while it was still running. Keeping them in a
    # per-run directory next to the output also makes the final shutil.move a
    # same-filesystem rename.
    tmp_root = os.environ.get("START_STRUCTURES_TMPDIR") or args.output_dir
    os.makedirs(tmp_root, exist_ok=True)
    job_tmpdir = tempfile.mkdtemp(prefix=f".tmp_{args.target}_", dir=tmp_root)

    rename_pdb_file = tempfile.mktemp(dir=job_tmpdir)
    rename_chains(pdb_file, rename_pdb_file, start_chain='A')
    
    # Parse distance constraints
    dist_constraints = parse_dist_constraints(args.dist_constraints_file, rename_pdb_file)
    
    # Get all chains in the structure
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', rename_pdb_file)
    all_chains = [chain.id for chain in structure[0]]
    
    # Initialize with first chain as fixed
    fixed_chains = [all_chains[0]]
    moving_chains = all_chains[1:]
    
    # Store all candidate structures for each iteration
    current_candidates = [(rename_pdb_file, fixed_chains, moving_chains)]
    
    # Keep track of temporary files to clean up
    temp_files = set([rename_pdb_file])
    
    # Store final results
    final_results = []
    
    # Track used chain pairs and their constraints
    used_chain_pairs = set()
    
    try:
        iteration = 0
        # Iterative process
        while current_candidates and current_candidates[0][2]:  # While we have candidates and they have moving chains
            iteration += 1
            print(f"\nIteration {iteration}")
            
            next_candidates = []
            
            for candidate_idx, (current_pdb, fixed_chains, moving_chains) in enumerate(current_candidates):
                # Initialize analyzer with current structure
                analyzer = SurfaceDistanceAnalyzer(current_pdb, dist_constraints)
                
                # Process current iteration
                results, used_chain_pairs = process_multi_chain_iterative(
                    analyzer,
                    dist_constraints,
                    fixed_chains,
                    moving_chains,
                    args.search_step_degree,
                    args.use_multiple_constraints,
                    args.output_dir,
                    args.target,
                    args.ground_truth_pdb,
                    args.candidates_per_iteration,
                    used_chain_pairs,
                    output_extra_rotation_pdbs=args.output_rotation_pdbs
                )
                
                if results:
                    # Apply transformations for each result
                    for result_idx, result in enumerate(results):
                        pocket_Rs, translation, pivot_point = result[0], result[1], result[2]
                        extra_rotations = result[3] if len(result) > 3 else None
                        for i, R in enumerate(pocket_Rs):
                            # Create temporary file for transformed structure
                            transformed_pdb = tempfile.mktemp(dir=job_tmpdir)
                            temp_files.add(transformed_pdb)
                            
                            analyzer.apply_transformation(
                                transformed_pdb,
                                translation if not args.no_translation else np.zeros(3),
                                pivot_point,
                                R,
                                moving_chains=moving_chains
                            )
                            
                            # Calculate RMSD for ground truth transformation
                            if args.ground_truth_pdb:
                                gt_rmsd = calculate_rmsd(args.ground_truth_pdb, transformed_pdb)
                                print(f"RMSD: {gt_rmsd:.2f}Å")
                            
                            # Update fixed and moving chains
                            new_fixed_chains = fixed_chains + [moving_chains[0]]
                            new_moving_chains = moving_chains[1:]
                            
                            # If this is the last chain, save to final results
                            if not new_moving_chains:
                                final_results.append((transformed_pdb, gt_rmsd if args.ground_truth_pdb else None))
                            else:
                                # Add to next iteration's candidates
                                next_candidates.append((transformed_pdb, new_fixed_chains, new_moving_chains))
                        # Output extra rotation PDBs (8 fixed + top3 clusters x 3) when flag is set
                        if extra_rotations is not None:
                            trans = translation if not args.no_translation else np.zeros(3)
                            for label, R in extra_rotations["fixed"]:
                                safe_label = label.replace(",", "_")
                                out_path = os.path.join(args.output_dir, f"{args.target}_iter{iteration}_cand{candidate_idx}_constraint{result_idx}_fixed_{safe_label}.pdb")
                                analyzer.apply_transformation(out_path, trans, pivot_point, R, moving_chains=moving_chains)
                            for c_idx, cluster_rots in enumerate(extra_rotations["clusters"]):
                                for r_idx, (label, R) in enumerate(cluster_rots):
                                    safe_label = label.replace(",", "_")
                                    out_path = os.path.join(args.output_dir, f"{args.target}_iter{iteration}_cand{candidate_idx}_constraint{result_idx}_cluster{c_idx}_rot{r_idx}_{safe_label}.pdb")
                                    analyzer.apply_transformation(out_path, trans, pivot_point, R, moving_chains=moving_chains)
                else:
                    # If no valid results found, use current structure for next iteration
                    print(f"No constraints available between {fixed_chains[-1]} and {moving_chains[0]}, continuing with current structure")
                    new_fixed_chains = fixed_chains + [moving_chains[0]]
                    new_moving_chains = moving_chains[1:]
                    
                    if not new_moving_chains:
                        final_results.append((current_pdb, None))
                    else:
                        next_candidates.append((current_pdb, new_fixed_chains, new_moving_chains))
            
            # Update current candidates for next iteration
            # Only keep the specified number of candidates
            if len(next_candidates) > args.candidates_per_iteration:
                # Remove excess candidates and their temporary files
                for candidate_pdb, _, _ in next_candidates[args.candidates_per_iteration:]:
                    if candidate_pdb in temp_files:
                        try:
                            os.unlink(candidate_pdb)
                            temp_files.remove(candidate_pdb)
                        except:
                            pass
                next_candidates = next_candidates[:args.candidates_per_iteration]
            current_candidates = next_candidates
        
        # Save final results
        print(f"\nSaving {len(final_results)} final structures")
        for i, (final_pdb, gt_rmsd) in enumerate(final_results):
            output_pdb = f"{args.output_dir}/{args.target}_start_{i}.pdb"
            shutil.move(final_pdb, output_pdb)
            temp_files.remove(final_pdb)
            if gt_rmsd is not None:
                print(f"Structure {i}: RMSD = {gt_rmsd:.2f}Å")
            
    finally:
        # Clean up all temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        shutil.rmtree(job_tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()