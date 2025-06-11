#!/usr/bin/env python3
"""
Simple CIF to LogMD Demo Script

This script provides a basic demonstration of visualizing CIF files using LogMD.
It includes basic CIF parsing and structure alignment functionality.
"""

import os
import glob
import numpy as np
from natsort import natsorted
from logmd import LogMD


def simple_parse_cif(cif_file):
    """
    Simple CIF parser that extracts atom coordinates and basic info.
    Returns a dictionary with atom information.
    """
    atoms = []
    
    with open(cif_file, 'r') as f:
        lines = f.readlines()
    
    # Find the start of atom records
    in_atom_section = False
    for line in lines:
        line = line.strip()
        
        if line.startswith('_atom_site.'):
            in_atom_section = True
            continue
            
        if in_atom_section and line.startswith('ATOM'):
            # Split on whitespace but handle multiple spaces properly
            parts = line.split()
            if len(parts) >= 18:  # Ensure we have enough columns based on the actual format
                try:
                    atom_id = int(parts[1])
                    atom_name = parts[3]  # label_atom_id
                    residue_name = parts[5]  # label_comp_id
                    chain_id = parts[9]  # label_asym_id  
                    residue_seq = int(parts[6])  # label_seq_id
                    x = float(parts[10])  # Cartn_x
                    y = float(parts[11])  # Cartn_y  
                    z = float(parts[12])  # Cartn_z
                    b_factor = float(parts[17])  # B_iso_or_equiv
                    
                    atoms.append({
                        'id': atom_id,
                        'name': atom_name,
                        'residue': residue_name,
                        'chain': chain_id,
                        'res_seq': residue_seq,
                        'coords': np.array([x, y, z]),
                        'b_factor': b_factor
                    })
                except (ValueError, IndexError) as e:
                    print(f"    -> Parsing error for line: {line[:50]}... Error: {e}")
                    continue
    
    return atoms


def atoms_to_pdb_string(atoms):
    """
    Convert atom list to PDB format string.
    """
    pdb_lines = []
    
    for atom in atoms:
        # Format PDB ATOM record
        line = f"ATOM  {atom['id']:5d} {atom['name']:>4s} {atom['residue']:>3s} {atom['chain']:>1s}{atom['res_seq']:4d}    " \
               f"{atom['coords'][0]:8.3f}{atom['coords'][1]:8.3f}{atom['coords'][2]:8.3f}  1.00{atom['b_factor']:6.2f}          "
        pdb_lines.append(line)
    
    return '\n'.join(pdb_lines)


def np_kabsch(a, b):
    """Kabsch algorithm for optimal rotation matrix"""
    # Calculate covariance matrix
    ab = a.T @ b
    
    # Singular value decomposition
    u, s, vh = np.linalg.svd(ab)
    
    # Handle reflection case
    flip = np.linalg.det(u @ vh) < 0
    if flip:
        u[:, -1] = -u[:, -1]
    
    return u @ vh


def align_structures(atoms1, atoms2):
    """
    Align atoms1 to atoms2 using Kabsch algorithm.
    Returns aligned coordinates for atoms1.
    """
    # Extract coordinates
    coords1 = np.array([atom['coords'] for atom in atoms1])
    coords2 = np.array([atom['coords'] for atom in atoms2])
    
    # Use minimum number of atoms for alignment
    min_atoms = min(len(coords1), len(coords2))
    coords1_align = coords1[:min_atoms]
    coords2_align = coords2[:min_atoms]
    
    # Center coordinates
    center1 = coords1_align.mean(axis=0)
    center2 = coords2_align.mean(axis=0)
    
    coords1_centered = coords1_align - center1
    coords2_centered = coords2_align - center2
    
    # Get rotation matrix
    R = np_kabsch(coords1_centered, coords2_centered)
    
    # Apply transformation to all atoms
    coords1_all_centered = coords1 - center1
    coords1_aligned = coords1_all_centered @ R.T + center2
    
    # Update atom coordinates
    aligned_atoms = []
    for i, atom in enumerate(atoms1):
        new_atom = atom.copy()
        new_atom['coords'] = coords1_aligned[i]
        aligned_atoms.append(new_atom)
    
    return aligned_atoms


def run_cif_demo(cif_files, max_structures=None):
    """
    Main demo function to visualize CIF files with LogMD
    """
    # Initialize LogMD
    logmd = LogMD()
    logmd.notebook()
    print(f"LogMD visualization URL: {logmd.url}")
    
    # Optionally limit number of structures
    if max_structures is not None:
        cif_files = cif_files[:max_structures]
    print(f"\nProcessing {len(cif_files)} CIF files...")
    
    all_atoms = []
    
    # Parse all CIF files
    for i, cif_file in enumerate(cif_files):
        print(f"  [{i+1}/{len(cif_files)}] Loading {os.path.basename(cif_file)}...")
        
        try:
            atoms = simple_parse_cif(cif_file)
            if atoms:
                all_atoms.append((atoms, cif_file))
                print(f"    -> Found {len(atoms)} atoms")
            else:
                print(f"    -> Warning: No atoms found")
        except Exception as e:
            print(f"    -> Error: {e}")
    
    if not all_atoms:
        print("No structures loaded successfully!")
        return
    
    print(f"\nSuccessfully loaded {len(all_atoms)} structures")
    
    # Use last structure as reference for alignment
    reference_atoms = all_atoms[-1][0]
    print(f"Using {os.path.basename(all_atoms[-1][1])} as reference structure")
    
    # Process and visualize each structure
    print("\nAligning and visualizing structures...")
    for i, (atoms, filename) in enumerate(all_atoms):
        print(f"  [{i+1}/{len(all_atoms)}] Processing {os.path.basename(filename)}...")
        
        # Align to reference (except reference itself)
        if i < len(all_atoms) - 1:
            try:
                aligned_atoms = align_structures(atoms, reference_atoms)
            except Exception as e:
                print(f"    -> Alignment error: {e}, using original coordinates")
                aligned_atoms = atoms
        else:
            aligned_atoms = atoms
        
        # Convert to PDB and visualize
        try:
            pdb_string = atoms_to_pdb_string(aligned_atoms)
            logmd(pdb_string)
            print(f"    -> Successfully visualized")
        except Exception as e:
            print(f"    -> Visualization error: {e}")
    
    print(f"\nDemo complete! View the molecular dynamics at: {logmd.url}")
    print("Each frame represents a different CIF structure, aligned for comparison.")


def main():
    """Main entry point"""
    print("=== CIF to LogMD Demo ===")
    
    # Search for CIF files
    search_patterns = [
        #"results/generated_proteins/*/fold_*_model_*.cif",
        "logmd_utils/trajectory/*.cif"
    ]
    
    cif_files = []
    for pattern in search_patterns:
        found_files = glob.glob(pattern, recursive=True)
        cif_files.extend(found_files)
        print(f"Found {len(found_files)} files matching pattern: {pattern}")
    
    if not cif_files:
        print("\nNo CIF files found!")
        print("Please ensure you have CIF files in your workspace.")
        print("Expected locations:")
        for pattern in search_patterns:
            print(f"  - {pattern}")
        return
    
    # Remove duplicates and sort naturally
    cif_files = natsorted(list(set(cif_files)))
    
    print(f"\nTotal: {len(cif_files)} unique CIF files found")
    print("\nFirst few files:")
    for i, f in enumerate(cif_files[:5]):
        print(f"  {i+1}. {f}")
    if len(cif_files) > 5:
        print(f"  ... and {len(cif_files) - 5} more")
    
    # Get user preference
    print("\n" + "="*50)
    print("Options:")
    print("1. Visualize all files")
    print("2. Choose a specific protein family")
    print("3. Visualize all files (same as option 1)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "2":
        # Show available families
        families = set()
        for f in cif_files:
            if "generated_proteins" in f:
                parts = f.split(os.sep)
                for part in parts:
                    if "cond_best" in part or "uncond_best" in part:
                        families.add(part)
        
        if families:
            print("\nAvailable protein families:")
            for i, family in enumerate(sorted(families), 1):
                print(f"  {i}. {family}")
            
            family_choice = input("\nEnter family name: ").strip()
            family_files = [f for f in cif_files if family_choice in f]
            
            if family_files:
                print(f"\nFound {len(family_files)} files for family '{family_choice}'")
                run_cif_demo(family_files)
            else:
                print(f"No files found for family '{family_choice}'")
        else:
            print("No protein families detected, using all files")
            run_cif_demo(cif_files)
    
    elif choice == "3":
        print(f"\nVisualizing all {len(cif_files)} files...")
        run_cif_demo(cif_files)
    
    else:
        print(f"\nVisualizing all {len(cif_files)} files...")
        run_cif_demo(cif_files)


if __name__ == "__main__":
    main() 