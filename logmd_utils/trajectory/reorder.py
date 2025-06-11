import pandas as pd
import os
from glob import glob

def extract_sequence_from_cif(filepath):
    """Extract the B chain sequence from a CIF file."""
    with open(filepath, 'r') as f:
        found = False
        sequence_lines = []
        for line in f:
            if not found:
                if '2 polypeptide(L) no no B' in line:
                    found = True
            elif found:
                if line.strip() == ';':
                    break
                sequence_lines.append(line.strip().replace('\n', '').replace(';', ''))
        return ''.join(sequence_lines)

# Read trajectory.csv and get SeqB sequences
df = pd.read_csv('trajectory.csv')
seqs_b = df['SeqB'].tolist()
print(f"Found {len(seqs_b)} sequences in trajectory.csv")

# Get all .cif files
cif_files = glob('*.cif')
print(f"Found {len(cif_files)} .cif files")

# Extract sequences from all .cif files
cif_sequences = {}
for cif_file in cif_files:
    seq = extract_sequence_from_cif(cif_file)
    cif_sequences[cif_file] = seq
    print(f"Extracted sequence from {cif_file}: {seq[:50]}..." if len(seq) > 50 else f"Extracted sequence from {cif_file}: {seq}")

# Match .cif files to seqs_b order
file_mapping = {}
unmatched_files = []

for i, target_seq in enumerate(seqs_b):
    matched = False
    for cif_file, cif_seq in cif_sequences.items():
        if cif_seq == target_seq:
            file_mapping[cif_file] = f"{i+1}.cif"
            print(f"Match found: {cif_file} -> {i+1}.cif")
            matched = True
            break
    if not matched:
        print(f"Warning: No .cif file found for sequence {i+1}")

# Check for unmatched .cif files
for cif_file in cif_files:
    if cif_file not in file_mapping:
        unmatched_files.append(cif_file)

if unmatched_files:
    print(f"Warning: {len(unmatched_files)} .cif files could not be matched to seqs_b:")
    for file in unmatched_files:
        print(f"  - {file}")

# Rename files
print("\nRenaming files...")
for old_name, new_name in file_mapping.items():
    if os.path.exists(new_name):
        print(f"Warning: {new_name} already exists, skipping {old_name}")
        continue
    
    try:
        os.rename(old_name, new_name)
        print(f"Renamed: {old_name} -> {new_name}")
    except Exception as e:
        print(f"Error renaming {old_name} to {new_name}: {e}")

print(f"\nReordering complete! Renamed {len(file_mapping)} files.")






