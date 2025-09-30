import pandas as pd
import numpy as np
import torch
import random
import subprocess
import os
import psutil
import gzip
import pathlib
from collections import defaultdict
from tqdm.auto import tqdm

base_path = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/lhallee-cpu/code/'
if os.path.exists(base_path):
    cache_root = f"{base_path}/hf_cache"
    tmp_root   = f"{base_path}/tmp"
    pathlib.Path(cache_root).mkdir(parents=True, exist_ok=True)
    pathlib.Path(tmp_root).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"]            = cache_root
    os.environ["HF_DATASETS_CACHE"]  = f"{cache_root}/datasets"
    os.environ["TRANSFORMERS_CACHE"] = f"{cache_root}/transformers" # this is deprecated, but does not hurt anything
    os.environ["HF_HUB_CACHE"]       = f"{cache_root}/hub"
    print(f"HF_HOME: {os.environ['HF_HOME']}")
    print(f"HF_DATASETS_CACHE: {os.environ['HF_DATASETS_CACHE']}")
    print(f"TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")
    print(f"HF_HUB_CACHE: {os.environ['HF_HUB_CACHE']}")


from datasets import load_dataset


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def extract_species_id(protein_id):
    """Extract species ID from protein ID by splitting on period"""
    return protein_id.split('.')[0]


def split_with_sim(
        df: pd.DataFrame,
        seq_dict: dict,
        similarity_threshold: float = 0.5,
        min_rows: int = 1000, # number of test clusters
        n: int = 3, # word size, 5 is faster but 3 is more sensitive
        memory_percentage: float = 0.5,
        save: bool = False,
        minimum_confidence_train: int = 150,
        minimum_confidence_eval: int = 150,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_seed(42)
    
    # Ensure processed_datasets directory exists
    os.makedirs("processed_datasets", exist_ok=True)
    
    train_file = f"processed_datasets/split_with_sim_{similarity_threshold}_{minimum_confidence_train}_train.csv"
    test_file = f"processed_datasets/split_with_sim_{similarity_threshold}_{minimum_confidence_eval}_{min_rows}_test.csv"
    val_file = f"processed_datasets/split_with_sim_{similarity_threshold}_{minimum_confidence_eval}_{min_rows}_val.csv"
    if os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(val_file):
        print(f'Loading cached train, test, and validation sets from {train_file}, {test_file}, and {val_file}')
        return pd.read_csv(train_file), pd.read_csv(test_file), pd.read_csv(val_file)

    # Write all_seqs to a FASTA file
    base_path = 'string_data'
    fasta_path = f"{base_path}.fasta"
    with open(fasta_path, "w") as f:
        for id, seq in seq_dict.items():
            f.write(f">{id}\n{seq}\n")

    # Run cd-hit in Docker
    output_path = f"{base_path}/output_{similarity_threshold}"

    if os.path.exists(output_path):
        print(f'Output file {output_path} already exists')
    else:
        # Build the cd-hit Docker image if not already built
        num_cpu = os.cpu_count() - 4 if os.cpu_count() > 4 else 1
        memory_max = int(memory_percentage * psutil.virtual_memory().total / 1024 / 1024)  # in MB
        print(f'Using {num_cpu} CPUs and {memory_max} MB memory')

        print("Building cd-hit Docker image...")
        docker_image = "cd-hit"
        dockerfile_url = "https://raw.githubusercontent.com/weizhongli/cdhit/master/Docker/Dockerfile"
        # Build the Docker image
        try:
            subprocess.run([
                "docker", "build", "--tag", docker_image, dockerfile_url
            ], check=True)

            print(f'Clustering {len(seq_dict)} sequences')
            subprocess.run([
                "docker", "run",
                "-v", f"{os.getcwd()}:/data",
                "-w", "/data",
                docker_image,
                "cd-hit",
                "-i", fasta_path,
                "-o", output_path,
                "-d", "0",
                "-c", str(similarity_threshold),
                "-n", str(n),
                "-T", str(num_cpu),
                "-M", str(memory_max)
            ], check=True)
        except:
            subprocess.run([
                "sudo", "docker", "build", "--tag", docker_image, dockerfile_url
            ], check=True)

            print(f'Clustering {len(seq_dict)} sequences')
            subprocess.run([
                "sudo", "docker", "run",
                "-v", f"{os.getcwd()}:/data",
                "-w", "/data",
                docker_image,
                "cd-hit",
                "-i", fasta_path,
                "-o", output_path,
                "-d", "0",
                "-c", str(similarity_threshold),
                "-n", str(n),
                "-T", str(num_cpu),
                "-M", str(memory_max)
            ], check=True)    

    cluster_file = f"{output_path}.clstr"

    # Read the output clusters file
    cluster_dict = defaultdict(list)
    with open(cluster_file, "r") as f:
        for line in tqdm(f, desc="Reading cluster file"):
            if line.startswith(">"):
                cluster_id = line.split('Cluster')[1].split("\n")[0].strip()
            else:
                seq_id = line.split('>')[1].split('...')[0].strip()
                cluster_dict[cluster_id].append(seq_id)

    print(f'Number of unique sequences: {len(seq_dict)}, Number of clusters: {len(cluster_dict)}, Number of rows: {len(df)}')

    # Build a mapping from sequence ID to cluster ID
    seq_id_to_cluster = {}
    for cluster_id, seq_ids in cluster_dict.items():
        for seq_id in seq_ids:
            seq_id_to_cluster[seq_id] = cluster_id

    print(f'Cluster ids: {list(cluster_dict.keys())[:10]}')

    # map each row to their clusters (cluster ID, not list)
    df['cluster_a'] = df['IdA'].map(seq_id_to_cluster)
    df['cluster_b'] = df['IdB'].map(seq_id_to_cluster)

    print(f'Cluster a: {df["cluster_a"].head()}')
    print(f'Cluster b: {df["cluster_b"].head()}')

    # calculate the impact of each cluster on the dataset (vectorized)
    a_counts = df['cluster_a'].value_counts()
    b_counts = df['cluster_b'].value_counts()
    all_clusters = set(a_counts.index).union(b_counts.index)
    cluster_impact = {cluster: a_counts.get(cluster, 0) + b_counts.get(cluster, 0) for cluster in all_clusters}

    # sort the clusters by impact, not actually used anymore
    sorted_clusters = sorted(cluster_impact.items(), key=lambda x: x[1], reverse=True)

    test_clusters = set()
    valid_clusters = set()
    test_mask = np.zeros(len(df), dtype=bool)
    valid_mask = np.zeros(len(df), dtype=bool)
    
    # Randomly shuffle the bottom half clusters
    np.random.shuffle(sorted_clusters)
    
    # Randomly select test clusters from bottom half
    pbar = tqdm(desc="Selecting test clusters", total=len(sorted_clusters))
    for cluster, _ in sorted_clusters:
        test_clusters.add(cluster)
        # Update mask: both cluster_a and cluster_b must be in test_clusters
        new_mask = df["cluster_a"].isin(test_clusters) & df["cluster_b"].isin(test_clusters)
        new_mask = new_mask & (df['labels'] >= minimum_confidence_eval) # for eval sets, we ensure high confidence
        if new_mask.sum() >= min_rows:
            test_mask = new_mask
            break
        pbar.update(1)
        pbar.set_postfix(test_size=new_mask.sum())
    pbar.close()

    test_df = df[test_mask]
    if len(test_df) == 0:
        raise ValueError("No test rows found")
    else:
        print(f'Test set size: {len(test_df)}')

    print(f'Used {len(test_clusters)} test clusters')

    # Continue selecting validation clusters (same target size as test)
    target_valid_size = len(test_df)
    remaining_clusters = [item for item in sorted_clusters if item[0] not in test_clusters]
    
    pbar = tqdm(desc="Selecting validation clusters", total=len(remaining_clusters))
    for cluster, _ in remaining_clusters:
        valid_clusters.add(cluster)
        # Update mask: both cluster_a and cluster_b must be in val_clusters
        new_mask = df["cluster_a"].isin(valid_clusters) & df["cluster_b"].isin(valid_clusters)
        new_mask = new_mask & (df['labels'] >= minimum_confidence_eval) # for eval sets, we ensure high confidence
        if new_mask.sum() >= target_valid_size:
            valid_mask = new_mask
            break
        pbar.update(1)
        pbar.set_postfix(valid_size=new_mask.sum(), target=target_valid_size)
    pbar.close()

    valid_df = df[valid_mask]
    if len(valid_df) == 0:
        raise ValueError("No validation rows found")
    else:
        print(f'Validation set size: {len(valid_df)}')

    print(f'Used {len(valid_clusters)} validation clusters')

    before_len = len(df)
    print(f'Train before trimming: {before_len}')

    # remove rows that have any protein in test or validation clusters
    all_excluded_clusters = test_clusters.union(valid_clusters)
    train_df = df[~df["cluster_a"].isin(all_excluded_clusters) & ~df["cluster_b"].isin(all_excluded_clusters)]
    after_len_cluster_trim = len(train_df)
    train_df = train_df[train_df['labels'] >= minimum_confidence_train]
    after_len_confidence_trim = len(train_df)
    print(f'Train after trimming: {after_len_cluster_trim}')
    print(f'Train after confidence trimming: {after_len_confidence_trim}')
    print(f'Test set size: {len(test_df)}')
    print(f'Validation set size: {len(valid_df)}')
    trimmed_len = before_len - after_len_confidence_trim
    print(f'Trimmed {trimmed_len} rows')

    if trimmed_len == 0:
        raise ValueError("No rows were trimmed")
    elif trimmed_len == len(test_df) + len(valid_df):
        print("Warning: ONLY test and validation rows were removed from the training set. We usually expect more training rows than test+val rows to be removed.")

    # Verify disjoint sets
    # For proteins
    test_proteins = set(test_df['IdA']).union(set(test_df['IdB']))
    valid_proteins = set(valid_df['IdA']).union(set(valid_df['IdB']))
    train_proteins = set(train_df['IdA']).union(set(train_df['IdB']))

    print(f'Test proteins: {len(test_proteins)}, Valid proteins: {len(valid_proteins)}, Train proteins: {len(train_proteins)}')

    # Check for overlaps (proteins)
    test_valid_overlap = test_proteins.intersection(valid_proteins)
    test_train_overlap = test_proteins.intersection(train_proteins)
    valid_train_overlap = valid_proteins.intersection(train_proteins)

    if test_valid_overlap:
        print(f'WARNING: {len(test_valid_overlap)} proteins overlap between test and validation')
    if test_train_overlap:
        print(f'WARNING: {len(test_train_overlap)} proteins overlap between test and train')
    if valid_train_overlap:
        print(f'WARNING: {len(valid_train_overlap)} proteins overlap between validation and train')

    if not test_valid_overlap and not test_train_overlap and not valid_train_overlap:
        print('SUCCESS: All three protein sets are completely disjoint!')

    # For clusters
    test_clusters_set = set(test_df['cluster_a']).union(set(test_df['cluster_b']))
    valid_clusters_set = set(valid_df['cluster_a']).union(set(valid_df['cluster_b']))
    train_clusters_set = set(train_df['cluster_a']).union(set(train_df['cluster_b']))

    print(f'Test clusters: {len(test_clusters_set)}, Valid clusters: {len(valid_clusters_set)}, Train clusters: {len(train_clusters_set)}')

    # Check for overlaps (clusters)
    test_valid_cluster_overlap = test_clusters_set.intersection(valid_clusters_set)
    test_train_cluster_overlap = test_clusters_set.intersection(train_clusters_set)
    valid_train_cluster_overlap = valid_clusters_set.intersection(train_clusters_set)

    if test_valid_cluster_overlap:
        print(f'WARNING: {len(test_valid_cluster_overlap)} clusters overlap between test and validation')
    if test_train_cluster_overlap:
        print(f'WARNING: {len(test_train_cluster_overlap)} clusters overlap between test and train')
    if valid_train_cluster_overlap:
        print(f'WARNING: {len(valid_train_cluster_overlap)} clusters overlap between validation and train')

    if not test_valid_cluster_overlap and not test_train_cluster_overlap and not valid_train_cluster_overlap:
        print('SUCCESS: All three cluster sets are completely disjoint!')

    # shuffle all three sets
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)
    valid_df = valid_df.sample(frac=1).reset_index(drop=True)

    if save:
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        valid_df.to_csv(val_file, index=False)
    return train_df, valid_df, test_df


def get_single_species_data(
        link_file: str,
        similarity_threshold: float = 0.5,
        min_rows: int = 1000,
        n: int = 3,
        save: bool = False,
        minimum_confidence_train: int = 150,
        minimum_confidence_eval: int = 150,
):
    with gzip.open(link_file, 'rt') as f:
        link_df = pd.read_csv(f) # (ida, idb, label)
    link_df = link_df.rename(columns={'protein1': 'IdA', 'protein2': 'IdB', 'combined_score': 'labels'})

    # build set of all interactions
    interaction_set = set('_'.join(sorted([ida, idb])) for ida, idb in zip(link_df['IdA'], link_df['IdB']))
    print(f'Interaction set size: {len(interaction_set)}')

    seq_dict = load_dataset('Synthyra/StringDBSeqsv12', split='train')
    seq_dict = dict(zip(seq_dict['id'], seq_dict['sequence']))

    train_df, valid_df, test_df = split_with_sim(
        df=link_df,
        seq_dict=seq_dict,
        similarity_threshold=similarity_threshold,
        min_rows=min_rows,
        n=n,
        save=save,
        minimum_confidence_train=minimum_confidence_train,
        minimum_confidence_eval=minimum_confidence_eval,
    )

    return train_df, valid_df, test_df, seq_dict, interaction_set


if __name__ == "__main__":
    # py -m data.ppi_data_splitting
    link_file = 'protein.links.v12.0.min900.onlyAB.csv.gz'
    similarity_threshold = 0.4
    min_rows = 10000
    n = 2
    minimum_confidence_train = 150
    minimum_confidence_eval = 150
    print('Testing single species data...')
    train_df, valid_df, test_df, seq_dict, interaction_set = get_single_species_data(
        link_file,
        similarity_threshold,
        min_rows,
        n,
        True,
        minimum_confidence_train,
        minimum_confidence_eval,
    )
