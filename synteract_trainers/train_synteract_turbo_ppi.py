#! /usr/bin/env python3
import entrypoint_setup

import gc
import os
import torch
from tqdm.auto import tqdm

from training.arguments import parse_args
from training.base_synteract_turbo_trainer import BaseSynteractTurboTrainer
from data_processing.download_ppi_data import download_clustered_ppi_data
from utils.training import set_seed


class SynteractTurboTrainer(BaseSynteractTurboTrainer):
    """
    Complete trainer for Synteract4/5 with full STRING data loading.
    
    Supports two embedding modes:
    - ON_THE_FLY (default): Uses frozen PLM to embed sequences during training.
      Allows learning to weight PLM hidden layers (linear/conv processing).
    - PRECOMPUTED (--preembed): Pre-compute embeddings before training.
      Only works with encoder_layer_processing='last_hidden_state'.
    """
    
    def __init__(self, args):
        super().__init__(args)    
        # === LOAD DATA FROM HUGGINGFACE ===
        self._print(f"Loading data from HuggingFace: {args.ppi_data_repo}")
        self.MIN_ORG_SIZE_TRAIN = 1000
        self.MIN_ORG_SIZE_EVAL = 100
        
        if self.is_main_process:
            (
                train_df,
                valid_df,
                test_df,
                self.interaction_set,
                self.seq_dict,
            ) = download_clustered_ppi_data(
                data_type=args.ppi_data_type,
                cluster_percentage=args.cluster_percentage,
                descriptor=args.ppi_data_descriptor,
                hf_repo=args.ppi_data_repo,
            )

            ### Synteract models only need the positive interaction, filter out labels == 0
            print(f"Number of total train: {len(train_df)}")
            print(f"Number of total valid interactions: {len(valid_df)}")
            print(f"Number of total test interactions: {len(test_df)}")
            train_df = train_df[train_df['labels'] == 1]
            valid_df = valid_df[valid_df['labels'] == 1]
            test_df = test_df[test_df['labels'] == 1]
            print(f"Number of train positive interactions after filtering: {len(train_df)}")
            print(f"Number of valid positive interactions after filtering: {len(valid_df)}")
            print(f"Number of test positive interactions after filtering: {len(test_df)}")

            if args.bugfix:
                train_df = train_df.sample(frac=0.01).reset_index(drop=True)
                valid_df = valid_df.sample(frac=0.01).reset_index(drop=True)
                test_df = test_df.sample(frac=0.01).reset_index(drop=True)
                self.MIN_ORG_SIZE_TRAIN = 10
                self.MIN_ORG_SIZE_EVAL = 10
            
            self.train_dataframes, self.valid_dataframes, self.test_dataframes = {}, {}, {}
            unique_orgs = sorted(list(set(train_df['OrgA'].unique()).union(set(train_df['OrgB'].unique()))))
            train_same_org = train_df[train_df['OrgA'] == train_df['OrgB']]
            valid_same_org = valid_df[valid_df['OrgA'] == valid_df['OrgB']]
            test_same_org = test_df[test_df['OrgA'] == test_df['OrgB']]
            train_grouped = train_same_org.groupby('OrgA', sort=False)
            valid_grouped = valid_same_org.groupby('OrgA', sort=False)
            test_grouped = test_same_org.groupby('OrgA', sort=False)
            for org in tqdm(unique_orgs, desc="Sampling dataframes", total=len(unique_orgs)):
                if org not in train_grouped.groups:
                    train_len = 0
                else:
                    train_df_org = train_grouped.get_group(org)
                    train_len = len(train_df_org)
                if train_len < self.MIN_ORG_SIZE_TRAIN:
                    self._print(f"Skipping org {org} because it has {train_len} examples, which is less than {self.MIN_ORG_SIZE_TRAIN} interactions")
                    continue
                else:
                    self.train_dataframes[org] = train_df_org.sample(frac=1.0).reset_index(drop=True)
                if org not in valid_grouped.groups:
                    valid_len = 0
                else:
                    valid_df_org = valid_grouped.get_group(org)
                    valid_len = len(valid_df_org)
                if valid_len < self.MIN_ORG_SIZE_EVAL:
                    self._print(f"Skipping org {org} because it has {valid_len} examples, which is less than {self.MIN_ORG_SIZE_EVAL} interactions")
                    continue
                else:
                    self.valid_dataframes[org] = valid_df_org.sample(frac=1.0).reset_index(drop=True)
                if org not in test_grouped.groups:
                    test_len = 0
                else:
                    test_df_org = test_grouped.get_group(org)
                    test_len = len(test_df_org)
                if test_len < self.MIN_ORG_SIZE_EVAL:
                    self._print(f"Skipping org {org} because it has {test_len} examples, which is less than {self.MIN_ORG_SIZE_EVAL} interactions")
                    continue
                else:
                    self.test_dataframes[org] = test_df_org.sample(frac=1.0).reset_index(drop=True)
            print(f"Number of train dataframes: {len(self.train_dataframes)}")
            print(f"Number of valid dataframes: {len(self.valid_dataframes)}")
            print(f"Number of test dataframes: {len(self.test_dataframes)}")
            
            # Free original DataFrames - data is now split per-organism
            del train_df, valid_df, test_df
            gc.collect()

            # Cache for other workers
            if self.is_distributed:
                cache_path = os.path.join(self.save_dir, 'data_cache.pt')
                
                # Check if cache already exists (e.g. from resumed run) to avoid race/redundant work
                if not os.path.exists(cache_path):
                    self._print(f"Saving data cache to {cache_path} for distributed workers...")
                    # Atomic write: save to tmp then rename
                    tmp_path = cache_path + '.tmp'
                    torch.save((
                        self.train_dataframes,
                        self.valid_dataframes,
                        self.test_dataframes,
                        self.interaction_set,
                        self.seq_dict,
                    ), tmp_path)
                    os.replace(tmp_path, cache_path)
                else:
                    self._print(f"Data cache already exists at {cache_path}, skipping save.")

        self._synchronize()
        
        if self.is_distributed and not self.is_main_process:
            # Load from cache
            cache_path = os.path.join(self.save_dir, 'data_cache.pt')
            (
                self.train_dataframes,
                self.valid_dataframes,
                self.test_dataframes,
                self.interaction_set,
                self.seq_dict,
            ) = torch.load(cache_path, map_location="cpu", weights_only=False)
        
        train_orgs = set(self.train_dataframes.keys())
        valid_orgs = set(self.valid_dataframes.keys())
        test_orgs = set(self.test_dataframes.keys())
        assert len(train_orgs) > 0, 'No training organisms available after filtering'
        assert len(valid_orgs) > 0, 'No validation organisms available after filtering'
        assert len(test_orgs) > 0, 'No test organisms available after filtering'
        
        self.is_single_organism = (len(train_orgs) == 1 and len(valid_orgs) == 1 and len(test_orgs) == 1)
        if self.is_single_organism:
            assert train_orgs == valid_orgs == test_orgs, 'Single-organism detection requires identical orgs across splits'
            self.single_organism_id = next(iter(train_orgs))
            self._print(f"Detected single-organism dataset: {self.single_organism_id}")
        else:
            self.single_organism_id = None
            self._print(
                f"Detected multi-organism dataset: train={len(train_orgs)}, valid={len(valid_orgs)}, test={len(test_orgs)}"
            )
        
        # Violation detection settings
        self.skip_violations = args.skip_violations
        
        # Print data stats
        total_train = sum(len(df) for df in self.train_dataframes.values())
        total_valid = sum(len(df) for df in self.valid_dataframes.values())
        total_test = sum(len(df) for df in self.test_dataframes.values())
        
        self._print(f"Train: {total_train}, Valid: {total_valid}, Test: {total_test}")
        self._print(f"Loaded {len(self.seq_dict)} unique sequences")
        self._print(f"Interaction set size: {len(self.interaction_set)}")


if __name__ == "__main__":
    args = parse_args()
    set_seed(42)
    
    # Create trainer and train
    trainer = SynteractTurboTrainer(args)
    trainer.prep_for_training(memory_mode=args.memory_mode)
    model = trainer.train()
    
    if os.environ['WANDB_AVAILABLE'] == 'true' and trainer.is_main_process:
        import wandb
        wandb.finish()
