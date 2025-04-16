import torch
import os
from typing import Optional, List, Callable
from transformers import PreTrainedTokenizerBase
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from .pooler import Pooler


class ProteinDataset(Dataset):
    """Simple dataset for protein sequences."""
    def __init__(self, sequences: list[str]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx]


def build_collator(tokenizer) -> Callable[[list[str]], tuple[torch.Tensor, torch.Tensor]]:
    def _collate_fn(sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate function for batching sequences."""
        return tokenizer(sequences, return_tensors="pt", padding='longest', pad_to_multiple_of=8)
    return _collate_fn


class EmbeddingMixin:
    def embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device

    def _read_sequences_from_db(self, db_path: str) -> set[str]:
        """Read sequences from SQLite database."""
        import sqlite3
        sequences = []
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT sequence FROM embeddings")
            while True:
                row = c.fetchone()
                if row is None:
                    break
                sequences.append(row[0])
        return set(sequences)

    def embed_dataset(
        self,
        sequences: List[str],
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 2,
        max_len: int = 512,
        full_embeddings: bool = False,
        embed_dtype: torch.dtype = torch.float32,
        pooling_types: List[str] = ['mean'],
        num_workers: int = 0,
        sql: bool = False,
        save: bool = True,
        sql_db_path: str = 'embeddings.db',
        save_path: str = 'embeddings.pth',
    ) -> Optional[dict[str, torch.Tensor]]:
        """Embed a dataset of protein sequences.
        
        Args:
            sequences: List of protein sequences
            batch_size: Batch size for processing
            max_len: Maximum sequence length
            full_embeddings: Whether to return full residue-wise (True) embeddings or pooled (False)
            pooling_type: Type of pooling ('mean' or 'cls')
            num_workers: Number of workers for data loading, 0 for the main process
            sql: Whether to store embeddings in SQLite database - will be stored in float32
            sql_db_path: Path to SQLite database
            
        Returns:
            Dictionary mapping sequences to embeddings, or None if sql=True

        Note:
            - If sql=True, embeddings can only be stored in float32
            - sql is ideal if you need to stream a very large dataset for training in real-time
            - save=True is ideal if you can store the entire embedding dictionary in RAM
            - sql will be used if it is True and save is True or False
            - If your sql database or .pth file is already present, they will be scanned first for already embedded sequences
            - Sequences will be truncated to max_len and sorted by length in descending order for faster processing

        Example:
            >>> embedder = EmbeddingMixin()
            >>> embedding_dict = embedder.embed_dataset(
                sequences=[
                    'MALWMRLLPLLALLALWGPDPAAA', ... # list of protein sequences
                ],
                tokenizer=tokenizer,
                batch_size=2, # adjust for your GPU memory
                max_len=512, # adjust for your needs
                full_embeddings=False, # if True, no pooling is performed
                embed_dtype=torch.float32, # cast to what dtype you want
                pooling_type=['mean', 'cls'], # more than one pooling type will be concatenated together
                num_workers=0, # if you have many cpu cores, we find that num_workers = 4 is fast for large datasets
                sql=False, # if True, embeddings will be stored in SQLite database
                sql_db_path='embeddings.db',
                save=True, # if True, embeddings will be saved as a .pth file
                save_path='embeddings.pth',
            )
            >>> # embedding_dict is a dictionary mapping sequences to their embeddings as tensors for .pth or numpy arrays for sql
        """
        sequences = list(set([seq[:max_len] for seq in sequences]))
        sequences = sorted(sequences, key=len, reverse=True)
        collate_fn = build_collator(tokenizer)
        device = self.device
        pooler = Pooler(pooling_types) if not full_embeddings else None

        def get_embeddings(residue_embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            if full_embeddings or residue_embeddings.ndim == 2: # if already pooled or want residue-wise embeddings
                return residue_embeddings
            else:
                return pooler(residue_embeddings, attention_mask)

        if sql:
            import sqlite3
            conn = sqlite3.connect(sql_db_path)
            c = conn.cursor()
            c.execute('CREATE TABLE IF NOT EXISTS embeddings (sequence text PRIMARY KEY, embedding blob)')
            already_embedded = self._read_sequences_from_db(sql_db_path)
            to_embed = [seq for seq in sequences if seq not in already_embedded]
            print(f"Found {len(already_embedded)} already embedded sequences in {sql_db_path}")
            print(f"Embedding {len(to_embed)} new sequences")
            if len(to_embed) > 0:
                dataset = ProteinDataset(to_embed)
                dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=False)
                with torch.no_grad():
                    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Embedding batches'):
                        seqs = to_embed[i * batch_size:(i + 1) * batch_size]
                        input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                        residue_embeddings = self.embed(input_ids, attention_mask).float() # sql requires float32
                        embeddings = get_embeddings(residue_embeddings, attention_mask).cpu()
                        for seq, emb, mask in zip(seqs, embeddings, attention_mask):
                            if full_embeddings:
                                emb = emb[mask.bool()]
                            c.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", 
                                    (seq, emb.cpu().numpy().tobytes()))
                        
                        if (i + 1) % 100 == 0:
                            conn.commit()
            
                conn.commit()
            conn.close()
            return None

        embeddings_dict = {}
        if os.path.exists(save_path):
            embeddings_dict = torch.load(save_path, map_location='cpu', weights_only=True)
            to_embed = [seq for seq in sequences if seq not in embeddings_dict]
            print(f"Found {len(embeddings_dict)} already embedded sequences in {save_path}")
            print(f"Embedding {len(to_embed)} new sequences")
        else:
            to_embed = sequences
            print(f"Embedding {len(to_embed)} new sequences")

        if len(to_embed) > 0:
            dataset = ProteinDataset(to_embed)
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=False)
            with torch.no_grad():
                for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Embedding batches'):
                    seqs = to_embed[i * batch_size:(i + 1) * batch_size]
                    input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                    residue_embeddings = self.embed(input_ids, attention_mask)
                    embeddings = get_embeddings(residue_embeddings, attention_mask).to(embed_dtype).cpu()
                    for seq, emb in zip(seqs, embeddings):
                        embeddings_dict[seq] = emb

        if save:
            torch.save(embeddings_dict, save_path)

        return embeddings_dict
