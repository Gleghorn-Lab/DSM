import random
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset

"""
FromList takes lists of matched data
FromHF takes HuggingFace datasets
FromDisk takes data from a SQLite database
"""


class IterableDatasetFromHF(IterableDataset):
    def __init__(self, dataset, col_name='seqs', **kwargs):
        """
        Wrap a streaming Hugging Face dataset (IterableDataset) into a PyTorch IterableDataset.
        
        Args:
            dataset (IterableDataset): Streaming Hugging Face dataset.
            col_name (str): The column name containing the sequences.
        """
        self.dataset = dataset
        self.col_name = col_name

    def __iter__(self):
        for example in self.dataset:
            yield example[self.col_name]


class SequenceDatasetFromList(TorchDataset):
    def __init__(self, sequences, **kwargs):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class SequenceDatasetFromHF(TorchDataset):    
    def __init__(self, dataset, col_name='seqs', **kwargs):
        self.seqs = dataset[col_name]
        self.lengths = [len(seq) for seq in self.seqs]

    def avg(self):
        return sum(self.lengths) / len(self.lengths)

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return seq


class SequenceLabelDatasetFromLists(TorchDataset):
    def __init__(self, seqs, labels, **kwargs):
        self.seqs = seqs
        self.labels = labels
        self.lengths = [len(seq) for seq in self.seqs]

    def avg(self):
        return sum(self.lengths) / len(self.lengths)

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        label = self.labels[idx]
        return seq, label


class SequenceLabelDatasetFromHF(TorchDataset):    
    def __init__(self, dataset, col_name='seqs', label_col='labels', **kwargs):
        self.seqs = dataset[col_name]
        self.labels = dataset[label_col]
        self.lengths = [len(seq) for seq in self.seqs]

    def avg(self):
        return sum(self.lengths) / len(self.lengths)

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        label = self.labels[idx]
        return seq, label


class PairDatasetTrainFromLists(TorchDataset):
    def __init__(self, seqs_a, seqs_b, labels, **kwargs):
        self.seqs_a = seqs_a
        self.seqs_b = seqs_b
        self.labels = labels

    def avg(self):
        return sum(len(seqa) + len(seqb) for seqa, seqb in zip(self.seqs_a, self.seqs_b)) / len(self.seqs_a)

    def __len__(self):
        return len(self.seqs_a)
    
    def __getitem__(self, idx):
        seq_a, seq_b, label = self.seqs_a[idx], self.seqs_b[idx], self.labels[idx]
        if random.random() < 0.5:
            seq_a, seq_b = seq_b, seq_a
        return seq_a, seq_b, label


class PairDatasetTestFromLists(TorchDataset):
    def __init__(self, seqs_a, seqs_b, labels, **kwargs):
        self.seqs_a = seqs_a
        self.seqs_b = seqs_b
        self.labels = labels

    def avg(self):
        return sum(len(seqa) + len(seqb) for seqa, seqb in zip(self.seqs_a, self.seqs_b)) / len(self.seqs_a)

    def __len__(self):
        return len(self.seqs_a)
    
    def __getitem__(self, idx):
        return self.seqs_a[idx], self.seqs_b[idx], self.labels[idx]


class PairDatasetTrainHF(TorchDataset):
    def __init__(self, data, col_a, col_b, label_col, **kwargs):
        self.seqs_a = data[col_a]
        self.seqs_b = data[col_b]
        self.labels = data[label_col]

    def avg(self):
        return sum(len(seqa) + len(seqb) for seqa, seqb in zip(self.seqs_a, self.seqs_b)) / len(self.seqs_a)

    def __len__(self):
        return len(self.seqs_a)

    def __getitem__(self, idx):
        seq_a, seq_b = self.seqs_a[idx], self.seqs_b[idx]
        if random.random() < 0.5:
            seq_a, seq_b = seq_b, seq_a
        return seq_a, seq_b, self.labels[idx]
    

class PairDatasetTestHF(TorchDataset):
    def __init__(self, data, col_a, col_b, label_col, **kwargs):
        self.seqs_a = data[col_a]
        self.seqs_b = data[col_b]
        self.labels = data[label_col]

    def avg(self):
        return sum(len(seqa) + len(seqb) for seqa, seqb in zip(self.seqs_a, self.seqs_b)) / len(self.seqs_a)

    def __len__(self):
        return len(self.seqs_a)

    def __getitem__(self, idx):
        return self.seqs_a[idx], self.seqs_b[idx], self.labels[idx]
    

class NWDataset(TorchDataset):
    def __init__(self, dataset, sequence_col: str = 'Sequence', **kwargs):
        self.sequences = list(set(dataset[sequence_col]))

    def __len__(self):
        return len(self.sequences)

    def _mutate_seq(self, seq: str) -> str:
        # pick to random indicies, then shuffle in between
        if len(seq) < 3:
            return seq
        
        idx1 = random.randint(0, len(seq) - 3)
        idx2 = random.randint(idx1 + 2, len(seq) - 1)
        
        # Extract the segment to shuffle
        segment = list(seq[idx1:idx2])
        random.shuffle(segment)
        shuffled_segment = ''.join(segment)
        # Reconstruct the sequence with the shuffled segment
        return seq[:idx1] + shuffled_segment + seq[idx2:]
        
    def __getitem__(self, idx):
        seq_a = random.choice(self.sequences)
        if random.random() < 0.5:
            seq_b = random.choice(self.sequences)
        else:
            seq_b = self._mutate_seq(seq_a)
        
        if random.random() < 0.5:
            seq_a, seq_b = seq_b, seq_a

        return seq_a, seq_b


class NWDatasetEval(TorchDataset):
    def __init__(self, dataset, seq_a_col: str = 'SeqA', seq_b_col: str = 'SeqB', **kwargs):
        self.seq_a = dataset[seq_a_col]
        self.seq_b = dataset[seq_b_col]

    def __len__(self):
        return len(self.seq_a)
    
    def __getitem__(self, idx):
        return self.seq_a[idx], self.seq_b[idx]


class DiffATDataset(TorchDataset):
    def __init__(self, data, max_seq_length: int = 512, max_ann_length: int = 64):
        self.seqs = data['sequence']
        self.annotations = data['annotations']
        self.max_seq_length = max_seq_length
        self.max_ann_length = max_ann_length

    def _get_total_length(self, sequence: str, annotations: List[int]) -> int:
        return len(sequence) + len(annotations)

    def _get_ann(self, idx: int) -> List[int]:
        ann = self.annotations[idx]
        length = randint(8, self.max_ann_length)
        shuffle(ann)
        return sorted(ann[:length])

    def avg(self):
        total_len = sum(self._get_total_length(self.seqs[i], self.anns[i]) for i in range(len(self)))
        return total_len / len(self)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx][:self.max_seq_length]
        return seq, self._get_ann(idx)
