import random
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset

"""
FromList takes lists of matched data
FromHF takes HuggingFace datasets
FromDisk takes data from a SQLite database
"""


class IterableDatasetFromHF(IterableDataset):
    def __init__(self, dataset, col_name='seqs'):
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
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class SequenceDatasetFromHF(TorchDataset):    
    def __init__(self, dataset, col_name='seqs'):
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
    def __init__(self, seqs, labels):
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
    def __init__(self, dataset, col_name='seqs', label_col='labels'):
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
    def __init__(self, seqs_a, seqs_b, labels):
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
    def __init__(self, seqs_a, seqs_b, labels):
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
    def __init__(self, data, col_a, col_b, label_col):
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
    def __init__(self, data, col_a, col_b, label_col):
        self.seqs_a = data[col_a]
        self.seqs_b = data[col_b]
        self.labels = data[label_col]

    def avg(self):
        return sum(len(seqa) + len(seqb) for seqa, seqb in zip(self.seqs_a, self.seqs_b)) / len(self.seqs_a)

    def __len__(self):
        return len(self.seqs_a)

    def __getitem__(self, idx):
        return self.seqs_a[idx], self.seqs_b[idx], self.labels[idx]