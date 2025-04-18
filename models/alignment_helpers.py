import biotite.sequence.align as align
import numpy as np
import torch
from typing import Union
from transformers import EsmTokenizer
from biotite.sequence.align import align_optimal
from biotite.sequence import ProteinSequence


def sanitize_sequence(sequence):
    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
    return ''.join(aa if aa in standard_aa else 'X' for aa in sequence.upper())


class NWScorer:
    def __init__(self, gap_score=-10):
        self.gap_score = gap_score
        self.canonical_aa = set("ACDEFGHIKLMNPQRSTVWY")
        self.matrix = align.SubstitutionMatrix.std_protein_matrix()
        self.gap_penalty = gap_score

    def _sanitize_sequence(self, sequence):
        return ''.join(aa if aa in self.canonical_aa else 'X' for aa in sequence.upper())

    def __call__(self, seq1: str, seq2: str) -> float:
        seq1, seq2 = ProteinSequence(self._sanitize_sequence(seq1)), ProteinSequence(self._sanitize_sequence(seq2))
        return align_optimal(seq1, seq2, self.matrix, gap_penalty=self.gap_penalty)[0].score


class AlignmentScorer:
    def __init__(self, gap_score=-10):
        self.gap_score = gap_score
        self.canonical_aa = set("ACDEFGHIKLMNPQRSTVWY")
        self.matrix = align.SubstitutionMatrix.std_protein_matrix()
        self.gap_penalty = gap_score

    def _sanitize_sequence(self, sequence):
        return ''.join(aa if aa in self.canonical_aa else 'X' for aa in sequence.upper())

    def __call__(self, seq1: str, seq2: str) -> float:
        seq1, seq2 = ProteinSequence(self._sanitize_sequence(seq1)), ProteinSequence(self._sanitize_sequence(seq2))
        max_len = max(len(seq1), len(seq2))
        aa = align_optimal(seq1, seq1, self.matrix, gap_penalty=self.gap_penalty)[0].score
        ab = align_optimal(seq1, seq2, self.matrix, gap_penalty=self.gap_penalty)[0].score
        return max_len / (aa - ab + max_len)
    

class AlignmentLossLike:
    def __init__(self, gap_score=-10):
        self.scorer = AlignmentScorer(gap_score)
        self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

    def __call__(self, logits: Union[np.ndarray, torch.Tensor], labels: Union[np.ndarray, torch.Tensor]) -> float:
        scores = []
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        for logit, label in zip(logits, labels):
            pred = logit.argmax(axis=-1)
            pred = pred.flatten().tolist()
            label = label.flatten().tolist()
            pred = self.tokenizer.decode(pred, skip_special_tokens=True).replace(' ', '')
            label = self.tokenizer.decode(label, skip_special_tokens=True).replace(' ', '')
            score = self.scorer(label, pred)
            scores.append(score)
        scores = np.array(scores)
        ideal_labels = np.ones_like(scores)
        loss = ideal_labels - scores
        return loss.mean(), scores


class SequenceComparator:
    def __init__(self, comparison_method="alignment"):
        self.comparison_method = comparison_method
        if comparison_method == "alignment":
            self.matrix = align.SubstitutionMatrix.std_protein_matrix()
            self.gap_penalty = -10
            self.run_call = self.alignment_call
        elif comparison_method == "indel":
            from rapidfuzz.fuzz import ratio
            self.indel_calc = ratio
            self.run_call = self.indel_call
        else:
            raise ValueError(f"Invalid comparison method: {comparison_method}")

    def alignment_call(self, seq1, seq2):
        seq1, seq2 = ProteinSequence(self.sanitize_sequence(seq1)), ProteinSequence(self.sanitize_sequence(seq2))
        
        alignment = align.align_optimal(seq1, seq2, 
                                        self.matrix,
                                        gap_penalty=self.gap_penalty,
                                        terminal_penalty=False)
        
        # Get aligned sequences as strings
        aligned_seq1, aligned_seq2 = alignment[0].get_gapped_sequences()
        aligned_seq1_str = str(aligned_seq1)
        aligned_seq2_str = str(aligned_seq2)
        
        identical_count = sum(a == b for a, b in zip(aligned_seq1_str, aligned_seq2_str) 
                            if a != '-' and b != '-')
        total_length = len(seq1)  # Use the length of the original sequence
        return (identical_count / total_length) * 100
    
    def indel_call(self, seq1, seq2):
        return self.indel_calc(seq1, seq2)

    def __call__(self, seq1, seq2):
        return self.run_call(seq1, seq2)


def alignment_score(label, pred, gap_score=-10):
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    label = ''.join(c if c in amino_acids else 'X' for c in label)
    pred = ''.join(c if c in amino_acids else 'X' for c in pred)
    
    label_obj = ProteinSequence(label)
    pred_obj = ProteinSequence(pred)
    
    max_len = max(len(label), len(pred))
    
    # Score between label and itself
    aa_alignment = align.align_optimal(label_obj, label_obj, 
                                     matrix,
                                     gap_penalty=gap_score,
                                     terminal_penalty=False)
    aa = aa_alignment[0].score
    
    # Score between label and prediction
    ab_alignment = align.align_optimal(label_obj, pred_obj, 
                                     matrix,
                                     gap_penalty=gap_score,
                                     terminal_penalty=False)
    ab = ab_alignment[0].score
    
    score = max_len / (aa - ab + max_len)
    return score


def analyze_two_seqs(label, pred, gap_score=-10):
    # ANSI escape codes for colors
    label = sanitize_sequence(label)
    pred = sanitize_sequence(pred)
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    
    # Get biotite matrix for scoring
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    
    # Convert to biotite sequences
    label_obj = seq.ProteinSequence(label)
    pred_obj = seq.ProteinSequence(pred)
    
    # Get alignment
    alignment = align.align_optimal(label_obj, pred_obj, 
                                  matrix,
                                  gap_penalty=gap_score,
                                  terminal_penalty=False)
    
    # Get aligned sequences as strings
    aligned_label_obj, aligned_pred_obj = alignment[0].get_gapped_sequences()
    aligned_label = str(aligned_label_obj)
    aligned_pred = str(aligned_pred_obj)
    
    result = []
    positive_count = 0
    total_aligned = 0
    
    for a, b in zip(aligned_label, aligned_pred):
        total_aligned += 1
        if a == '-' or b == '-':
            result.append((BLUE, a, b, gap_score))
        else:
            # Access the substitution matrix score
            score = matrix.get_score(a, b)
            if score > 0:
                result.append((GREEN, a, b, score))
                positive_count += 1
            elif score < 0:
                result.append((RED, a, b, score))
            else:
                result.append((YELLOW, a, b, score))
                positive_count += 1
                
    percent_positive = (positive_count / total_aligned) * 100 if total_aligned > 0 else 0
    a_score = alignment_score(label, pred, gap_score)

    label_line = ""
    pred_line = ""
    score_line = ""

    for color, a, b, score in result:
        label_line += f"{color}{a}{RESET}"
        pred_line += f"{color}{b}{RESET}"
        if a == '-' or b == '-':
            score_line += f"{color}-{RESET}"
        elif score > 0:
            score_line += f"{color}+{RESET}"
        elif score < 0:
            score_line += f"{color}-{RESET}"
        else:
            score_line += f"{color}0{RESET}"

    print(f"\nPositive: {percent_positive:.2f}%")
    print(f"Score: {a_score:.4f}")
    print(f"Label: {label_line}")
    print(f"Pred : {pred_line}")
    print(f"Score: {score_line}\n")
    return a_score
