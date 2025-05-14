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
    

class GetAlignmentScoreFromLogits:
    def __init__(self, gap_score=-10):
        self.scorer = AlignmentScorer(gap_score)
        self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.canonical_aa = set("ACDEFGHIKLMNPQRSTVWY")
        self.canonical_tokens = set([self.tokenizer.encode(aa, add_special_tokens=False)[0] for aa in list(self.canonical_aa)])
        self.alanine_token = self.tokenizer.encode('A', add_special_tokens=False)[0]
        self.vocab_size = len(self.tokenizer)

    def _sanitize_pred(self, pred: list[int]) -> list[int]:
        return [token if token in self.canonical_tokens else self.alanine_token for token in pred]

    def batched_call(self, logits: Union[np.ndarray, torch.Tensor], labels: Union[np.ndarray, torch.Tensor]) -> float:
        scores = []

        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        for logit, label in zip(logits, labels):
            label = label.flatten().tolist()
            label = self.tokenizer.decode(label, skip_special_tokens=True).replace(' ', '')
            logit = logit.reshape(-1, self.vocab_size)
            pred = logit.argmax(axis=-1)
            pred = pred.flatten().tolist()
            pred = pred[1:len(label)+1] # remove cls and padding
            pred = self._sanitize_pred(pred)
            pred = self.tokenizer.decode(pred, skip_special_tokens=True).replace(' ', '')
            score = self.scorer(label, pred)
            scores.append(score)
        return np.array(scores)

    def __call__(self, logits: Union[np.ndarray, torch.Tensor], labels: Union[np.ndarray, torch.Tensor]) -> float:
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        labels = labels.flatten().tolist()
        labels = self.tokenizer.decode(labels, skip_special_tokens=True).replace(' ', '')
        logits = logits.reshape(-1, self.vocab_size)
        preds = logits.argmax(axis=-1)
        preds = preds.flatten().tolist()
        preds = preds[1:len(labels)+1] # remove cls and padding
        preds = self._sanitize_pred(preds)
        preds = self.tokenizer.decode(preds, skip_special_tokens=True).replace(' ', '')
        score = self.scorer(labels, preds)
        return score


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


def analyze_two_seqs(label, pred, gap_score=-10, wrap_length=100, save_fig=None):
    # ANSI escape codes for colors
    label = sanitize_sequence(label)
    pred = sanitize_sequence(pred)
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BLACK_BOX = '\033[40m'  # Black background
    
    # Get biotite matrix for scoring
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    
    # Convert to biotite sequences
    label_obj = ProteinSequence(label)
    pred_obj = ProteinSequence(pred)
    
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
            result.append((BLUE, a, b, gap_score, True))  # Different
        else:
            # Access the substitution matrix score
            score = matrix.get_score(a, b)
            is_different = a != b
            if score > 0:
                result.append((GREEN, a, b, score, is_different))
                positive_count += 1
            elif score < 0:
                result.append((RED, a, b, score, is_different))
            else:
                result.append((YELLOW, a, b, score, is_different))
                positive_count += 1
                
    percent_positive = (positive_count / total_aligned) * 100 if total_aligned > 0 else 0
    a_score = alignment_score(label, pred, gap_score)

    # Prepare wrapped output
    def wrap_result(result, wrap_length):
        chunks = []
        for i in range(0, len(result), wrap_length):
            chunks.append(result[i:i+wrap_length])
        return chunks
    
    chunks = wrap_result(result, wrap_length)
    
    print(f"\nPositive: {percent_positive:.2f}%")
    print(f"Score: {a_score:.4f}")
    
    for chunk in chunks:
        label_line = ""
        pred_line = ""
        score_line = ""
        
        for color, a, b, score, is_different in chunk:
            # Add black box background for differing residues
            if is_different:
                label_line += f"{color}{BLACK_BOX}{a}{RESET}"
                pred_line += f"{color}{BLACK_BOX}{b}{RESET}"
            else:
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

        print(f"Label: {label_line}")
        print(f"Pred : {pred_line}")
        print(f"Score: {score_line}\n")
    
    # Save visualization if requested
    if save_fig:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.colors import LinearSegmentedColormap
            import numpy as np
            
            # Set up the figure - add extra space on the right for the legend
            fig_height = max(3, len(chunks) * 0.6)
            fig, ax = plt.subplots(figsize=(12, fig_height))
            
            # Generate matrix representation for visualization
            matrix_data = []
            
            for i in range(len(chunks)):
                row_label = []
                row_pred = []
                row_scores = []
                
                for color, a, b, score, is_different in chunks[i]:
                    # Convert to numeric representation for visualization
                    if a == '-' or b == '-':
                        row_scores.append(-0.5)  # Gap
                    elif score > 0:
                        row_scores.append(1.0)  # Positive
                    elif score < 0:
                        row_scores.append(-1.0)  # Negative
                    else:
                        row_scores.append(0.0)  # Neutral
                    
                    row_label.append(a)
                    row_pred.append(b)
                
                # Pad shorter rows with empty spaces
                if len(row_scores) < wrap_length:
                    padding = wrap_length - len(row_scores)
                    row_scores.extend([np.nan] * padding)  # Using NaN instead of None
                    row_label.extend([' '] * padding)
                    row_pred.extend([' '] * padding)
                
                matrix_data.append((row_label, row_pred, row_scores))
            
            # Create better colormap where neutral values are clearly visible
            # Red for negative, blue for gaps, light gray for background, 
            # gold/yellow for neutral, green for positive
            colors = [
                (0.8, 0, 0),      # Red - negative
                (0, 0, 0.8),      # Blue - gap
                (0.9, 0.9, 0.9),  # Light gray - background (NaN)
                (0.95, 0.85, 0),  # Gold - neutral (0 score)
                (0, 0.8, 0)       # Green - positive
            ]
            positions = [0, 0.25, 0.5, 0.5, 1]
            cmap = LinearSegmentedColormap.from_list("alignment_cmap", list(zip(positions, colors)))
            
            # Create an image-like representation
            im_height = len(matrix_data) * 2  # 2 rows per chunk (label & pred)
            im_data = np.full((im_height, wrap_length), np.nan, dtype=float)  # Using NaN
            
            for i, (label_row, pred_row, scores) in enumerate(matrix_data):
                row_idx = i * 2
                for j, (l, p, s) in enumerate(zip(label_row, pred_row, scores)):
                    if not np.isnan(s):  # Check for NaN instead of None
                        # Convert score to color value between -1 and 1
                        # -1: negative (red), -0.5: gap (blue), 0: neutral (yellow), 1: positive (green)
                        im_data[row_idx, j] = s
                        im_data[row_idx+1, j] = s
            
            # Normalize to 0-1 range for plotting
            norm_data = np.copy(im_data)  # Copy instead of creating new array
            mask = ~np.isnan(im_data)  # Using isnan check
            if np.any(mask):  # Only normalize if there are non-NaN values
                norm_data[mask] = (im_data[mask] + 1) / 2  # Convert from [-1,1] to [0,1]
            
            # Plot the image with the new colormap
            im = ax.imshow(norm_data, cmap=cmap, aspect='auto', interpolation='nearest')
            
            # Add text annotations
            for i, (label_row, pred_row, scores) in enumerate(matrix_data):
                row_idx = i * 2
                for j, (l, p, s) in enumerate(zip(label_row, pred_row, scores)):
                    if l != ' ':
                        # Check if position is valid before accessing
                        if 0 <= row_idx < im_height and 0 <= j < wrap_length and not np.isnan(norm_data[row_idx, j]):
                            # Use black text for yellow/gold (neutral) and green (positive) backgrounds
                            # Use white text for red (negative) and blue (gap) backgrounds
                            text_color = 'black' if norm_data[row_idx, j] >= 0.5 else 'white'
                            ax.text(j, row_idx, l, ha='center', va='center', color=text_color, fontsize=8)
                    if p != ' ':
                        # Check if position is valid before accessing
                        if 0 <= row_idx+1 < im_height and 0 <= j < wrap_length and not np.isnan(norm_data[row_idx+1, j]):
                            text_color = 'black' if norm_data[row_idx+1, j] >= 0.5 else 'white'
                            ax.text(j, row_idx+1, p, ha='center', va='center', color=text_color, fontsize=8)
            
            # Add labels
            chunk_labels = [f"Label ({i*wrap_length+1}-{min((i+1)*wrap_length, total_aligned)})" for i in range(len(chunks))]
            chunk_preds = [f"Pred ({i*wrap_length+1}-{min((i+1)*wrap_length, total_aligned)})" for i in range(len(chunks))]
            
            # Create custom y-tick positions and labels
            y_positions = []
            y_labels = []
            for i in range(len(chunks)):
                y_positions.extend([i*2, i*2+1])
                y_labels.extend([chunk_labels[i], chunk_preds[i]])
            
            ax.set_yticks(y_positions)
            ax.set_yticklabels(y_labels)
            
            # Remove x-ticks
            ax.set_xticks([])
            
            # Add a title with scores
            plt.title(f"Sequence Alignment\nPositive: {percent_positive:.2f}%, Alignment Score: {a_score:.2f}")
            
            # Add only a legend (no colorbar)
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=colors[0], label='Negative score'),
                Patch(facecolor=colors[1], label='Gap'),
                Patch(facecolor=colors[3], label='Neutral score'),
                Patch(facecolor=colors[4], label='Positive score')
            ]
            
            # Adjust figure to make room for the legend on the right
            plt.tight_layout(rect=[0, 0, 0.8, 1])  # Leave 20% of width for legend
            
            # Create a new axis for the legend on the right side
            legend_ax = fig.add_axes([0.82, 0.5, 0.15, 0.4])  # [left, bottom, width, height]
            legend_ax.axis('off')  # Turn off axis
            legend = legend_ax.legend(handles=legend_elements, loc='center left', frameon=True)
            
            # Save the figure
            plt.savefig(save_fig, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Visualization saved to {save_fig}")
            
        except ImportError:
            print("Could not create visualization: matplotlib is required.")
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
    
    return a_score
