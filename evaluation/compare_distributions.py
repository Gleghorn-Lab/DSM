import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple, Set
from scipy.stats import chi2_contingency
from scipy.spatial.distance import jensenshannon
from IPython.display import display
import os


AA20 = set("ACDEFGHIKLMNPQRSTVWY")
SS4 = set("CHED")
SS9 = set("BCDEGHIST")


class CorpusComparator:
    def __init__(self, vocabulary: Set[str] = AA20):
        """
        Initialize the comparator with a vocabulary set.
        
        Args:
            vocabulary: Set of valid characters to include in k-mers
        """
        self.vocabulary = vocabulary

    def _kmer_counter(self, seqs: List[str], k: int) -> Counter:
        """
        Fast k‑mer counter that skips characters outside the vocabulary.
        """
        kmers = Counter()
        for s in seqs:
            s = s.upper()
            for i in range(len(s) - k + 1):
                kmer = s[i:i+k]
                if all(c in self.vocabulary for c in kmer):   # ignore gaps/ambiguous
                    kmers[kmer] += 1
        return kmers
    
    def _annotation_kmer_counter(self, seqs: List[str], k: int) -> Counter:
        """
        K-mer counter for annotation strings where units are separated by semicolons.
        """
        kmers = Counter()
        for s in seqs:
            if not s:  # Skip empty strings
                continue
            units = s.split(";")
            units = [u for u in units if u]  # Filter out empty units
            
            for i in range(len(units) - k + 1):
                kmer = ";".join(units[i:i+k])
                if kmer in self.vocabulary:  # Only count known annotations
                    kmers[kmer] += 1
        return kmers

    def _align_counters(self, c1: Counter, c2: Counter) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Return aligned count arrays and the shared alphabet order.
        """
        alphabet = sorted(set(c1) | set(c2))
        a = np.array([c1[k] for k in alphabet], dtype=np.int64)
        b = np.array([c2[k] for k in alphabet], dtype=np.int64)
        return a, b, alphabet

    def compare_corpora_kmers(
        self,
        corpus_A: List[str],
        corpus_B: List[str],
        ks: Tuple[int, ...] = (1, 2, 3),
        pseudo: float = 1e-9,
        is_annotation: bool = False,
    ) -> Dict[int, Dict[str, float | pd.DataFrame]]:
        """
        Compare two corpora at multiple k‑mer sizes.

        Args:
            corpus_A: First set of sequences
            corpus_B: Second set of sequences
            ks: Tuple of k-mer sizes to analyze
            pseudo: Pseudocount for frequency calculations
            is_annotation: Whether the input are annotation strings with semicolon-separated units

        Returns:
            A dict keyed by k. Each entry contains:
                * 'chi2'   - χ² statistic
                * 'dof'    - degrees of freedom
                * 'p'      - p-value
                * 'js'     - Jensen–Shannon divergence
                * 'table'  - tidy DataFrame with counts & frequencies of both corpora
        """
        results = {}
        for k in ks:
            if is_annotation:
                cnt_A = self._annotation_kmer_counter(corpus_A, k)
                cnt_B = self._annotation_kmer_counter(corpus_B, k)
            else:
                cnt_A = self._kmer_counter(corpus_A, k)
                cnt_B = self._kmer_counter(corpus_B, k)

            a, b, alphabet = self._align_counters(cnt_A, cnt_B)
            cont = np.vstack((a, b))

            chi2, p, dof, _ = chi2_contingency(cont, correction=False)

            pA = (a + pseudo) / (a.sum() + pseudo * len(a))
            pB = (b + pseudo) / (b.sum() + pseudo * len(b))
            js = jensenshannon(pA, pB, base=2) ** 2      # scipy returns √JS

            freq_A = pA
            freq_B = pB
            df = pd.DataFrame({
                f"count_A": a,
                f"count_B": b,
                f"ratio_A": freq_A,
                f"ratio_B": freq_B,
                "log2_fold": np.log2(freq_A / freq_B)
            }, index=alphabet)

            results[k] = {"chi2": chi2, "dof": dof, "p": p, "js": js, "table": df}

        return results
    
    @staticmethod
    def build_annotation_vocabulary(corpus_A: List[str], corpus_B: List[str]) -> Set[str]:
        """
        Build a vocabulary set from annotation strings in both corpora.
        
        Args:
            corpus_A: First set of annotation strings
            corpus_B: Second set of annotation strings
            
        Returns:
            Set of unique annotation units
        """
        vocabulary = set()
        
        for corpus in [corpus_A, corpus_B]:
            for s in corpus:
                if not s:  # Skip empty strings
                    continue
                units = s.split(";")
                units = [u for u in units if u]  # Filter out empty units
                vocabulary.update(units)
        
        return vocabulary


if __name__ == "__main__":
    # py -m evaluation.compare_distributions
    path = 'evaluation/annotated_comparisons.csv'
    df = pd.read_csv(path).astype(str)
    seqs1 = df['natural'].tolist()
    seqs2 = df['generated'].tolist()
    nat_ss4_preds = df['nat-ss4'].tolist()
    gen_ss4_preds = df['gen-ss4'].tolist()
    nat_ss9_preds = df['nat-ss9'].tolist()
    gen_ss9_preds = df['gen-ss9'].tolist()
    
    # Get annotation sequences if they exist
    nat_annotations = df['natural_annotations'].tolist() if 'natural_annotations' in df.columns else []
    gen_annotations = df['generated_annotations'].tolist() if 'generated_annotations' in df.columns else []

    # Create a directory to save raw data
    raw_data_dir = 'evaluation/raw_data'
    os.makedirs(raw_data_dir, exist_ok=True)

    # Create a comparator with default AA20 vocabulary
    comparator = CorpusComparator(vocabulary=AA20)
    stats = comparator.compare_corpora_kmers(seqs1, seqs2)

    # Open file for writing results
    with open('evaluation/compare_distributions.txt', 'w', encoding='utf-8') as f:
        # Amino acid comparison
        print('Amino acid comparison\n')
        f.write('Amino acid comparison\n')
        
        for k, res in stats.items():
            print(f"\n=== {k}-mer comparison ===")
            print(f"Chi^2 = {res['chi2']:.2f}  (dof={res['dof']}),  p = {res['p']:.3g}")
            print(f"Jensen-Shannon divergence = {res['js']:.4f} bits")
            display(res["table"].sort_values("log2_fold", ascending=False).head(10))
            
            f.write(f"\n=== {k}-mer comparison ===\n")
            f.write(f"Chi^2 = {res['chi2']:.2f}  (dof={res['dof']}),  p = {res['p']:.3g}\n")
            f.write(f"Jensen-Shannon divergence = {res['js']:.4f} bits\n")
            f.write(res["table"].sort_values("log2_fold", ascending=False).head(10).to_string())
            
            # Save raw data
            res["table"].to_csv(f"{raw_data_dir}/aa_kmer{k}.csv")
        
        # Secondary structure comparison
        ss_comparator = CorpusComparator(vocabulary=SS4)
        ss_stats = ss_comparator.compare_corpora_kmers(nat_ss4_preds, gen_ss4_preds)
        
        print('\n\nSecondary structure comparison 4\n')
        f.write('\n\nSecondary structure comparison 4\n')
        
        for k, res in ss_stats.items():
            print(f"\n=== {k}-mer comparison ===")
            print(f"Chi^2 = {res['chi2']:.2f}  (dof={res['dof']}),  p = {res['p']:.3g}")
            print(f"Jensen-Shannon divergence = {res['js']:.4f} bits")
            display(res["table"].sort_values("log2_fold", ascending=False).head(10))
            
            f.write(f"\n=== {k}-mer comparison ===\n")
            f.write(f"Chi^2 = {res['chi2']:.2f}  (dof={res['dof']}),  p = {res['p']:.3g}\n")
            f.write(f"Jensen-Shannon divergence = {res['js']:.4f} bits\n")
            f.write(res["table"].sort_values("log2_fold", ascending=False).head(10).to_string())
            
            # Save raw data
            res["table"].to_csv(f"{raw_data_dir}/ss4_kmer{k}.csv")

        # Secondary structure comparison
        ss_comparator = CorpusComparator(vocabulary=SS9)
        ss_stats = ss_comparator.compare_corpora_kmers(nat_ss9_preds, gen_ss9_preds)
        
        print('\n\nSecondary structure comparison 9\n')
        f.write('\n\nSecondary structure comparison 9\n')
        
        for k, res in ss_stats.items():
            print(f"\n=== {k}-mer comparison ===")
            print(f"Chi^2 = {res['chi2']:.2f}  (dof={res['dof']}),  p = {res['p']:.3g}")
            print(f"Jensen-Shannon divergence = {res['js']:.4f} bits")
            display(res["table"].sort_values("log2_fold", ascending=False).head(10))
            
            f.write(f"\n=== {k}-mer comparison ===\n")
            f.write(f"Chi^2 = {res['chi2']:.2f}  (dof={res['dof']}),  p = {res['p']:.3g}\n")
            f.write(f"Jensen-Shannon divergence = {res['js']:.4f} bits\n")
            f.write(res["table"].sort_values("log2_fold", ascending=False).head(10).to_string())
            
            # Save raw data
            res["table"].to_csv(f"{raw_data_dir}/ss9_kmer{k}.csv")
        
        # Annotation comparison (if annotations exist)
        if nat_annotations and gen_annotations:
            # Build annotation vocabulary from both corpora
            annotation_vocab = CorpusComparator.build_annotation_vocabulary(nat_annotations, gen_annotations)
            
            # Create a comparator with the annotation vocabulary
            ann_comparator = CorpusComparator(vocabulary=annotation_vocab)
            
            # Only perform 1-mer comparison for annotations
            ann_stats = ann_comparator.compare_corpora_kmers(
                nat_annotations, 
                gen_annotations, 
                ks=(1,),
                is_annotation=True
            )
            
            print('\n\nAnnotation comparison\n')
            f.write('\n\nAnnotation comparison\n')
            
            for k, res in ann_stats.items():
                print(f"\n=== {k}-mer comparison ===")
                print(f"Chi^2 = {res['chi2']:.2f}  (dof={res['dof']}),  p = {res['p']:.3g}")
                print(f"Jensen-Shannon divergence = {res['js']:.4f} bits")
                display(res["table"].sort_values("log2_fold", ascending=False).head(10))
                
                f.write(f"\n=== {k}-mer comparison ===\n")
                f.write(f"Chi^2 = {res['chi2']:.2f}  (dof={res['dof']}),  p = {res['p']:.3g}\n")
                f.write(f"Jensen-Shannon divergence = {res['js']:.4f} bits\n")
                f.write(res["table"].sort_values("log2_fold", ascending=False).head(10).to_string())
                
                # Save raw data
                res["table"].to_csv(f"{raw_data_dir}/annotations_kmer{k}.csv")

    # Save the original input data
    df.to_csv(f"{raw_data_dir}/input_data.csv", index=False)

