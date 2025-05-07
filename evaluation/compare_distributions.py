import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple
from scipy.stats import chi2_contingency
from scipy.spatial.distance import jensenshannon
from IPython.display import display


AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def _kmer_counter(seqs: List[str], k: int) -> Counter:
    """
    Fast k‑mer counter that skips characters outside the 20 canonical AAs.
    """
    kmers = Counter()
    for s in seqs:
        s = s.upper()
        for i in range(len(s) - k + 1):
            kmer = s[i:i+k]
            if all(c in AA20 for c in kmer):   # ignore gaps/ambiguous
                kmers[kmer] += 1
    return kmers


def _align_counters(c1: Counter, c2: Counter) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Return aligned count arrays and the shared alphabet order.
    """
    alphabet = sorted(set(c1) | set(c2))
    a = np.array([c1[k] for k in alphabet], dtype=np.int64)
    b = np.array([c2[k] for k in alphabet], dtype=np.int64)
    return a, b, alphabet


def compare_corpora_kmers(
    corpus_A: List[str],
    corpus_B: List[str],
    ks: Tuple[int, ...] = (1, 2, 3),
    pseudo: float = 1e-9,
) -> Dict[int, Dict[str, float | pd.DataFrame]]:
    """
    Compare two corpora at multiple k‑mer sizes.

    Returns a dict keyed by k.  Each entry contains:
        * 'chi2'   - χ² statistic
        * 'dof'    - degrees of freedom
        * 'p'      - p-value
        * 'js'     - Jensen–Shannon divergence
        * 'table'  - tidy DataFrame with counts & frequencies of both corpora
    """
    results = {}
    for k in ks:
        cnt_A = _kmer_counter(corpus_A, k)
        cnt_B = _kmer_counter(corpus_B, k)

        a, b, alphabet = _align_counters(cnt_A, cnt_B)
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


if __name__ == "__main__":
    # py -m evaluation.compare_distributions
    seqs1 = ["MKTFFVAI", "GGLVTR", "MKVAA"]
    seqs2 = ["MKTFFVAR", "GGVVTR", "MKAAA"]

    stats = compare_corpora_kmers(seqs1, seqs2)

    for k, res in stats.items():
        print(f"\n=== {k}-mer comparison ===")
        print(f"χ² = {res['chi2']:.2f}  (dof={res['dof']}),  p = {res['p']:.3g}")
        print(f"Jensen-Shannon divergence = {res['js']:.4f} bits")
        display(res["table"].sort_values("log2_fold", ascending=False).head(10))

