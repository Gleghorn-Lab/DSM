from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy.spatial.distance import jensenshannon
from wordcloud import WordCloud
from collections import Counter


sns.set_theme(style="ticks", context="paper", font_scale=1.6)


def _k(p: Path) -> int:
    return int(re.search(r"kmer(\d+)", p.stem).group(1))


def load_tbls(raw_dir: Path, prefix: str) -> dict[int, pd.DataFrame]:
    return {_k(p): pd.read_csv(p, index_col=0)
            for p in sorted(raw_dir.glob(f"{prefix}_kmer*.csv"), key=_k)}


def chi_p_js(df: pd.DataFrame) -> tuple[float, float, float]:
    a, b = df["count_A"].values, df["count_B"].values
    chi2, p, *_ = chi2_contingency(np.vstack((a, b)), correction=False)
    js = jensenshannon(a / a.sum(), b / b.sum(), base=2.0) ** 2
    return chi2, p, js


def add_scatter(ax, df: pd.DataFrame, k: int):
    x, y = df["ratio_A"].clip(1e-12), df["ratio_B"].clip(1e-12)
    ax.scatter(x, y, s=25, alpha=0.65)
    lo, hi = np.min([x.min(), y.min()]), np.max([x.max(), y.max()])
    ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Natural frequency", fontsize=14, labelpad=20)
    ax.set_ylabel("Generated frequency", fontsize=14)
    
    # More descriptive title based on k value
    if k == 1:
        title = f"{k}-mer Distribution"
    elif k == 2:
        title = f"{k}-mer Distribution"
    elif k == 3:
        title = f"{k}-mer Distribution"
    else:
        title = f"{k}-mer Distribution"
    
    ax.set_title(title, fontsize=16)
    
    χ2, p, js = chi_p_js(df)
    ax.text(0.02, 0.98,
            f"$\\chi^2$={χ2:.0f}\n$p$={p:.1e}\n$JS$={js:.4f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=16,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75))


def make_table(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    df = df.copy()
    df["abs"] = df["log2_fold"].abs()
    dissim = df.nlargest(top, "abs")
    simil  = df.nsmallest(top, "abs")
    ordered = pd.concat([dissim, simil])
    ordered["rank"] = range(1, len(ordered)+1)
    table = ordered[["rank", "log2_fold", "ratio_A", "ratio_B"]]
    table.rename(columns={"log2_fold": "log2-fold",
                          "ratio_A": "Nat freq",
                          "ratio_B": "Gen freq"}, inplace=True)
    return table


def add_table(ax, df: pd.DataFrame, k: int):
    tbl = make_table(df)
    ax.axis("off")
    the_table = ax.table(cellText=np.round(tbl.values, 4),
                         colLabels=tbl.columns,
                         loc="center",
                         cellLoc="center")
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1, 1.5)   # more breathing room
    
    # More descriptive title based on k value
    if k == 2:
        title = f"{k}-mers"
    elif k == 3:
        title = f"{k}-mers"
    else:
        title = f"{k}-mers"
        
    ax.set_title(title + "\n10 most dissimilar ↑  |  10 most similar ↓",
                 pad=15, fontsize=14)


def scatter_figure(prefix: str, tbls, out):
    ks = sorted(tbls)
    fig, axes = plt.subplots(1, len(ks),
                             figsize=(6.0*len(ks), 6.0),
                             constrained_layout=True)
    if len(ks) == 1: axes = [axes]
    for ax, k in zip(axes, ks): add_scatter(ax, tbls[k], k)
    
    # More descriptive main titles based on prefix
    if prefix == "aa":
        title = "Amino Acid Distribution Comparison"
    elif prefix == "ss4":
        title = "Secondary Structure (4-class) Distribution Comparison"
    elif prefix == "ss9":
        title = "Secondary Structure (9-class) Distribution Comparison"
    else:
        title = prefix.upper() + " Distribution Comparison"
    
    fig.suptitle(title, fontsize=18, weight="bold")
    fig.savefig(out / f"{prefix}_comparison.png", dpi=300)
    plt.close(fig)


def annotation_figure(tbls: dict[int, pd.DataFrame], out: Path):
    ks_present = sorted(tbls)
    fig, axes = plt.subplots(1, len(ks_present),
                             figsize=(6.0*len(ks_present), 6.0),
                             constrained_layout=True)
    if len(ks_present) == 1: axes = [axes]

    for ax, k in zip(axes, ks_present):
        if k == 1:
            add_scatter(ax, tbls[k], k)
        else:
            add_table(ax, tbls[k], k)

    fig.suptitle("Protein Annotation Distribution Comparison", fontsize=18, weight="bold")
    fig.savefig(out / "annotations_comparison.png", dpi=300)
    plt.close(fig)


def create_annotation_wordclouds(raw_dir: Path, out: Path):
    """
    Create word clouds of natural and generated sequence annotations.
    Colors indicate similarity in frequency between datasets:
    - Red for similar frequencies
    - Blue for different frequencies
    """
    # Load raw data
    input_data = pd.read_csv(raw_dir / "input_data.csv")
    
    # Extract annotations
    natural_annotations = ';'.join(input_data['natural_annotations'].dropna())
    generated_annotations = ';'.join(input_data['generated_annotations'].dropna())
    #natural_annotations = natural_annotations.replace(' ', '-')
    #generated_annotations = generated_annotations.replace(' ', '-')

    remove_terms = [
        'Reference proteome'
    ]
    
    # manually calculate the frequences of each entity after split by ';'
    natural_annotations = natural_annotations.split(';')
    generated_annotations = generated_annotations.split(';')

    for term in remove_terms:
        natural_annotations = [t for t in natural_annotations if t != term]
        generated_annotations = [t for t in generated_annotations if t != term]

    natural_annotations_counts = Counter(natural_annotations)
    generated_annotations_counts = Counter(generated_annotations)
    
    natural_annotations_freq = {k: v / sum(natural_annotations_counts.values()) for k, v in natural_annotations_counts.items()}
    generated_annotations_freq = {k: v / sum(generated_annotations_counts.values()) for k, v in generated_annotations_counts.items()}
    
    # Get frequency data for coloring
    # Load the annotations comparison data from raw data
    ann_data = pd.read_csv(raw_dir / "annotations_kmer1.csv", index_col=0)
    
    # Create a mapping of words to their log2_fold and use this for coloring
    word_metrics, word_similarity = {}, {}
    max_fold_change, min_fold_change = 0, 0
    for word, row in ann_data.iterrows():
        word_metrics[word] = {
            'log2_fold': row['log2_fold'],
            'ratio_A': row['ratio_A'],
            'ratio_B': row['ratio_B']
        }
        word_similarity[str(word).lower()] = row['log2_fold']
        max_fold_change = max(max_fold_change, row['log2_fold'])
        min_fold_change = min(min_fold_change, row['log2_fold'])
    

    # Custom color function based on log2 fold change
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        if word.lower() in word_similarity:
            log2_fold = word_similarity[word.lower()]
        else:
            log2_fold = 0
        
        # Normalize log2_fold to [0,1] range
        normalized_value = (log2_fold - min_fold_change) / (max_fold_change - min_fold_change)
        # Ensure the value is within bounds
        normalized_value = max(0, min(1, normalized_value))
        
        # Scale to RGB range (0-255)
        if abs(log2_fold) < 2:  # similar - red
            return f"rgb(255, 0, 0)"
        else:  # different - blue
            return f"rgb(0, 0, 255)"
    
    # Create figure with dark border
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    fig.patch.set_facecolor('white')
    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(2)
    
    # Natural annotations wordcloud
    natural_wc = WordCloud(
        width=1000, height=800,
        background_color='white',
        max_words=500,
        min_font_size=8,
        max_font_size=48,
        collocations=False,
        contour_color='black',
        contour_width=2,
        random_state=42
    ).generate_from_frequencies(natural_annotations_freq)
    
    # Apply custom color function
    natural_wc.recolor(color_func=color_func, random_state=42)
    
    axes[0].imshow(natural_wc, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title('Natural Protein Sequence Annotations', fontsize=18, weight='bold')
    
    # Add border to subplot
    for spine in axes[0].spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
    
    # Generated annotations wordcloud
    generated_wc = WordCloud(
        width=1000, height=800,
        background_color='white',
        max_words=500,
        min_font_size=8,
        max_font_size=48,
        collocations=False,
        contour_color='black',
        contour_width=2,
        random_state=42
    ).generate_from_frequencies(generated_annotations_freq)
    
    # Apply same color function for consistency
    generated_wc.recolor(color_func=color_func, random_state=42)
    
    axes[1].imshow(generated_wc, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title('Generated Protein Sequence Annotations', fontsize=18, weight='bold')
    
    # Add border to subplot
    for spine in axes[1].spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)

    # Add main title with explanation
    fig.suptitle('Protein Annotation Term Frequency Comparison', fontsize=20, weight='bold', y=0.98)
    fig.savefig(out / "annotation_wordclouds.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    import argparse
    import os

    def arg_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--token', type=str, default=None)
        parser.add_argument("--raw_dir", type=Path, default="raw_data")
        parser.add_argument("--out_dir", default="figures", type=Path)
        return parser.parse_args()

    args = arg_parser()
    args.raw_dir = Path(os.path.join('evaluation', 'comparisons', args.raw_dir))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for pref in ("aa", "ss4", "ss9"):
        tbls = load_tbls(args.raw_dir, pref)
        if tbls:
            scatter_figure(pref, tbls, args.out_dir)

    ann_tbls = load_tbls(args.raw_dir, "annotations")
    if ann_tbls:
        annotation_figure(ann_tbls, args.out_dir)
    
    # Create word clouds for annotations
    create_annotation_wordclouds(args.raw_dir, args.out_dir)

    print("PNGs written to", args.out_dir.resolve())


if __name__ == "__main__":
    main()
