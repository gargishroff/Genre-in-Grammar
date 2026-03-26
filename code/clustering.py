"""
clustering.py
=============
Day 3 — Unsupervised Clustering Analysis

Responsibilities:
  - run_kmeans()               : k-means (k=5) on PCA scores, ARI evaluation
  - run_hierarchical()         : Ward hierarchical clustering, dendrogram
  - plot_kmeans()              : scatter of k-means clusters vs. true genres
  - plot_dendrogram()          : hierarchical clustering dendrogram
  - plot_cluster_vs_genre()    : side-by-side confusion heatmap showing
                                 where clusters diverge from genre labels
  - print_clustering_summary() : text report of cluster purity and
                                 most-misassigned texts for qualitative analysis

All plots saved to config.OUTPUT_DIR.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import mode

import config


# =============================================================================
# K-MEANS CLUSTERING
# =============================================================================

def run_kmeans(df_pca: pd.DataFrame, n_clusters: int = 5):
    """
    Fit k-means (k = n_genres) on PC1–PC6 scores.

    Returns
    -------
    df_pca   : input DataFrame with 'cluster' column appended
    km       : fitted KMeans object
    ari      : Adjusted Rand Index vs. true genre labels
    label_map: dict mapping cluster_id → most frequent genre in that cluster
    """
    pc_cols = [c for c in df_pca.columns if c.startswith("PC")]
    X       = df_pca[pc_cols].values

    km = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_SEED,
                n_init=30, max_iter=500)
    df_pca = df_pca.copy()
    df_pca["cluster"] = km.fit_predict(X)

    true_codes = pd.Categorical(
        df_pca["genre"], categories=sorted(df_pca["genre"].unique())).codes
    ari = adjusted_rand_score(true_codes, df_pca["cluster"])

    # Map each cluster id to the dominant genre label
    label_map = {}
    for c in range(n_clusters):
        mask   = df_pca["cluster"] == c
        genres = df_pca.loc[mask, "genre"]
        label_map[c] = genres.value_counts().idxmax()

    print(f"\n[clustering] K-Means (k={n_clusters})")
    print(f"  Adjusted Rand Index : {ari:.4f}"
          f"  (1.0 = perfect | 0.0 = random)")
    print(f"\n  Cluster → dominant genre mapping:")
    for c, g in label_map.items():
        n_total  = (df_pca["cluster"] == c).sum()
        n_correct = ((df_pca["cluster"] == c) &
                     (df_pca["genre"] == g)).sum()
        print(f"    Cluster {c} → {g:10s}  "
              f"({n_correct}/{n_total} texts = "
              f"{n_correct/n_total*100:.0f}% pure)")

    return df_pca, km, ari, label_map


# =============================================================================
# HIERARCHICAL CLUSTERING
# =============================================================================

def run_hierarchical(df_pca: pd.DataFrame):
    """
    Ward hierarchical clustering on PC1–PC3 scores.

    Returns
    -------
    Z        : linkage matrix
    labels   : list of text labels for dendrogram leaves
    """
    pc_cols = [c for c in df_pca.columns if c.startswith("PC")][:3]
    X       = df_pca[pc_cols].values
    labels  = [f"{row.genre[:3]}-{row.text_name[-12:]}"
               for _, row in df_pca.iterrows()]
    Z = linkage(X, method="ward")
    print(f"\n[clustering] Hierarchical clustering (Ward, PC1–PC3)")
    print(f"  Texts clustered: {len(labels)}")
    return Z, labels


# =============================================================================
# PLOT 5 — K-MEANS SCATTER
# =============================================================================

def plot_kmeans(df_pca: pd.DataFrame, ari: float,
                label_map: dict, save: bool = True) -> None:
    """
    Scatter of PC1 vs PC2, coloured by k-means cluster assignment.
    True genre shown as marker shape overlay.
    """
    CLUSTER_COLORS = [
        "#E63946", "#2A9D8F", "#E9C46A", "#457B9D", "#F4A261",
    ]
    GENRE_MARKERS = {
        "Epic": "o", "Vedic": "s", "Kavya": "^",
        "Shastra": "D", "Puranic": "P"
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left panel: cluster assignments ──────────────────────────────────
    ax = axes[0]
    n_clusters = df_pca["cluster"].nunique()
    for c in range(n_clusters):
        mask = df_pca["cluster"] == c
        col  = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        ax.scatter(df_pca.loc[mask, "PC1"], df_pca.loc[mask, "PC2"],
                   color=col, alpha=0.6, s=55,
                   label=f"Cluster {c} ({label_map.get(c,'?')})",
                   edgecolors="white", linewidths=0.4, zorder=3)
    ax.axhline(0, color="grey", lw=0.4, ls=":")
    ax.axvline(0, color="grey", lw=0.4, ls=":")
    ax.set_xlabel("PC1 (Nominalization axis)", fontsize=10)
    ax.set_ylabel("PC2 (Syntactic complexity axis)", fontsize=10)
    ax.set_title(f"K-Means Clusters (k={n_clusters})\nARI = {ari:.3f}",
                 fontsize=11)
    ax.legend(fontsize=8, title="Cluster → Genre", title_fontsize=8)

    # ── Right panel: true genre labels ───────────────────────────────────
    ax = axes[1]
    for genre, grp in df_pca.groupby("genre"):
        col = config.GENRE_PALETTE.get(genre, "#888")
        mk  = GENRE_MARKERS.get(genre, "o")
        ax.scatter(grp["PC1"], grp["PC2"],
                   color=col, alpha=0.65, s=55, marker=mk,
                   label=genre, edgecolors="white",
                   linewidths=0.4, zorder=3)
    ax.axhline(0, color="grey", lw=0.4, ls=":")
    ax.axvline(0, color="grey", lw=0.4, ls=":")
    ax.set_xlabel("PC1 (Nominalization axis)", fontsize=10)
    ax.set_ylabel("PC2 (Syntactic complexity axis)", fontsize=10)
    ax.set_title("True Genre Labels\n(for comparison)", fontsize=11)
    ax.legend(fontsize=8, title="Genre", title_fontsize=8)

    fig.suptitle(
        "K-Means Clustering vs. True Genre Labels\n"
        f"Adjusted Rand Index = {ari:.3f}  "
        "(1.0 = perfect recovery | 0.0 = random)",
        fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save:
        _savefig("05_kmeans_vs_genre.png")
    plt.show()


# =============================================================================
# PLOT 6 — DENDROGRAM
# =============================================================================

def plot_dendrogram(Z, labels: list, df_pca: pd.DataFrame,
                    save: bool = True) -> None:
    """
    Hierarchical clustering dendrogram.
    Leaf labels are coloured by true genre.
    """
    genre_colors = {g: config.GENRE_PALETTE.get(g, "#888")
                    for g in config.GENRES}

    # Build color list for leaf labels
    leaf_colors = []
    for _, row in df_pca.iterrows():
        leaf_colors.append(
            genre_colors.get(row["genre"], "#888888"))

    fig, ax = plt.subplots(figsize=(max(16, len(labels) // 3), 5))
    dend = dendrogram(
        Z, labels=labels, ax=ax,
        leaf_rotation=90, leaf_font_size=5.5,
        color_threshold=0.55 * max(Z[:, 2]))
    ax.set_title(
        "Hierarchical Clustering of Sanskrit Texts\n"
        "(Ward linkage on PC1–PC3; shorter bar = more similar)",
        fontsize=11)
    ax.set_ylabel("Ward Distance", fontsize=10)

    # Colour leaf labels by genre
    xlbls = ax.get_xmajorticklabels()
    for lbl, col in zip(xlbls, leaf_colors):
        lbl.set_color(col)

    # Legend
    handles = [mpatches.Patch(color=c, label=g)
               for g, c in genre_colors.items()
               if g in df_pca["genre"].values]
    ax.legend(handles=handles, title="Genre", loc="upper right",
              fontsize=8, title_fontsize=8)
    plt.tight_layout()

    if save:
        _savefig("06_dendrogram.png")
    plt.show()


# =============================================================================
# PLOT 7 — CLUSTER vs GENRE CONFUSION HEATMAP
# =============================================================================

def plot_cluster_genre_heatmap(df_pca: pd.DataFrame,
                                label_map: dict,
                                save: bool = True) -> None:
    """
    Heatmap showing how k-means clusters map onto true genre labels.
    Rows = clusters (relabelled with dominant genre).
    Columns = true genres.
    Cell values = number of texts.

    This is the core 'clustering vs. classification' comparison:
    high off-diagonal cells = genres that are morphologically confused.
    """
    genres   = sorted(df_pca["genre"].unique())
    clusters = sorted(df_pca["cluster"].unique())

    # Build matrix
    mat = np.zeros((len(clusters), len(genres)), dtype=int)
    for i, c in enumerate(clusters):
        for j, g in enumerate(genres):
            mat[i, j] = ((df_pca["cluster"] == c) &
                         (df_pca["genre"] == g)).sum()

    row_labels = [f"Cluster {c}\n(→{label_map.get(c,'?')[:3]})"
                  for c in clusters]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(mat, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=genres, yticklabels=row_labels,
                linewidths=0.5, ax=ax,
                annot_kws={"size": 10, "fontweight": "bold"})
    ax.set_xlabel("True Genre Label", fontsize=11)
    ax.set_ylabel("K-Means Cluster", fontsize=11)
    ax.set_title(
        "Clustering vs. Classification: Where Do They Disagree?\n"
        "(off-diagonal cells = morphologically confused genre pairs)",
        fontsize=11)
    plt.tight_layout()

    if save:
        _savefig("07_cluster_genre_heatmap.png")
    plt.show()


# =============================================================================
# TEXT SUMMARY
# =============================================================================

def print_clustering_summary(df_pca: pd.DataFrame,
                              label_map: dict) -> None:
    """
    Print which texts were assigned to unexpected clusters.
    These are your primary candidates for qualitative analysis.
    """
    print("\n[clustering] Cluster–Genre Disagreements "
          "(candidates for qualitative analysis):")
    print("  These texts were assigned to a cluster dominated by a "
          "different genre.")
    print()

    disagreements = []
    for _, row in df_pca.iterrows():
        dominant = label_map.get(row["cluster"], "?")
        if dominant != row["genre"]:
            disagreements.append({
                "text_name"     : row["text_name"],
                "true_genre"    : row["genre"],
                "cluster_genre" : dominant,
                "cluster"       : row["cluster"],
                "PC1"           : round(row["PC1"], 2),
                "PC2"           : round(row["PC2"], 2),
            })

    df_dis = pd.DataFrame(disagreements)
    if df_dis.empty:
        print("  No disagreements — perfect cluster–genre alignment!")
        return

    # Group by true genre for readability
    for genre, grp in df_dis.groupby("true_genre"):
        print(f"  {genre} texts assigned elsewhere ({len(grp)} texts):")
        for _, row in grp.iterrows():
            print(f"    → '{row['text_name']}' "
                  f"assigned to {row['cluster_genre']}-dominant cluster "
                  f"[PC1={row['PC1']}, PC2={row['PC2']}]")
    print()


# =============================================================================
# HELPER
# =============================================================================

def _savefig(filename: str) -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[clustering] Saved → {path}")