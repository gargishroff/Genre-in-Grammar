"""
pca_analysis.py
===============
All Day 2 analysis: PCA fitting, dimensionality reduction, and visualizations.

Responsibilities:
  - run_pca()              : fit PCA on standardized feature matrix,
                             return scores DataFrame + fitted PCA object
  - plot_scree()           : bar + line chart of explained variance
  - plot_pca_biplot()      : PC1 vs PC2 scatter by genre with 95% confidence
                             ellipses and top-6 loading arrows
  - plot_loading_heatmap() : heatmap of feature loadings across PC1–PC4
  - plot_mci_by_genre()    : boxplot of Morphological Complexity Index
  - print_pca_summary()    : text summary of loadings interpretation

All plots are saved to config.OUTPUT_DIR.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config
from features import get_feature_columns


# =============================================================================
# PCA
# =============================================================================

def run_pca(df_feat: pd.DataFrame):
    """
    Standardize the feature matrix and fit PCA.

    Parameters
    ----------
    df_feat : DataFrame
        Output of features.extract_features().

    Returns
    -------
    pca     : fitted sklearn PCA object
    df_pca  : df_feat + PC1…PC6 score columns appended
    feat_cols: list of feature column names used
    scaler  : fitted StandardScaler (for inverse-transforming loadings)
    """
    feat_cols = get_feature_columns(df_feat)
    X         = df_feat[feat_cols].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_comp = min(len(feat_cols), len(df_feat))
    pca    = PCA(n_components=n_comp, random_state=config.RANDOM_SEED)
    scores = pca.fit_transform(X_scaled)

    df_pca = df_feat[["text_name", "genre"]].copy()
    if "period" in df_feat.columns:
        df_pca["period"] = df_feat["period"].values
    df_pca["MCI"] = df_feat["MCI"].values

    for i in range(min(6, scores.shape[1])):
        df_pca[f"PC{i+1}"] = scores[:, i]

    print("\n[pca] Explained Variance:")
    cumvar = 0.0
    for i, ev in enumerate(pca.explained_variance_ratio_[:6]):
        cumvar += ev
        print(f"  PC{i+1}: {ev*100:.1f}%  (cumulative: {cumvar*100:.1f}%)")

    return pca, df_pca, feat_cols, scaler


# =============================================================================
# PLOT 1 — SCREE PLOT
# =============================================================================

def plot_scree(pca, save: bool = True) -> None:
    """Bar + cumulative line chart of explained variance per PC."""
    n   = min(10, len(pca.explained_variance_ratio_))
    evr = pca.explained_variance_ratio_[:n]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, n+1), evr * 100,
           color="#457B9D", alpha=0.8, label="Individual")
    ax.plot(range(1, n+1), np.cumsum(evr) * 100,
            "o-", color="#E63946", label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("Scree Plot — Sanskrit Morphological Feature Space")
    ax.legend()
    ax.set_xticks(range(1, n+1))
    plt.tight_layout()

    if save:
        _savefig("01_scree_plot.png")
    plt.show()


# =============================================================================
# PLOT 2 — PCA BIPLOT
# =============================================================================

def plot_pca_biplot(pca, df_pca: pd.DataFrame,
                   feat_cols: list, save: bool = True) -> None:
    """
    PC1 vs PC2 scatter coloured by genre.

    Fixes applied (v2):
      1. Loading arrow labels use adaptive offset to prevent overlap —
         labels are pushed radially outward and nudged apart when close.
      2. Vedic Treebank outliers (PC2 > 4.0) are marked with a distinct
         marker (star) and annotated to flag the artefactual syntactic signal.
      3. Shastra ellipse is omitted (genre too internally diverse to be
         meaningful as a single ellipse); Shastra points shown with lower
         alpha and smaller size to reduce visual dominance.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # ── Scatter points ────────────────────────────────────────────────────
    for genre, grp in df_pca.groupby("genre"):
        col   = config.GENRE_PALETTE.get(genre, "#888888")
        alpha = 0.40 if genre == "Shastra" else 0.75
        size  = 50   if genre == "Shastra" else 80

        # Separate Vedic Treebank outliers (PC2 > 4) from rest
        if genre == "Vedic":
            normal   = grp[grp["PC2"] <= 4.0]
            treebank = grp[grp["PC2"] >  4.0]
            ax.scatter(normal["PC1"], normal["PC2"],
                       color=col, alpha=alpha, s=size, label=genre,
                       edgecolors="white", linewidths=0.5, zorder=3)
            if len(treebank) > 0:
                ax.scatter(treebank["PC1"], treebank["PC2"],
                           color=col, alpha=0.9, s=120, marker="*",
                           edgecolors="#333333", linewidths=0.6, zorder=4,
                           label="Vedic (Treebank)")
                # Annotate cluster
                ax.annotate(
                    "Vedic Treebank\n(syntactic features\nonly available here)",
                    xy=(treebank["PC1"].mean(), treebank["PC2"].mean()),
                    xytext=(treebank["PC1"].mean() + 2.5,
                            treebank["PC2"].mean() - 1.0),
                    fontsize=7.5, color="#2A9D8F",
                    arrowprops=dict(arrowstyle="->", color="#2A9D8F",
                                    lw=1.0, connectionstyle="arc3,rad=0.2"),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#2A9D8F", alpha=0.85))
        else:
            ax.scatter(grp["PC1"], grp["PC2"],
                       color=col, alpha=alpha, s=size, label=genre,
                       edgecolors="white", linewidths=0.5, zorder=3)

    # ── 95% confidence ellipses (all genres except Shastra) ──────────────
    for genre, grp in df_pca.groupby("genre"):
        if genre == "Shastra":
            continue          # Fix 3: skip Shastra ellipse
        if len(grp) < 3:
            continue
        # Use non-Treebank Vedic points only for the Vedic ellipse
        if genre == "Vedic":
            grp = grp[grp["PC2"] <= 4.0]
        if len(grp) < 3:
            continue

        col  = config.GENRE_PALETTE.get(genre, "#888888")
        cov  = np.cov(grp[["PC1", "PC2"]].values.T)
        mean = grp[["PC1", "PC2"]].mean().values
        vals, vecs = np.linalg.eigh(cov)
        idx  = vals.argsort()[::-1]
        vals, vecs = vals[idx], vecs[:, idx]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w = 2 * 2.0 * np.sqrt(abs(vals[0]))
        h = 2 * 2.0 * np.sqrt(abs(vals[1]))
        ell = mpatches.Ellipse(
            mean, w, h, angle=angle,
            edgecolor=col, facecolor=col,
            alpha=0.10, linewidth=2, linestyle="--")
        ax.add_patch(ell)

    # Add a note about Shastra ellipse being omitted
    ax.text(0.01, 0.01,
            "Note: Shastra ellipse omitted\n(high internal genre diversity)",
            transform=ax.transAxes, fontsize=7.5,
            color=config.GENRE_PALETTE["Shastra"],
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # ── Loading arrows with non-overlapping labels (Fix 1) ───────────────
    loadings   = pca.components_[:2].T
    magnitudes = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
    top_idx    = magnitudes.argsort()[-6:][::-1]
    scale      = 2.0   # slightly shorter arrows to leave room for labels

    arrow_coords = []  # track (lx, ly, label_x, label_y) for overlap check
    for idx in top_idx:
        lx, ly = loadings[idx]
        # Base label position: radially outward from arrow tip
        label_x = lx * scale * 1.30
        label_y = ly * scale * 1.30

        # Nudge if too close to a previously placed label (> 0.6 unit apart)
        for _, _, prev_lx, prev_ly in arrow_coords:
            dist = np.sqrt((label_x - prev_lx)**2 + (label_y - prev_ly)**2)
            if dist < 0.80:
                # Push perpendicular to the arrow direction
                perp_x = -ly / max(magnitudes[idx], 1e-9)
                perp_y =  lx / max(magnitudes[idx], 1e-9)
                label_x += perp_x * 0.55
                label_y += perp_y * 0.55

        lbl = (feat_cols[idx]
               .replace("pos_",    "POS:")
               .replace("case_",   "Case:")
               .replace("tense_",  "T:")
               .replace("mood_",   "M:")
               .replace("avg_dep_length",    "AvgDepLen")
               .replace("nom_verb_ratio",    "NomVerbR")
               .replace("subordination_rate","SubordR")
               .replace("verb_final_rate",   "VerbFinal"))

        ax.annotate("", xy=(lx*scale, ly*scale), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>",
                                   color="#111111", lw=1.8))
        ax.text(label_x, label_y, lbl,
                fontsize=8.5, ha="center", va="center",
                fontweight="bold", color="#111111",
                bbox=dict(boxstyle="round,pad=0.25",
                          fc="white", ec="#cccccc", alpha=0.90))
        arrow_coords.append((lx, ly, label_x, label_y))

    # ── Axes and titles ───────────────────────────────────────────────────
    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.axvline(0, color="grey", lw=0.5, ls=":")
    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var) "
        f"— Nominalization axis",
        fontsize=11)
    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var) "
        f"— Syntactic complexity axis",
        fontsize=11)
    ax.set_title(
        "PCA Biplot: Sanskrit Morphological Space by Genre\n"
        "(95% ellipses | top-6 feature loadings | ★ = Treebank chapters)",
        fontsize=11)
    ax.legend(title="Genre", loc="upper right", fontsize=9,
              framealpha=0.9)
    plt.tight_layout()

    if save:
        _savefig("02_pca_biplot.png")
    plt.show()


# =============================================================================
# PLOT 3 — LOADING HEATMAP
# =============================================================================

def plot_loading_heatmap(pca, feat_cols: list,
                         n_pc: int = 4, top_n: int = 12,
                         save: bool = True) -> None:
    """
    Heatmap of the top_n features by maximum absolute loading
    across the first n_pc principal components.
    """
    n_pc     = min(n_pc, pca.n_components_)
    loadings = pd.DataFrame(
        pca.components_[:n_pc].T,
        index=feat_cols,
        columns=[f"PC{i+1}" for i in range(n_pc)]
    )
    top_feats = loadings.abs().max(axis=1).nlargest(top_n).index

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(
        loadings.loc[top_feats],
        annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax,
        vmin=-1, vmax=1, annot_kws={"size": 8})
    ax.set_title(f"Top {top_n} Feature Loadings (PC1–PC{n_pc})",
                 fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    plt.tight_layout()

    if save:
        _savefig("03_loading_heatmap.png")
    plt.show()


# =============================================================================
# PLOT 4 — MCI BOXPLOT
# =============================================================================

def plot_mci_by_genre(df_feat: pd.DataFrame, save: bool = True) -> None:
    """
    Boxplot of Morphological Complexity Index (MCI) by genre.
    Ordered by median MCI descending.

    Improvements (v2):
      - Strip plot overlay shows individual text points
      - Shastra low outliers annotated with example text names
    """
    present = [g for g in config.GENRES if g in df_feat["genre"].values]
    order   = sorted(
        present,
        key=lambda g: df_feat[df_feat["genre"] == g]["MCI"].median(),
        reverse=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Boxplot
    bp = ax.boxplot(
        [df_feat[df_feat["genre"] == g]["MCI"].values for g in order],
        patch_artist=True, labels=order,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker="o", markersize=4,
                        markerfacecolor="white", markeredgewidth=0.8),
        widths=0.5)
    for patch, genre in zip(bp["boxes"], order):
        patch.set_facecolor(config.GENRE_PALETTE.get(genre, "#aaaaaa"))
        patch.set_alpha(0.65)

    # Strip plot overlay — individual texts as semi-transparent dots
    np.random.seed(config.RANDOM_SEED)
    for i, genre in enumerate(order, 1):
        vals  = df_feat[df_feat["genre"] == genre]["MCI"].values
        jitter = np.random.uniform(-0.18, 0.18, len(vals))
        col    = config.GENRE_PALETTE.get(genre, "#888")
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=col, alpha=0.35, s=18, zorder=2,
                   edgecolors="none")

    # Annotate the extreme Shastra low outliers (MCI < 2.3)
    if "Shastra" in order:
        shastra_df = df_feat[df_feat["genre"] == "Shastra"]
        extreme    = shastra_df[shastra_df["MCI"] < 2.3].nsmallest(3, "MCI")
        sha_pos    = order.index("Shastra") + 1
        for _, row in extreme.iterrows():
            short_name = row["text_name"][:20] + "…" \
                         if len(row["text_name"]) > 20 else row["text_name"]
            ax.annotate(
                f"{short_name}\n(MCI={row['MCI']:.2f})",
                xy=(sha_pos, row["MCI"]),
                xytext=(sha_pos + 0.55, row["MCI"] + 0.05),
                fontsize=6.5, color="#457B9D",
                arrowprops=dict(arrowstyle="->", color="#457B9D",
                                lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", ec="#457B9D", alpha=0.85))

    ax.set_xlabel("Genre", fontsize=11)
    ax.set_ylabel("MCI (bits)", fontsize=11)
    ax.set_title(
        "Morphological Complexity Index by Sanskrit Genre\n"
        "(dots = individual texts; ○ = outliers)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        _savefig("04_mci_boxplot.png")
    plt.show()


# =============================================================================
# TEXT SUMMARY
# =============================================================================

def print_pca_summary(pca, feat_cols: list, n_pc: int = 3) -> None:
    """
    Print the top 5 positive and negative loading features for each PC,
    giving a linguistic interpretation prompt.
    """
    n_pc = min(n_pc, pca.n_components_)
    print("\n[pca] Loading interpretation summary:")
    for i in range(n_pc):
        loadings = pd.Series(pca.components_[i], index=feat_cols)
        top_pos  = loadings.nlargest(5)
        top_neg  = loadings.nsmallest(5)
        print(f"\n  PC{i+1} "
              f"({pca.explained_variance_ratio_[i]*100:.1f}% variance):")
        print(f"    High scores (→ right/up): "
              + ", ".join(f"{k}={v:.2f}" for k, v in top_pos.items()))
        print(f"    Low scores  (← left/down): "
              + ", ".join(f"{k}={v:.2f}" for k, v in top_neg.items()))


# =============================================================================
# HELPER
# =============================================================================

def _savefig(filename: str) -> None:
    """Save current figure to config.OUTPUT_DIR."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[pca] Saved → {path}")