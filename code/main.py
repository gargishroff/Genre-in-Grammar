"""
main.py
=======
Orchestrates the full Day 1 + Day 2 pipeline for the Sanskrit
Morphological Complexity project.

Usage
-----
  python main.py                  # run full pipeline (Day 1 + Day 2)
  python main.py --day 1          # Day 1 only: load data + save features
  python main.py --day 2          # Day 2 only: load saved features + PCA
  python main.py --day 1 --save   # Day 1: load + save feature CSV
  python main.py --day 2 --load   # Day 2: skip parsing, load saved CSV

Pipeline
--------
  Day 1:
    1. Load / generate corpus         (parser.py)
    2. Extract feature matrix         (features.py)
    3. Save feature matrix to CSV     (features.py)
    4. Print summary statistics

  Day 2:
    5. Load feature matrix            (features.py)
    6. Fit PCA                        (pca_analysis.py)
    7. Plot scree, biplot, heatmap    (pca_analysis.py)
    8. Plot MCI boxplot               (pca_analysis.py)
    9. Print loading interpretation   (pca_analysis.py)

Edit config.py to switch between real DCS data and synthetic demo.
"""

import os
import sys
import argparse
import pandas as pd

import config
import parser          as par
import features        as feat
import pca_analysis    as pca_mod
import clustering      as clust_mod
import classification  as clf_mod
import correlations    as corr_mod
import qualitative_analysis as qual_mod

def extract_features(save_csv: bool = True):
    """
    Load the corpus, extract features, optionally save to CSV.

    Returns
    -------
    df_feat : pd.DataFrame
        Text-level feature matrix.
    """
    print("=" * 65)
    print("Data Loading & Feature Extraction")
    print("=" * 65)

    # Step 1: Load corpus
    if config.DATA_SOURCE == "conllu":
        df_tokens = par.load_conllu_corpus(config.CONLLU_ROOT)
    else:
        df_tokens = par.generate_synthetic_corpus(n_texts_per_genre=8)

    # Step 2: Feature extraction
    df_feat = feat.extract_features(df_tokens)

    # Step 3: Save feature matrix
    if save_csv:
        feat.save_features(df_feat)

    # Step 4: Summary statistics
    _print_summary(df_feat)

    return df_feat


def _print_summary(df_feat: pd.DataFrame) -> None:
    """Summary table of mean MCI and compound rate by genre."""
    print("\n" + "=" * 65)
    print("SUMMARY — Mean Features by Genre")
    print("=" * 65)
    cols = ["MCI", "compound_rate", "mantra_rate",
            "pos_NOUN", "pos_VERB", "nom_verb_ratio"]
    present = [c for c in cols if c in df_feat.columns]
    summary = (df_feat.groupby("genre")[present]
               .agg(["mean", "std"]).round(3))
    print(summary.to_string())
    print()


def run_pca(df_feat: pd.DataFrame = None, load_csv: bool = False):
    """
    Run PCA and generate visualizations.

    Parameters
    ----------
    df_feat  : pre-loaded feature DataFrame (from extract_features), or None
    load_csv : if True, reload from saved CSV instead of using df_feat
    """
    print("=" * 65)
    print("PCA & Visualizations")
    print("=" * 65)

    # Load features if not passed in
    if load_csv or df_feat is None:
        df_feat = feat.load_features()

    # Ensure output directory exists
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Step 5: Fit PCA
    pca, df_pca, feat_cols, scaler = pca_mod.run_pca(df_feat)

    # Step 6–9: Visualizations
    pca_mod.plot_scree(pca)
    pca_mod.plot_pca_biplot(pca, df_pca, feat_cols)
    pca_mod.plot_loading_heatmap(pca, feat_cols, n_pc=4, top_n=12)
    pca_mod.plot_mci_by_genre(df_feat)
    pca_mod.print_pca_summary(pca, feat_cols, n_pc=3)

    print("\n" + "=" * 65)
    print(f"Figures saved to: '{config.OUTPUT_DIR}/'")
    print("=" * 65)

    return pca, df_pca, feat_cols


def clustering_classification(df_pca: pd.DataFrame = None, pca=None,
             feat_cols: list = None, df_feat: pd.DataFrame = None):
    """
    Run clustering (k-means + hierarchical) and classification
    (SVM, Logistic Regression, LDA) on PCA scores.

    Parameters
    ----------
    df_pca    : PCA scores DataFrame (output of pca_mod.run_pca)
    pca       : fitted PCA object
    feat_cols : feature column names
    df_feat   : feature matrix (needed to reload PCA if not passed)
    """
    print("=" * 65)
    print("Clustering & Classification Benchmarking")
    print("=" * 65)

    # Reload from CSV if PCA results not passed in
    if df_pca is None:
        df_feat = feat.load_features()
        pca, df_pca, feat_cols, _ = pca_mod.run_pca(df_feat)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ── Clustering ──
    df_pca, km, ari, label_map = clust_mod.run_kmeans(df_pca)
    clust_mod.plot_kmeans(df_pca, ari, label_map)

    Z, labels = clust_mod.run_hierarchical(df_pca)
    clust_mod.plot_dendrogram(Z, labels, df_pca)
    clust_mod.plot_cluster_genre_heatmap(df_pca, label_map)
    clust_mod.print_clustering_summary(df_pca, label_map)

    # ── Classification ──
    pc_cols = [c for c in df_pca.columns if c.startswith("PC")]
    X = df_pca[pc_cols].values
    y = df_pca["genre"].values

    results_df, pred_dict, y, X = clf_mod.run_all_classifiers(df_pca)

    # Accuracy + Macro F1 bar chart
    clf_mod.plot_classifier_comparison(
        results_df,
        title_suffix="Class Weights (balanced)")

    # Confusion matrices
    clf_mod.plot_confusion_matrices(
        pred_dict, y,
        title="Class Weights (balanced)")

    # LDA projection
    clf_mod.plot_lda_projection(df_pca, X, y)

    # Detailed per-genre report
    clf_mod.print_classification_report(
        pred_dict, y, label="Class Weights (balanced)")

    print("\n" + "=" * 65)
    print(f"Figures saved to: '{config.OUTPUT_DIR}/'")
    print("=" * 65)

    return results_df, pred_dict, df_pca


def diachronic_correlation(df_feat: pd.DataFrame = None,
             df_pca:  pd.DataFrame = None,
             pred_dict: dict = None,
             y_true = None):
    """
    Spearman diachronic correlations + qualitative boundary analysis.

    Parameters
    ----------
    df_feat   : feature matrix (from Day 1)
    df_pca    : PCA scores (from Day 2)
    pred_dict : classifier OOF predictions (from Day 3)
    y_true    : true genre labels (from Day 3)
    """
    print("=" * 65)
    print("Correlations & Qualitative Analysis")
    print("=" * 65)

    # Reload if not passed in
    if df_feat is None:
        df_feat = feat.load_features()
    if df_pca is None:
        pca, df_pca, feat_cols, _ = pca_mod.run_pca(df_feat)
    if pred_dict is None or y_true is None:
        print("[main] pred_dict not available.")
        pred_dict = {}
        y_true    = df_feat["genre"].values

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ── Correlations ──
    df_feat_p = corr_mod.assign_periods(df_feat)
    corr_mod.run_spearman(df_feat_p)
    corr_mod.plot_diachronic_scatter(df_feat_p)
    corr_mod.plot_feature_period_grid(df_feat_p)
    corr_mod.plot_genre_period_box(df_feat_p)

    # ── Qualitative analysis ──
    df_feat_aligned = df_feat.copy().reset_index(drop=True)
    df_pca_aligned  = df_pca.copy().reset_index(drop=True)

    df_boundary = qual_mod.find_boundary_texts(
        df_feat_aligned, df_pca_aligned, pred_dict, y_true)

    if not df_boundary.empty:
        qual_mod.plot_boundary_texts(df_pca_aligned, df_boundary)
        qual_mod.plot_feature_profiles(df_feat_aligned, df_boundary)
        qual_mod.print_qualitative_report(df_feat_aligned, df_boundary)

    print("\n" + "=" * 65)
    print(f"Figures saved to: '{config.OUTPUT_DIR}/'")
    print("=" * 65)

    return df_feat_p, df_boundary


def parse_args():
    p = argparse.ArgumentParser(
        description="Sanskrit Morphological Complexity — PCA Pipeline")
    p.add_argument(
        "--save", action="store_true",
        help="Save feature matrix to CSV after Day 1.")
    p.add_argument(
        "--load", action="store_true",
        help="Load feature matrix from saved CSV (skip re-parsing).")
    return p.parse_args()


def main():
    args = parse_args()

    df_feat = extract_features(save_csv=True)
    pca, df_pca, feat_cols = run_pca(df_feat=df_feat)
    results_df, pred_dict, df_pca = clustering_classification(
        df_pca=df_pca, pca=pca,
        feat_cols=feat_cols, df_feat=df_feat)
    diachronic_correlation(df_feat=df_feat, df_pca=df_pca,
                pred_dict=pred_dict,
                y_true=df_feat["genre"].values)


if __name__ == "__main__":
    main()