"""
classification.py
=================
Supervised Classification Benchmarking
With class imbalance handling via four strategies.

Strategies
----------
1. BASELINE      : unweighted classifiers (original)
2. CLASS WEIGHTS : class_weight='balanced' on SVM and LR
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     cross_val_predict)
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, accuracy_score)

import config


def _get_cv(y, n_splits=5):
    k = min(n_splits, pd.Series(y).value_counts().min())
    return StratifiedKFold(n_splits=k, shuffle=True,
                           random_state=config.RANDOM_SEED)


def _make_classifiers(class_weight=None):
    lda_kw = {"priors": None} if class_weight == "balanced" else {}
    return {
        "SVM (linear)"       : SVC(kernel="linear", C=1.0,
                                    class_weight=class_weight,
                                    probability=True,
                                    random_state=config.RANDOM_SEED),
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0,
                                    class_weight=class_weight,
                                    solver="lbfgs",
                                    random_state=config.RANDOM_SEED),
        "LDA"                : LinearDiscriminantAnalysis(**lda_kw),
    }


def _run_classifiers(X, y, classifiers, label):
    cv = _get_cv(y)
    results, preds = [], {}
    print(f"\n[classification] {label}  ({cv.n_splits}-fold CV, n={len(y)})")
    dist = ", ".join(f"{g}={n}" for g,n in
                     sorted(pd.Series(y).value_counts().items()))
    print(f"  Classes: {dist}")
    print(f"  {'Classifier':<22} {'Acc':>7} {'±':>4} {'F1':>7} {'±':>4}")
    print("  " + "─"*50)

    for name, clf in classifiers.items():
        acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        f1  = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")
        yp  = cross_val_predict(clf, X, y, cv=cv)
        preds[name] = yp
        results.append({"classifier": name,
                         "cv_accuracy_mean": acc.mean(),
                         "cv_accuracy_std" : acc.std(),
                         "cv_f1_mean"      : f1.mean(),
                         "cv_f1_std"       : f1.std()})
        print(f"  {name:<22} {acc.mean()*100:>6.1f}%  "
              f"{acc.std()*100:>4.1f}%  "
              f"{f1.mean()*100:>6.1f}%  {f1.std()*100:>4.1f}%")

    best = max(results, key=lambda r: r["cv_f1_mean"])
    print(f"  Best F1: {best['classifier']} ({best['cv_f1_mean']*100:.1f}%)")
    return pd.DataFrame(results), preds


def _get_XY(df_pca):
    pc = [c for c in df_pca.columns if c.startswith("PC")]
    return df_pca[pc].values, df_pca["genre"].values


def run_baseline(df_pca):
    X, y = _get_XY(df_pca)
    return _run_classifiers(X, y, _make_classifiers(), "BASELINE")


def run_class_weights(df_pca):
    """
    class_weight='balanced': weight each class by n_samples/(n_classes*n_i).
    Penalizes misclassifying rare classes more heavily.
    No data modification — changes only the loss function.
    Epic (4 texts) gets weight ~31x higher than Shastra (126 texts).
    """
    X, y = _get_XY(df_pca)
    return _run_classifiers(X, y,
                             _make_classifiers(class_weight="balanced"),
                             "CLASS WEIGHTS (balanced)")


def _smote(X, y, k=5):
    """
    Synthetic Minority Over-sampling Technique (Chawla et al., 2002).
    For each minority-class sample, k nearest neighbours are found.
    New samples are created: synthetic = sample + rand * (neighbour - sample).
    Applied only to training data inside each CV fold to prevent leakage.
    """
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()
    X_new, y_new = [X], [y]

    for cls, cnt in zip(classes, counts):
        if cnt >= target:
            continue
        n_synth = target - cnt
        X_cls   = X[y == cls]
        nn = NearestNeighbors(n_neighbors=min(k, len(X_cls)))
        nn.fit(X_cls)
        nbrs = nn.kneighbors(X_cls, return_distance=False)
        rng  = np.random.RandomState(config.RANDOM_SEED)
        idxs = rng.randint(0, len(X_cls), n_synth)
        synth = [X_cls[i] + rng.random() * (X_cls[rng.choice(nbrs[i])] - X_cls[i])
                 for i in idxs]
        X_new.append(np.array(synth))
        y_new.append(np.full(n_synth, cls))

    Xr = np.vstack(X_new); yr = np.concatenate(y_new)
    perm = np.random.RandomState(config.RANDOM_SEED).permutation(len(Xr))
    return Xr[perm], yr[perm]


def run_smote(df_pca):
    """
    SMOTE applied inside each CV fold (training data only — no leakage).
    Balances all classes to match Shastra (126 texts).
    Epic: 4 → 126 synthetic samples per fold.
    Synthetic samples are PC-space interpolations, NOT real Sanskrit texts.
    """
    X, y = _get_XY(df_pca)
    cv   = _get_cv(y)
    clfs = _make_classifiers()
    target = pd.Series(y).value_counts().max()

    print(f"\n[classification] SMOTE OVERSAMPLING "
          f"({cv.n_splits}-fold, SMOTE inside each fold)")
    print(f"  Original: " + ", ".join(f"{g}={n}" for g,n in
          sorted(pd.Series(y).value_counts().items())))
    print(f"  After SMOTE: all classes → {target} samples (training only)")
    print(f"  CAVEAT: synthetic samples are PC-space interpolations, "
          f"not real texts")
    print(f"  {'Classifier':<22} {'Acc':>7} {'±':>4} {'F1':>7} {'±':>4}")
    print("  " + "─"*50)

    results, preds = [], {n: np.empty(len(y), dtype=object) for n in clfs}

    for name, clf in clfs.items():
        fold_acc, fold_f1 = [], []
        for tr, te in cv.split(X, y):
            Xtr, ytr = _smote(X[tr], y[tr])
            clf.fit(Xtr, ytr)
            yp = clf.predict(X[te])
            preds[name][te] = yp
            fold_acc.append(accuracy_score(y[te], yp))
            fold_f1.append(f1_score(y[te], yp, average="macro",
                                    zero_division=0))
        am, astd = np.mean(fold_acc), np.std(fold_acc)
        fm, fstd = np.mean(fold_f1),  np.std(fold_f1)
        results.append({"classifier": name,
                         "cv_accuracy_mean": am, "cv_accuracy_std": astd,
                         "cv_f1_mean": fm,       "cv_f1_std": fstd})
        print(f"  {name:<22} {am*100:>6.1f}%  {astd*100:>4.1f}%  "
              f"{fm*100:>6.1f}%  {fstd*100:>4.1f}%")

    best = max(results, key=lambda r: r["cv_f1_mean"])
    print(f"  Best F1: {best['classifier']} ({best['cv_f1_mean']*100:.1f}%)")
    return pd.DataFrame(results), preds


# =============================================================================
# STRATEGY 4 — MERGED GENRES
# =============================================================================

def run_merged_genres(df_pca):
    """
    Merge Epic + Puranic → 'Narrative' (4-class problem).
    Linguistically justified: both share shloka metre, narrative register,
    and similar morphological profiles (confirmed by clustering).
    Uses balanced class weights on the merged classes.
    """
    df_m = df_pca.copy()
    df_m["genre"] = df_m["genre"].replace(
        {"Epic": "Narrative", "Puranic": "Narrative"})
    X  = df_m[[c for c in df_m.columns if c.startswith("PC")]].values
    y  = df_m["genre"].values
    results_df, preds = _run_classifiers(
        X, y, _make_classifiers(class_weight="balanced"),
        "MERGED (Epic+Puranic → Narrative, balanced weights)")
    return results_df, preds, df_m


# =============================================================================
# MAIN — run all strategies
# =============================================================================

def run_all_classifiers(df_pca):
    """
    Run SVM (linear), Logistic Regression, and LDA with
    class_weight='balanced' to handle the Shastra-heavy imbalance.

    Returns: results_df, pred_dict, y, X
    """
    print("\n" + "="*65)
    print("CLASSIFICATION: class_weight='balanced' Strategy")
    print("="*65)
    print("  Shastra (126 texts) dominates unweighted classifiers.")
    print("  Balanced weights: class i gets weight n / (n_classes * n_i).")
    print("  Epic (~4 texts) gets ~31x higher weight than Shastra.")
    print("  No data modification — only the loss function changes.")
    results_df, pred_dict = run_class_weights(df_pca)
    X, y = _get_XY(df_pca)
    return results_df, pred_dict, y, X


# =============================================================================
# PLOT 8 — STRATEGY COMPARISON
# =============================================================================

def plot_strategy_comparison(all_results, save=True):
    """
    Grouped bar chart: Macro F1 for each classifier across all strategies.
    The key question: which strategy most improves minority-class recall?
    """
    strategies = list(all_results.keys())
    clf_names  = list(all_results[strategies[0]]["classifier"])
    x      = np.arange(len(clf_names))
    width  = 0.18
    colors = ["#457B9D", "#2A9D8F", "#E9C46A", "#E63946"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (strat, res) in enumerate(all_results.items()):
        off  = (i - len(strategies)/2 + 0.5) * width
        f1v  = res["cv_f1_mean"].values * 100
        f1e  = res["cv_f1_std"].values  * 100
        bars = ax.bar(x + off, f1v, width, yerr=f1e,
                      label=strat, color=colors[i], alpha=0.85,
                      capsize=4, error_kw={"linewidth": 1.2})
        for bar, val, err in zip(bars, f1v, f1e):
            ax.text(bar.get_x() + bar.get_width()/2.,
                    bar.get_height() + err + 0.8,
                    f"{val:.0f}%", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(clf_names, fontsize=11)
    ax.set_ylabel("Macro F1 (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title(
        "Effect of Class Imbalance Handling on Macro F1\n"
        "(higher = better minority-class recall; error bars = ±1 std CV)",
        fontsize=12, fontweight="bold")
    ax.legend(title="Strategy", fontsize=9, loc="upper right")
    ax.axhline(20, color="red", lw=1.0, ls="--", alpha=0.5)
    ax.text(len(clf_names)-0.4, 21.5, "Chance (20%)",
            color="red", fontsize=8, alpha=0.7)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save: _savefig("08_strategy_comparison.png")
    plt.show()


def plot_classifier_comparison(results_df, title_suffix="Baseline",
                                save=True,
                                filename="08b_classifier_comparison.png"):
    clf_names = results_df["classifier"].tolist()
    x = np.arange(len(clf_names)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ba = ax.bar(x-w/2, results_df["cv_accuracy_mean"]*100, w,
                yerr=results_df["cv_accuracy_std"]*100,
                label="Accuracy", color="#457B9D", alpha=0.85,
                capsize=5, error_kw={"linewidth":1.5})
    bf = ax.bar(x+w/2, results_df["cv_f1_mean"]*100, w,
                yerr=results_df["cv_f1_std"]*100,
                label="Macro F1", color="#E9C46A", alpha=0.85,
                capsize=5, error_kw={"linewidth":1.5})
    for bar in list(ba)+list(bf):
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2., h+1.0,
                f"{h:.1f}%", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(clf_names, fontsize=10)
    ax.set_ylabel("Score (%)"); ax.set_ylim(0, 105)
    ax.set_title(f"Classifier Comparison — {title_suffix}\n"
                 "(Stratified CV | error bars = ±1 std)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.axhline(100/len(config.GENRES), color="red", lw=1.0,
               ls="--", alpha=0.5)
    plt.tight_layout()
    if save: _savefig(filename)
    plt.show()


# =============================================================================
# PLOT 9 — CONFUSION MATRICES
# =============================================================================

def plot_confusion_matrices(pred_dict, y_true,
                             title="Baseline", save=True,
                             filename="09_confusion_matrices.png"):
    genres = sorted(set(y_true)); n = len(pred_dict)
    fig, axes = plt.subplots(1, n, figsize=(5*n+1, 5))
    if n == 1: axes = [axes]
    for ax, (name, yp) in zip(axes, pred_dict.items()):
        cm = confusion_matrix(y_true, yp, labels=genres)
        cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        acc = accuracy_score(y_true, yp)
        f1  = f1_score(y_true, yp, average="macro", zero_division=0)
        sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=genres, yticklabels=genres,
                    linewidths=0.5, ax=ax, vmin=0, vmax=1,
                    annot_kws={"size":9})
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{name}\nAcc={acc*100:.1f}%  F1={f1*100:.1f}%",
                     fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)
    fig.suptitle(f"Confusion Matrices — {title}\n"
                 "Diagonal=correct | Off-diagonal=confused",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save: _savefig(filename)
    plt.show()


# =============================================================================
# PLOT 10 — LDA PROJECTION
# =============================================================================

def plot_lda_projection(df_pca, X, y, save=True):
    lda = LinearDiscriminantAnalysis(); lda.fit(X, y)
    ld  = lda.transform(X)
    df_ld = pd.DataFrame({
        "LD1": ld[:,0],
        "LD2": ld[:,1] if ld.shape[1]>1 else np.zeros(len(ld)),
        "genre": y})
    fig, ax = plt.subplots(figsize=(8, 6))
    for genre, grp in df_ld.groupby("genre"):
        col = config.GENRE_PALETTE.get(genre, "#888")
        ax.scatter(grp["LD1"], grp["LD2"], color=col, alpha=0.70,
                   s=70, label=genre, edgecolors="white",
                   linewidths=0.5, zorder=3)
    for genre, grp in df_ld.groupby("genre"):
        if len(grp) < 3: continue
        col  = config.GENRE_PALETTE.get(genre, "#888")
        cov  = np.cov(grp[["LD1","LD2"]].values.T)
        mean = grp[["LD1","LD2"]].mean().values
        vals, vecs = np.linalg.eigh(cov)
        idx = vals.argsort()[::-1]; vals, vecs = vals[idx], vecs[:,idx]
        angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        ell = mpatches.Ellipse(mean,
                               2*2.0*np.sqrt(abs(vals[0])),
                               2*2.0*np.sqrt(abs(vals[1])),
                               angle=angle, edgecolor=col,
                               facecolor=col, alpha=0.10,
                               linewidth=2, linestyle="--")
        ax.add_patch(ell)
    ax.set_xlabel("LD1 (max separation)", fontsize=11)
    ax.set_ylabel("LD2", fontsize=11)
    ax.set_title("LDA Projection: Maximum Genre Separability\n"
                 "(full data; 95% ellipses)")
    ax.legend(title="Genre", fontsize=9)
    plt.tight_layout()
    if save: _savefig("10_lda_projection.png")
    plt.show()


# =============================================================================
# PRINT REPORT
# =============================================================================

def print_classification_report(pred_dict, y_true, label="Baseline"):
    genres = sorted(set(y_true))
    print(f"\n[classification] Per-genre Report — {label}:")
    for name, yp in pred_dict.items():
        print(f"\n  ── {name} ──")
        for line in classification_report(
                y_true, yp, target_names=genres,
                digits=3, zero_division=0).split("\n"):
            print("    " + line)
    print(f"\n[classification] Most confused pairs — {label}:")
    combined = np.zeros((len(genres), len(genres)))
    for yp in pred_dict.values():
        cm = confusion_matrix(y_true, yp, labels=genres)
        np.fill_diagonal(cm, 0); combined += cm
    combined /= len(pred_dict)
    flat = [(combined[i,j], genres[i], genres[j])
            for i in range(len(genres))
            for j in range(len(genres)) if i!=j]
    flat.sort(reverse=True)
    for cnt, tg, pg in flat[:5]:
        print(f"  {tg:12s} → {pg:12s}  ({cnt:.1f} texts avg)")


# =============================================================================
# HELPER
# =============================================================================

def _savefig(filename):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[classification] Saved → {path}")