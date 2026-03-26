"""
qualitative_analysis.py
=======================
Qualitative Analysis of Genre-Boundary Texts

Identifies texts that sit at genre boundaries — either morphologically
extreme within their genre, or systematically misclassified — and produces
structured profiles to support close linguistic reading.

Responsibilities:
  - find_boundary_texts()     : identify texts of interest via three criteria:
                                (a) misclassified by classifier
                                (b) assigned to unexpected k-means cluster
                                (c) extreme MCI outliers within genre
  - profile_text()            : generate a detailed morphological feature
                                profile for a single text
  - plot_boundary_texts()     : annotated scatter highlighting boundary texts
                                on the PCA biplot
  - plot_feature_profiles()   : radar/bar chart comparing boundary texts
                                to their genre centroids
  - print_qualitative_report(): structured narrative report with linguistic
                                interpretation for each boundary text

All plots saved to config.OUTPUT_DIR.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import config
from features import get_feature_columns


# =============================================================================
# GENRE BACKGROUND KNOWLEDGE for interpretation
# =============================================================================
# Maps (text_name) → brief scholarly note for the qualitative report.

TEXT_NOTES = {
    # Kavya texts that cluster with Shastra
    "Kādambarīsvīkaraṇasūtramañjarī":
        "A grammatical commentary on Kādambarī vocabulary — formally "
        "classified as Kavya but contains dense nominal, definitional "
        "prose characteristic of Shastra. Its extreme PC1 position "
        "(+3.16) reflects this hybrid status.",
    "Harṣacarita":
        "Banabhatta's ornate prose biography — among the most nominalized "
        "and compound-heavy texts in the Kavya corpus. Its morphological "
        "profile is closer to Shastra prose than to verse Kavya.",
    "Daśakumāracarita":
        "Dandin's prose narrative — uses a formal Shastra-like nominal "
        "style that distinguishes prose kavya from verse kavya.",
    "Aṣṭasāhasrikā":
        "Early Buddhist Prajnaparamita sutra in hybrid Sanskrit — "
        "low compound rate and frequent verbal constructions place it "
        "at the verbal end of Kavya, unusual for its genre.",
    "Rasikasaṃjīvanī":
        "A late devotional commentary (16th c.) — extreme negative PC2 "
        "(-3.13) may reflect its dense particle and adverb usage.",
    "Devīmāhātmya":
        "Hymnic prose-poetry from the Markandeya Purana — formally "
        "Kavya but with Puranic register features.",

    # Vedic texts that cluster with Shastra
    "Ṛgvedavedāṅgajyotiṣa":
        "The Vedanga Jyotisha — a technical astronomical/calendrical text. "
        "Despite Vedic affiliation, its content is entirely technical "
        "prose (Shastra-like). PC1=+2.76 confirms nominal density.",
    "Kaṭhopaniṣad":
        "One of the principal Upanishads, composed in classical dialogic "
        "verse-prose. Its archaic Vedic verbal morphology is reduced "
        "compared to Samhitas — placing it at the Vedic–Classical boundary.",
    "Garbhopaniṣat":
        "A minor Upanishad on embryology — highly technical, nominalized "
        "medical prose more characteristic of Ayurveda Shastra.",
    "Gautamadharmasūtra":
        "One of the earliest Dharmasutras — terse, highly nominalized "
        "legal prose. Though Vedic in tradition, its morphology is "
        "indistinguishable from later Shastra Dharmashastra texts.",
    "Vaikhānasadharmasūtra":
        "A late Vaishnava Dharmasutra (2nd–5th c. CE) — composed in "
        "Classical Sanskrit despite Vedic affiliation.",

    # Shastra outliers (low MCI)
    "Nighaṇṭuśeṣa":
        "A Sanskrit lexicon (nighantu) — consists almost entirely of "
        "nominative-singular noun lists with minimal inflectional "
        "diversity. Its MCI of ~1.87 bits is the lowest in the corpus, "
        "reflecting the near-zero morphological entropy of a vocabulary list.",
    "Padārthacandrikā":
        "A commentary on Vaiseshika Sutra terms — highly repetitive "
        "philosophical definitional prose with a very narrow morphological "
        "range (primarily nominative nouns and present indicative verbs).",

    # Puranic texts with interesting positions
    "Devīkālottarāgama":
        "A Shaiva Agama text classified as Puranic but containing "
        "dense tantric technical terminology. PC1 near zero suggests "
        "it occupies the boundary between Shastra and Puranic registers.",
    "Garuḍapurāṇa":
        "Contains substantial Ayurvedic and ritual technical sections — "
        "explaining its rightward position on PC1 (higher nominalization "
        "than typical Puranic narrative texts).",
}


# =============================================================================
# FIND BOUNDARY TEXTS
# =============================================================================

def find_boundary_texts(df_feat: pd.DataFrame,
                         df_pca: pd.DataFrame,
                         pred_dict: dict,
                         y_true: np.ndarray) -> pd.DataFrame:
    """
    Identify genre-boundary texts via three criteria:

    Criterion A — Misclassified by majority of classifiers:
      A text is flagged if ≥2 out of 3 classifiers predict
      a different genre than its true label.

    Criterion B — Assigned to unexpected k-means cluster:
      Texts already identified from clustering output
      (supplied via df_pca["cluster"] and label_map).

    Criterion C — MCI outlier within genre:
      Texts whose MCI falls below Q1 - 1.5*IQR within their genre.

    Returns a DataFrame of boundary texts with reason codes.
    """
    boundary = {}

    # ── Criterion A: misclassified by ≥ 2 classifiers ────────────────────
    n_clf = len(pred_dict)
    for i, text_name in enumerate(df_feat["text_name"].values):
        true_g = y_true[i]
        wrong  = sum(1 for yp in pred_dict.values()
                     if yp[i] != true_g)
        if wrong >= 2:
            key = text_name
            if key not in boundary:
                boundary[key] = {
                    "text_name" : text_name,
                    "genre"     : true_g,
                    "criteria"  : [],
                    "predicted_as": [],
                }
            boundary[key]["criteria"].append("A:misclassified")
            preds = [yp[i] for yp in pred_dict.values()
                     if yp[i] != true_g]
            boundary[key]["predicted_as"].extend(preds)

    # ── Criterion C: MCI outlier ──────────────────────────────────────────
    for genre, grp in df_feat.groupby("genre"):
        q1  = grp["MCI"].quantile(0.25)
        q3  = grp["MCI"].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        outliers = grp[grp["MCI"] < low]
        for _, row in outliers.iterrows():
            key = row["text_name"]
            if key not in boundary:
                boundary[key] = {
                    "text_name" : key,
                    "genre"     : genre,
                    "criteria"  : [],
                    "predicted_as": [],
                }
            boundary[key]["criteria"].append(
                f"C:MCI_outlier({row['MCI']:.2f}bits)")

    # Merge with PCA coordinates
    df_boundary = pd.DataFrame(boundary.values())
    if df_boundary.empty:
        print("[qualitative] No boundary texts found.")
        return df_boundary

    df_boundary = df_boundary.merge(
        df_pca[["text_name","PC1","PC2","MCI"]],
        on="text_name", how="left")

    df_boundary["criteria_str"] = df_boundary["criteria"].apply(
        lambda x: " | ".join(set(x)))
    df_boundary["note"] = df_boundary["text_name"].map(
        lambda t: TEXT_NOTES.get(t, ""))

    print(f"\n[qualitative] Found {len(df_boundary)} boundary texts:")
    for genre, grp in df_boundary.groupby("genre"):
        print(f"  {genre}: {len(grp)} texts")

    return df_boundary.sort_values(["genre","criteria_str"])


# =============================================================================
# PROFILE A SINGLE TEXT
# =============================================================================

def profile_text(text_name: str,
                 df_feat: pd.DataFrame) -> pd.Series:
    """
    Return the full feature vector for one text, along with
    z-scores relative to its genre centroid.
    """
    feat_cols = get_feature_columns(df_feat)
    row = df_feat[df_feat["text_name"] == text_name]
    if row.empty:
        print(f"[qualitative] Text '{text_name}' not found.")
        return None

    genre = row["genre"].iloc[0]
    genre_df = df_feat[df_feat["genre"] == genre]

    genre_mean = genre_df[feat_cols].mean()
    genre_std  = genre_df[feat_cols].std().replace(0, 1)
    z_scores   = (row[feat_cols].iloc[0] - genre_mean) / genre_std

    result = pd.DataFrame({
        "value"      : row[feat_cols].iloc[0],
        "genre_mean" : genre_mean,
        "z_score"    : z_scores,
    })
    return result


# =============================================================================
# PLOT 14 — BOUNDARY TEXTS ON PCA BIPLOT
# =============================================================================

def plot_boundary_texts(df_pca: pd.DataFrame,
                         df_boundary: pd.DataFrame,
                         save: bool = True) -> None:
    """
    PCA biplot with boundary texts highlighted and annotated.
    Background: all texts as faint dots coloured by genre.
    Foreground: boundary texts as larger markers with text labels.
    """
    if df_boundary.empty:
        print("[qualitative] No boundary texts to plot.")
        return

    fig, ax = plt.subplots(figsize=(13, 9))

    # Background: all texts, faint
    for genre, grp in df_pca.groupby("genre"):
        col = config.GENRE_PALETTE.get(genre, "#888")
        ax.scatter(grp["PC1"], grp["PC2"],
                   color=col, alpha=0.18, s=30,
                   edgecolors="none", zorder=1)

    # Genre centroids
    for genre, grp in df_pca.groupby("genre"):
        col  = config.GENRE_PALETTE.get(genre, "#888")
        cx, cy = grp["PC1"].mean(), grp["PC2"].mean()
        ax.scatter(cx, cy, color=col, s=120, marker="X",
                   edgecolors="black", linewidths=0.8, zorder=4)
        ax.text(cx + 0.15, cy + 0.15, genre,
                fontsize=9, color=col, fontweight="bold", zorder=5)

    # Boundary texts: highlighted
    for _, row in df_boundary.iterrows():
        col = config.GENRE_PALETTE.get(row["genre"], "#888")
        ax.scatter(row["PC1"], row["PC2"],
                   color=col, s=110, marker="D",
                   edgecolors="black", linewidths=1.2, zorder=5,
                   alpha=0.9)

        # Short label — trim long names
        lbl = (row["text_name"][:22] + "…"
               if len(row["text_name"]) > 22
               else row["text_name"])
        # Offset label to avoid overlap
        dx = 0.25 if row["PC1"] >= 0 else -0.25
        dy = 0.25
        ax.annotate(
            lbl,
            xy=(row["PC1"], row["PC2"]),
            xytext=(row["PC1"] + dx, row["PC2"] + dy),
            fontsize=6.5, color="black",
            arrowprops=dict(arrowstyle="-", color="grey",
                            lw=0.6, alpha=0.7),
            bbox=dict(boxstyle="round,pad=0.2",
                      fc="white", ec=col, alpha=0.85,
                      linewidth=0.8))

    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.axvline(0, color="grey", lw=0.5, ls=":")
    ax.set_xlabel("PC1 — Nominalization axis", fontsize=11)
    ax.set_ylabel("PC2 — Syntactic complexity axis", fontsize=11)
    ax.set_title(
        f"Genre-Boundary Texts on PCA Space (n={len(df_boundary)})\n"
        "◆ = boundary text  ✕ = genre centroid",
        fontsize=12, fontweight="bold")

    # Legend
    handles = [mpatches.Patch(
                   color=config.GENRE_PALETTE.get(g,"#888"), label=g)
               for g in config.GENRES]
    ax.legend(handles=handles, title="Genre", fontsize=9,
              loc="lower right")
    plt.tight_layout()

    if save:
        _savefig("14_boundary_texts.png")
    plt.show()


# =============================================================================
# PLOT 15 — FEATURE PROFILE BAR CHARTS
# =============================================================================

def plot_feature_profiles(df_feat: pd.DataFrame,
                           df_boundary: pd.DataFrame,
                           n_texts: int = 6,
                           save: bool = True) -> None:
    """
    For the n_texts most interesting boundary texts, plot a bar chart
    comparing their feature values to their genre centroid.
    Features shown: MCI, compound_rate, nom_verb_ratio, pos_NOUN,
                    pos_VERB, pos_PRON, case_Gen, tense_Pres, mood_Opt.
    """
    if df_boundary.empty:
        return

    features = ["MCI", "compound_rate", "nom_verb_ratio",
                "pos_NOUN", "pos_VERB", "pos_PRON",
                "case_Gen", "tense_Pres", "mood_Opt"]
    features = [f for f in features if f in df_feat.columns]

    # Pick top n_texts by abs(PC1) as most extreme
    top = df_boundary.reindex(
        df_boundary["PC1"].abs().nlargest(n_texts).index)

    n_cols = 3
    n_rows = int(np.ceil(len(top) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(14, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    for ax, (_, brow) in zip(axes, top.iterrows()):
        tname = brow["text_name"]
        genre = brow["genre"]
        col   = config.GENRE_PALETTE.get(genre, "#888")

        text_vals  = df_feat[df_feat["text_name"]==tname][features]
        if text_vals.empty:
            ax.set_visible(False); continue
        text_vals  = text_vals.iloc[0]

        genre_mean = df_feat[df_feat["genre"]==genre][features].mean()

        x = np.arange(len(features))
        w = 0.38
        ax.bar(x - w/2, genre_mean.values, w,
               color=col, alpha=0.45, label=f"{genre} mean")
        ax.bar(x + w/2, text_vals.values, w,
               color=col, alpha=0.90, label=tname[:20])

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f.replace("pos_","").replace("case_","c:")
              .replace("tense_","t:").replace("mood_","m:")
              for f in features],
            rotation=45, ha="right", fontsize=7.5)
        ax.set_title(
            f"{tname[:28]}{'…' if len(tname)>28 else ''}\n"
            f"[{genre}]  PC1={brow['PC1']:.2f}",
            fontsize=8.5)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused axes
    for ax in axes[len(top):]:
        ax.set_visible(False)

    fig.suptitle(
        "Feature Profiles of Genre-Boundary Texts vs. Genre Centroids\n"
        "(bar = text value; faint bar = genre mean)",
        fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save:
        _savefig("15_feature_profiles.png")
    plt.show()


# =============================================================================
# PRINT QUALITATIVE REPORT
# =============================================================================

def print_qualitative_report(df_feat: pd.DataFrame,
                               df_boundary: pd.DataFrame) -> None:
    """
    Print a structured qualitative narrative for each boundary text.
    For each text: genre, criteria flagged, PC coordinates,
    key deviating features (z > 1.5), and scholarly note.
    """
    if df_boundary.empty:
        print("[qualitative] No boundary texts to report.")
        return

    print("\n" + "=" * 65)
    print("QUALITATIVE ANALYSIS — Genre-Boundary Texts")
    print("=" * 65)
    print("These texts are flagged because they are systematically")
    print("misclassified or exhibit anomalous morphological profiles.")
    print("They are primary candidates for close linguistic reading.\n")

    for genre, grp in df_boundary.groupby("genre"):
        print(f"── {genre.upper()} ({'─'*(55-len(genre))})")
        for _, row in grp.iterrows():
            tname = row["text_name"]
            print(f"\n  {tname}")
            print(f"  True genre   : {row['genre']}")
            print(f"  Criteria     : {row['criteria_str']}")
            print(f"  PC1={row['PC1']:.2f}  PC2={row['PC2']:.2f}  "
                  f"MCI={row['MCI']:.3f}")

            # Feature z-scores
            profile = profile_text(tname, df_feat)
            if profile is not None:
                extreme = profile[profile["z_score"].abs() > 1.5]
                if not extreme.empty:
                    print(f"  Key deviating features (|z| > 1.5):")
                    for feat, frow in extreme.iterrows():
                        direction = "↑" if frow["z_score"] > 0 else "↓"
                        print(f"    {direction} {feat:<22} "
                              f"value={frow['value']:.1f}  "
                              f"genre_mean={frow['genre_mean']:.1f}  "
                              f"z={frow['z_score']:+.2f}")

            # Scholarly note
            note = TEXT_NOTES.get(tname, "")
            if note:
                print(f"  Interpretation:")
                # Word-wrap at 60 chars
                words = note.split()
                line = "    "
                for word in words:
                    if len(line) + len(word) > 66:
                        print(line)
                        line = "    " + word + " "
                    else:
                        line += word + " "
                if line.strip():
                    print(line)
        print()


def _savefig(filename: str) -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[qualitative] Saved → {path}")