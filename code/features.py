"""
features.py
===========
Transforms the token-level DataFrame (output of parser.py) into a
text-level feature matrix ready for PCA and downstream analysis.

Responsibilities:
  - _flatten_feats()    : pre-expand the feats dict column into flat string
                          columns (vectorized, avoids slow row-by-row apply)
  - extract_features()  : aggregate tokens → one feature vector per text
  - get_feature_columns(): return the list of numeric feature column names
  - save_features()     : save the feature DataFrame to CSV
  - load_features()     : reload a previously saved feature CSV

Feature vector (30 features per text, all rates per 1000 tokens):
  pos_{NOUN,VERB,ADJ,ADP,PRON,NUM,ADV,PART}   (8)  — POS tag rates
  nom_verb_ratio                               (1)  — nominal/verbal balance
  case_{Nom,Acc,Inst,Dat,Abl,Gen,Loc,Voc}      (8)  — case rates (nominals)
  tense_{Pres,Past,Fut,Imp}                    (4)  — tense rates (verbals)
  mood_{Ind,Opt,Imp,Sub}                       (4)  — mood rates (verbals)
  compound_rate                                (1)  — compound token rate
  mantra_rate                                  (1)  — IsMantra token rate
  avg_dep_length                               (1)  — mean dep arc length*
  subordination_rate                           (1)  — clausal deprel rate*
  verb_final_rate                              (1)  — SOV verb-final rate*
  MCI                                          (1)  — morphological entropy

  * Syntactic features are 0 for non-Treebank chapters.
"""

import os
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import entropy

import config


# =============================================================================
# PRE-EXPANSION (vectorized for speed on 5M+ tokens)
# =============================================================================

def _flatten_feats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-expand the feats dict column into flat string columns.
    Avoids slow row-by-row .apply() during feature extraction.

    Adds columns: feat_Case, feat_Tense, feat_Mood, feat_Gender,
                  feat_Number, feat_VerbForm, feat_Voice, feat_all.
    """
    print("[features] Pre-expanding morphological features...")
    df = df.copy()
    for key in ["Case","Tense","Mood","Gender","Number","VerbForm","Voice"]:
        df[f"feat_{key}"] = df["feats"].map(
            lambda f, k=key: f.get(k, "") if isinstance(f, dict) else "")
    # All feature values joined for MCI entropy calculation
    df["feat_all"] = df["feats"].map(
        lambda f: "|".join(f.values()) if isinstance(f, dict) else "")
    return df


# =============================================================================
# MAIN FEATURE EXTRACTION
# =============================================================================

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate token-level DataFrame → text-level feature matrix.

    Parameters
    ----------
    df : DataFrame
        Output of parser.load_conllu_corpus() or
        parser.generate_synthetic_corpus().

    Returns
    -------
    df_feat : DataFrame
        One row per text. Columns: text_name, genre, [period],
        all feature columns, MCI.
    """
    df = _flatten_feats(df)

    group_cols = (["text_name", "genre", "period"]
                  if "period" in df.columns else ["text_name", "genre"])

    rows     = []
    n_texts  = df["text_name"].nunique()
    print(f"[features] Extracting features for {n_texts} texts...")

    for i, (keys, grp) in enumerate(df.groupby(group_cols), 1):
        if i % 50 == 0 or i == n_texts:
            print(f"  ... {i}/{n_texts} texts processed")

        keys = keys if isinstance(keys, tuple) else (keys,)
        feat = dict(zip(group_cols, keys))
        n    = len(grp)

        # ── POS rates ────────────────────────────────────────────────────
        pos_vc = grp["upos"].value_counts()
        for pos in config.POS_TAGS:
            feat[f"pos_{pos}"] = pos_vc.get(pos, 0) / n * 1000

        n_nom = sum(pos_vc.get(p, 0) for p in ["NOUN", "ADJ", "PRON"])
        n_vb  = pos_vc.get("VERB", 0)
        feat["nom_verb_ratio"] = n_nom / max(n_vb, 1)

        # ── Case rates (nominal tokens) ───────────────────────────────────
        nominal  = grp[grp["upos"].isin(["NOUN", "ADJ", "PRON"])]
        n_nom_t  = max(len(nominal), 1)
        case_vc  = nominal["feat_Case"].value_counts()
        for case in config.CASE_TAGS:
            feat[f"case_{case}"] = case_vc.get(case, 0) / n_nom_t * 1000

        # ── Tense and Mood rates (verbal tokens) ──────────────────────────
        verbal   = grp[grp["upos"] == "VERB"]
        n_vb_t   = max(len(verbal), 1)
        tense_vc = verbal["feat_Tense"].value_counts()
        mood_vc  = verbal["feat_Mood"].value_counts()
        for tense in config.TENSE_TAGS:
            feat[f"tense_{tense}"] = tense_vc.get(tense, 0) / n_vb_t * 1000
        for mood in config.MOOD_TAGS:
            feat[f"mood_{mood}"]   = mood_vc.get(mood, 0)  / n_vb_t * 1000

        # ── Compound rate ─────────────────────────────────────────────────
        # Real DCS: is_compound flag set by parser when Case=Cpd was present
        # Synthetic: is_compound also set directly in generate_synthetic_corpus
        cpd = (grp["is_compound"].sum()
               if "is_compound" in grp.columns
               else (grp["feat_Case"] == "Cpd").sum())
        feat["compound_rate"] = cpd / n * 1000

        # ── Mantra rate ───────────────────────────────────────────────────
        feat["mantra_rate"] = (
            grp["is_mantra"].sum() / n * 1000
            if "is_mantra" in grp.columns else 0.0)

        # ── Syntactic features (Treebank chapters only) ───────────────────
        if "has_dep" in grp.columns and grp["has_dep"].any():
            dep = grp[grp["head"] != "_"].copy().reset_index(drop=True)
            dep["head_int"] = pd.to_numeric(dep["head"], errors="coerce")
            dep["pos_int"]  = dep.index + 1

            # Average dependency arc length
            feat["avg_dep_length"] = (
                dep["head_int"] - dep["pos_int"]).abs().mean()

            # Subordination rate: clausal dependency relations per 1000 tokens
            clausal = {"advcl", "acl", "ccomp", "xcomp", "relcl"}
            feat["subordination_rate"] = (
                dep["deprel"].str.lower().isin(clausal).sum()
                / max(len(dep), 1) * 1000)

            # Verb-final tendency: VERB tokens whose head comes after them
            vb_dep = dep[dep["upos"] == "VERB"]
            feat["verb_final_rate"] = (
                (vb_dep["head_int"] > vb_dep["pos_int"]).sum()
                / max(len(vb_dep), 1) * 1000)
        else:
            feat["avg_dep_length"]     = 0.0
            feat["subordination_rate"] = 0.0
            feat["verb_final_rate"]    = 0.0

        # ── Morphological Complexity Index (Shannon entropy) ───────────────
        all_vals = grp["feat_all"].str.split("|").explode()
        all_vals = all_vals[all_vals != ""]
        if len(all_vals) > 0:
            counts = all_vals.value_counts().values
            feat["MCI"] = float(entropy(counts / counts.sum(), base=2))
        else:
            feat["MCI"] = 0.0

        rows.append(feat)

    df_feat = pd.DataFrame(rows).fillna(0)
    feat_cols = get_feature_columns(df_feat)
    print(f"[features] Feature matrix: {df_feat.shape[0]} texts "
          f"× {len(feat_cols)} features.")
    return df_feat


# =============================================================================
# HELPERS
# =============================================================================

def get_feature_columns(df_feat: pd.DataFrame) -> list:
    """Return only the numeric feature column names (excludes metadata)."""
    exclude = {"text_name", "genre", "period", "MCI"}
    return [c for c in df_feat.columns if c not in exclude]


def save_features(df_feat: pd.DataFrame, path: str = None) -> str:
    """
    Save the feature DataFrame to a CSV file.
    Returns the path where it was saved.
    """
    if path is None:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        path = os.path.join(config.OUTPUT_DIR, "feature_matrix.csv")
    df_feat.to_csv(path, index=False, encoding="utf-8")
    print(f"[features] Feature matrix saved → {path}")
    return path


def load_features(path: str = None) -> pd.DataFrame:
    """
    Load a previously saved feature CSV.
    The feats column (dict) is not stored — only numeric features are saved.
    """
    if path is None:
        path = os.path.join(config.OUTPUT_DIR, "feature_matrix.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Feature matrix not found at '{path}'.\n"
            "Run extract_features() first and save with save_features()."
        )
    df_feat = pd.read_csv(path, encoding="utf-8")
    print(f"[features] Loaded feature matrix: "
          f"{df_feat.shape[0]} texts × {df_feat.shape[1]} columns "
          f"from '{path}'.")
    return df_feat