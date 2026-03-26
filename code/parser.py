"""
parser.py
=========
Handles all data ingestion for the Sanskrit Morphological Complexity project.

Responsibilities:
  - parse_misc()        : parse the MISC field (col 10) of a CoNLL-U line
  - parse_feats()       : parse the FEATS field (col 6) of a CoNLL-U line
  - parse_conllu_file() : read one .conllu chapter file → list of token dicts
  - load_conllu_corpus(): walk the DCS 'files/' directory, load all chapters,
                          assign genre labels via GENRE_MAP, return a DataFrame
  - generate_synthetic_corpus(): create a realistic demo corpus when
                          DATA_SOURCE = "synthetic" (no DCS download needed)

Key DCS format notes (from file inspection):
  - Comment lines start with # (single) or ## (double) — both skipped
  - Multiword sandhi spans have IDs like "1-2" — skipped
  - Empty nodes have IDs like "1.1" — skipped
  - Case=Cpd in FEATS flags a compound member, NOT a grammatical case
    → extracted as is_compound=True; "Case" key removed from feats dict
  - HEAD/DEPREL filled only in Vedic Treebank chapters; "_" elsewhere
  - MISC field may contain: LemmaId, OccId, Unsandhied,
    UnsandhiedReconstructed, IsMantra, WordSem
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict

import config


# =============================================================================
# LOW-LEVEL PARSERS
# =============================================================================

def parse_misc(misc_str: str) -> dict:
    """
    Parse column 10 (MISC) of a DCS CoNLL-U line.
    Format: Key=Value|Key=Value|...   or   "_"

    Known DCS keys:
      LemmaId                — integer ID in lookup/dictionary.csv
      OccId                  — occurrence ID of this token
      Unsandhied             — padapatha (unsandhied) form
      UnsandhiedReconstructed— True/False flag
      IsMantra               — True if in Bloomfield's Vedic Concordance
      WordSem                — pipe-separated Sanskrit WordNet sense IDs
    """
    if not misc_str or misc_str == "_":
        return {}
    result = {}
    for part in misc_str.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def parse_feats(feats_str: str) -> dict:
    """
    Parse column 6 (FEATS) of a CoNLL-U line.
    Format: Feature=Value|Feature=Value|...   or   "_"

    Note: Case=Cpd signals a compound member, not a true grammatical case.
    This is handled in parse_conllu_file(), not here.
    """
    if not feats_str or feats_str == "_":
        return {}
    result = {}
    for feat in feats_str.split("|"):
        if "=" in feat:
            k, v = feat.split("=", 1)
            result[k.strip()] = v.strip()
    return result


# =============================================================================
# FILE-LEVEL PARSER
# =============================================================================

def parse_conllu_file(filepath: str, text_name: str, genre: str) -> list:
    """
    Parse one DCS .conllu chapter file → list of token dicts.

    Each token dict contains:
      text_name, genre, chapter, form, lemma, upos, xpos,
      feats (dict, cleaned), is_compound (bool), head, deprel,
      lemma_id, unsandhied, is_mantra, is_reconstructed, has_dep

    Skipped lines:
      - Blank lines (sentence boundaries)
      - # and ## comment lines (metadata)
      - Multiword spans: ID contains "-"  (e.g. "1-2")
      - Empty nodes:    ID contains "."  (e.g. "1.1")
    """
    tokens  = []
    chapter = os.path.basename(filepath)
    has_dep = False

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 10:
                continue

            token_id = parts[0]
            if "-" in token_id or "." in token_id:
                continue

            feats = parse_feats(parts[5])
            misc  = parse_misc(parts[9])

            # Case=Cpd → compound flag, not a real case
            is_compound = feats.pop("Case", None) == "Cpd"

            head   = parts[6]
            deprel = parts[7]
            if head != "_":
                has_dep = True

            tokens.append({
                "text_name"       : text_name,
                "genre"           : genre,
                "chapter"         : chapter,
                "form"            : parts[1],
                "lemma"           : parts[2],
                "upos"            : parts[3],
                "xpos"            : parts[4],
                "feats"           : feats,
                "is_compound"     : is_compound,
                "head"            : head,
                "deprel"          : deprel,
                "lemma_id"        : misc.get("LemmaId"),
                "unsandhied"      : misc.get("Unsandhied"),
                "is_mantra"       : misc.get("IsMantra", "False") == "True",
                "is_reconstructed": misc.get("UnsandhiedReconstructed",
                                             "False") == "True",
            })

    # Propagate has_dep to all tokens in this chapter
    for t in tokens:
        t["has_dep"] = has_dep

    return tokens


# =============================================================================
# CORPUS LOADER
# =============================================================================

def load_conllu_corpus(conllu_root: str) -> pd.DataFrame:
    """
    Walk conllu_root/, load every .conllu file, assign genre from GENRE_MAP.

    Directory structure expected:
      conllu_root/
        TextFolderName/
          TextFolderName-CHAPTERNUM-citation-ID.conllu
          ...

    Returns a DataFrame with one row per token across all loaded texts.
    Prints a warning listing any folders that had no genre mapping.
    """
    if not os.path.isdir(conllu_root):
        raise FileNotFoundError(
            f"DCS directory not found: '{conllu_root}'\n"
            f"Set CONLLU_ROOT in config.py to the 'files/' folder.\n"
            f"Clone from: https://github.com/OliverHellwig/sanskrit"
        )

    text_folders = sorted([
        d for d in os.listdir(conllu_root)
        if os.path.isdir(os.path.join(conllu_root, d))
    ])
    print(f"[parser] Found {len(text_folders)} text folders.")

    all_tokens   = []
    skipped      = set()
    genre_counts = {}

    for folder in text_folders:
        # Exact match first, then partial
        genre = config.GENRE_MAP.get(folder)
        if genre is None:
            for key, val in config.GENRE_MAP.items():
                if (key.lower() in folder.lower() or
                        folder.lower() in key.lower()):
                    genre = val
                    break

        if genre is None:
            skipped.add(folder)
            continue

        # Per-genre cap
        if config.MAX_TEXTS_PER_GENRE is not None:
            genre_counts.setdefault(genre, 0)
            if genre_counts[genre] >= config.MAX_TEXTS_PER_GENRE:
                continue
            genre_counts[genre] += 1

        text_dir = os.path.join(conllu_root, folder)
        for fname in sorted(
                f for f in os.listdir(text_dir) if f.endswith(".conllu")):
            tokens = parse_conllu_file(
                os.path.join(text_dir, fname), folder, genre)
            all_tokens.extend(tokens)

    if not all_tokens:
        raise ValueError(
            "No tokens loaded. Check CONLLU_ROOT and GENRE_MAP in config.py."
        )

    df = pd.DataFrame(all_tokens)
    print(f"[parser] Loaded {len(df):,} tokens | "
          f"{df['text_name'].nunique()} texts | "
          f"{df['genre'].nunique()} genres.")

    if skipped:
        print(f"[parser] WARNING: {len(skipped)} folders had no genre "
              f"mapping and were skipped.")
        print("         Add them to GENRE_MAP in config.py to include them.")
        print("         Unmapped folders:")
        for f in sorted(skipped):
            print(f"           '{f}'")

    return df


# =============================================================================
# SYNTHETIC DATA GENERATOR
# =============================================================================

def generate_synthetic_corpus(n_texts_per_genre: int = 8) -> pd.DataFrame:
    """
    Generate a realistic synthetic DCS-like corpus for demo / testing.
    Used when config.DATA_SOURCE = "synthetic".

    Genre profiles reflect known morphological tendencies:
      Vedic   : high verb rate, optative/subjunctive mood, instrumental heavy
      Epic    : balanced nominal/verbal, nominative-prominent narrative
      Kavya   : highest compound rate, genitive heavy, noun/adjective dense
      Shastra : very high nominal density, locative/ablative heavy
      Puranic : close to Epic, more present-tense narration
    """
    np.random.seed(config.RANDOM_SEED)

    profiles = {
        "Vedic"  : dict(
            up=[0.25, 0.30, 0.15, 0.05, 0.10, 0.02, 0.08, 0.05],
            cp=[0.18, 0.12, 0.18, 0.08, 0.06, 0.12, 0.14, 0.12],
            tp=[0.35, 0.25, 0.20, 0.20], mp=[0.45, 0.25, 0.20, 0.10],
            compound_prob=0.10),
        "Epic"   : dict(
            up=[0.35, 0.22, 0.15, 0.07, 0.08, 0.03, 0.06, 0.04],
            cp=[0.30, 0.18, 0.10, 0.08, 0.08, 0.10, 0.10, 0.06],
            tp=[0.40, 0.35, 0.15, 0.10], mp=[0.70, 0.15, 0.10, 0.05],
            compound_prob=0.15),
        "Kavya"  : dict(
            up=[0.38, 0.15, 0.25, 0.06, 0.06, 0.02, 0.05, 0.03],
            cp=[0.22, 0.15, 0.08, 0.06, 0.05, 0.20, 0.14, 0.10],
            tp=[0.50, 0.20, 0.20, 0.10], mp=[0.75, 0.15, 0.05, 0.05],
            compound_prob=0.40),
        "Shastra": dict(
            up=[0.50, 0.10, 0.18, 0.08, 0.04, 0.03, 0.04, 0.03],
            cp=[0.20, 0.12, 0.08, 0.05, 0.04, 0.08, 0.22, 0.12],
            tp=[0.55, 0.20, 0.15, 0.10], mp=[0.65, 0.20, 0.10, 0.05],
            compound_prob=0.25),
        "Puranic": dict(
            up=[0.33, 0.25, 0.15, 0.07, 0.08, 0.03, 0.06, 0.03],
            cp=[0.28, 0.17, 0.10, 0.08, 0.08, 0.10, 0.12, 0.07],
            tp=[0.45, 0.30, 0.15, 0.10], mp=[0.70, 0.15, 0.10, 0.05],
            compound_prob=0.18),
    }
    periods = {
        "Vedic": -800, "Epic": -200, "Kavya": 400,
        "Shastra": 200, "Puranic": 600,
    }

    records = []
    tid = 1
    for genre, prof in profiles.items():
        up = np.array(prof["up"]); up /= up.sum()
        cp = np.array(prof["cp"]); cp /= cp.sum()
        tp = np.array(prof["tp"]); tp /= tp.sum()
        mp = np.array(prof["mp"]); mp /= mp.sum()

        for _ in range(n_texts_per_genre):
            n     = np.random.randint(800, 1500)
            tname = f"{genre[:3].upper()}_TEXT{tid:03d}"
            text_period = periods[genre] + np.random.randint(-100, 100)

            _up = np.abs(up + np.random.normal(0, 0.01, 8)); _up /= _up.sum()
            _cp = np.abs(cp + np.random.normal(0, 0.01, 8)); _cp /= _cp.sum()
            _tp = np.abs(tp + np.random.normal(0, 0.01, 4)); _tp /= _tp.sum()
            _mp = np.abs(mp + np.random.normal(0, 0.01, 4)); _mp /= _mp.sum()

            for _ in range(n):
                upos  = np.random.choice(config.POS_TAGS, p=_up)
                feats = {}
                is_compound = False

                if upos in ("NOUN", "ADJ", "PRON"):
                    if np.random.random() < prof["compound_prob"]:
                        is_compound = True
                    else:
                        feats["Case"]   = np.random.choice(
                            config.CASE_TAGS, p=_cp)
                    feats["Number"] = np.random.choice(
                        ["Sing","Dual","Plur"], p=[0.60, 0.10, 0.30])
                    feats["Gender"] = np.random.choice(
                        ["Masc","Fem","Neut"], p=[0.45, 0.30, 0.25])

                elif upos == "VERB":
                    feats["Tense"]  = np.random.choice(
                        config.TENSE_TAGS, p=_tp)
                    feats["Mood"]   = np.random.choice(
                        config.MOOD_TAGS, p=_mp)
                    feats["Person"] = np.random.choice(
                        ["1","2","3"], p=[0.20, 0.15, 0.65])
                    feats["Number"] = np.random.choice(
                        ["Sing","Dual","Plur"], p=[0.60, 0.08, 0.32])
                    feats["Voice"]  = np.random.choice(
                        ["Act","Mid","Pass"], p=[0.55, 0.30, 0.15])

                records.append({
                    "text_name"  : tname,
                    "genre"      : genre,
                    "period"     : text_period,
                    "upos"       : upos,
                    "feats"      : feats,
                    "is_compound": is_compound,
                    "is_mantra"  : False,
                    "has_dep"    : False,
                    "head"       : "_",
                    "deprel"     : "_",
                })
            tid += 1

    df = pd.DataFrame(records)
    print(f"[parser] Synthetic corpus: {len(df):,} tokens | "
          f"{df['text_name'].nunique()} texts | {len(config.GENRES)} genres.")
    return df