"""
correlations.py
===============
Day 4 — Spearman Correlations & Diachronic Analysis

Research question: Does morphological complexity vary systematically
across Sanskrit's 2400-year documented history (500 BCE – 1900 CE)?

Responsibilities:
  - assign_periods()          : assign approximate BCE/CE date to each text
                                using a curated lookup table grounded in
                                Sanskrit literary history
  - run_spearman()            : compute Spearman r between MCI, compound_rate,
                                mantra_rate, nom_verb_ratio and text period
  - plot_diachronic_scatter() : MCI and compound_rate over time, coloured
                                by genre, with regression line
  - plot_feature_period_grid(): small-multiple scatter grid for all key
                                features vs. period
  - plot_genre_period_box()   : boxplot of text periods per genre to show
                                the temporal distribution of genres
  - print_correlation_table() : formatted Spearman r / p-value table

All plots saved to config.OUTPUT_DIR.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import spearmanr

import config


# =============================================================================
# TEXT PERIOD LOOKUP TABLE
# =============================================================================
# Approximate midpoint date (CE; negative = BCE) for each DCS text.
# Sources: standard Sanskrit literary chronology (Levi, Keith, Witzel,
# Bronkhorst, Olivelle, Warder).
# Where a wide range exists, the midpoint is used.

TEXT_PERIODS = {
    # ── VEDIC (mostly BCE) ──────────────────────────────────────────────
    "Ṛgveda"                           : -1100,
    "Ṛgvedakhilāni"                    : -900,
    "Atharvaveda (Śaunaka)"            : -900,
    "Atharvaveda (Paippalāda)"         : -900,
    "Maitrāyaṇīsaṃhitā"               : -900,
    "Kāṭhakasaṃhitā"                  : -900,
    "Taittirīyasaṃhitā"               : -850,
    "Vājasaneyisaṃhitā (Mādhyandina)" : -850,
    "Aitareyabrāhmaṇa"                 : -800,
    "Kauṣītakibrāhmaṇa"               : -800,
    "Śatapathabrāhmaṇa"               : -750,
    "Taittirīyabrāhmaṇa"              : -750,
    "Jaiminīyabrāhmaṇa"               : -750,
    "Pañcaviṃśabrāhmaṇa"              : -750,
    "Gopathabrāhmaṇa"                  : -700,
    "Sāmavidhānabrāhmaṇa"             : -700,
    "Ṣaḍviṃśabrāhmaṇa"               : -700,
    "Aitareya-Āraṇyaka"               : -700,
    "Taittirīyāraṇyaka"               : -700,
    "Kaṭhāraṇyaka"                    : -700,
    "Śāṅkhāyanāraṇyaka"               : -700,
    "Bṛhadāraṇyakopaniṣad"            : -700,
    "Chāndogyopaniṣad"                : -700,
    "Aitareyopaniṣad"                  : -650,
    "Kauṣītakyupaniṣad"               : -650,
    "Taittirīyopaniṣad"               : -600,
    "Kaṭhopaniṣad"                    : -400,
    "Muṇḍakopaniṣad"                  : -400,
    "Śvetāśvataropaniṣad"             : -400,
    "Praśnopaniṣad"                   : -300,
    "Garbhopaniṣat"                   : -100,
    "Brahmabindūpaniṣat"              : 100,
    "Amṛtabindūpaniṣat"               : 200,
    "Nādabindūpaniṣat"                : 300,
    "Śira'upaniṣad"                   : 300,
    "Jaiminīya-Upaniṣad-Brāhmaṇa"    : -600,
    "Atharvaprāyaścittāni"            : -500,
    "Atharvavedapariśiṣṭa"            : -200,
    "Āśvalāyanagṛhyasūtra"            : -400,
    "Āśvālāyanaśrautasūtra"           : -400,
    "Pāraskaragṛhyasūtra"             : -400,
    "Gobhilagṛhyasūtra"               : -400,
    "Śāṅkhāyanagṛhyasūtra"            : -400,
    "Śāṅkhāyanaśrautasūtra"           : -400,
    "Kauśikasūtra"                    : -400,
    "Kauśikasūtradārilabhāṣya"        : -200,
    "Kauśikasūtrakeśavapaddhati"      : 900,
    "Kauṣītakagṛhyasūtra"            : -400,
    "Khādiragṛhyasūtra"               : -300,
    "Kātyāyanaśrautasūtra"            : -300,
    "Kāṭhakagṛhyasūtra"              : -400,
    "Hiraṇyakeśigṛhyasūtra"          : -300,
    "Hiraṇyakeśiśrautasūtra"         : -300,
    "Baudhāyanadharmasūtra"           : -400,
    "Baudhāyanagṛhyasūtra"            : -400,
    "Baudhāyanaśrautasūtra"           : -400,
    "Bhāradvājagṛhyasūtra"            : -400,
    "Bhāradvājaśrautasūtra"           : -400,
    "Drāhyāyaṇaśrautasūtra"          : -300,
    "Mānavagṛhyasūtra"                : -300,
    "Mānavaśrautasūtra"               : -300,
    "Vaikhānasadharmasūtra"           : 200,
    "Vaikhānasagṛhyasūtra"            : 200,
    "Vaikhānasaśrautasūtra"           : 200,
    "Vaitānasūtra"                    : -500,
    "Vasiṣṭhadharmasūtra"             : -300,
    "Vārāhagṛhyasūtra"               : -200,
    "Vārāhaśrautasūtra"              : -200,
    "Gautamadharmasūtra"              : -500,
    "Jaiminigṛhyasūtra"               : -200,
    "Jaiminīyaśrautasūtra"            : -400,
    "Āpastambadharmasūtra"            : -400,
    "Āpastambagṛhyasūtra"             : -400,
    "Āpastambaśrautasūtra"            : -400,
    "Nirukta"                         : -500,
    "Ṛgvedavedāṅgajyotiṣa"           : -500,
    "Ṛgvidhāna"                       : 200,

    # ── EPIC ────────────────────────────────────────────────────────────
    "Mahābhārata"                     : -200,
    "Rāmāyaṇa"                        : -300,
    "Harivaṃśa"                       : 300,
    "Bhāratamañjarī"                  : 1100,

    # ── KAVYA ───────────────────────────────────────────────────────────
    "Buddhacarita"                    : 100,
    "Saundarānanda"                   : 100,
    "Acintyastava"                    : 200,
    "Bodhicaryāvatāra"                : 700,
    "Aṣṭasāhasrikā"                   : 100,
    "Lalitavistara"                   : 300,
    "Laṅkāvatārasūtra"               : 400,
    "Saddharmapuṇḍarīkasūtra"        : 200,
    "Avadānaśataka"                   : 200,
    "Divyāvadāna"                     : 300,
    "Saṅghabhedavastu"                : 400,
    "Jātakamālā"                      : 400,
    "Śikṣāsamuccaya"                  : 700,
    "Kumārasaṃbhava"                  : 400,
    "Raghuvaṃśa"                      : 400,
    "Meghadūta"                       : 400,
    "Ṛtusaṃhāra"                      : 400,
    "Abhijñānaśākuntalam"             : 400,
    "Kirātārjunīya"                   : 600,
    "Śiśupālavadha"                   : 650,
    "Naiṣadhīyacarita"               : 1100,
    "Bhaṭṭikāvya"                     : 600,
    "Daśakumāracarita"                : 600,
    "Kādambarī"                       : 650,
    "Harṣacarita"                     : 650,
    "Gītagovinda"                     : 1200,
    "Hitopadeśa"                      : 1200,
    "Kathāsaritsāgara"               : 1100,
    "Amaruśataka"                     : 800,
    "Śatakatraya"                     : 500,
    "Āryāsaptaśatī"                   : 400,
    "Caurapañcaśikā"                  : 900,
    "Vetālapañcaviṃśatikā"           : 1100,
    "Bhallaṭaśataka"                  : 900,
    "Bhramarāṣṭaka"                   : 1100,
    "Haṃsadūta"                       : 1200,
    "Haṃsasaṃdeśa"                    : 1400,
    "Kokilasaṃdeśa"                   : 1200,
    "Meghadūta"                       : 400,
    "Smaradīpikā"                     : 1300,
    "Mukundamālā"                     : 900,
    "Narmamālā"                       : 1100,
    "Rasikapriyā"                     : 1400,
    "Rasikasaṃjīvanī"                 : 1600,
    "Sūryaśataka"                     : 700,
    "Haribhaktivilāsa"                : 1500,
    "Bhadrabāhucarita"                : 100,
    "Bhairavastava"                   : 800,
    "Aṣṭāvakragīta"                   : 200,
    "Bṛhatkathāślokasaṃgraha"        : 900,
    "Tantrākhyāyikā"                  : 1000,
    "Śukasaptati"                     : 1200,
    "Ṭikanikayātrā"                   : 1500,
    "Kādambarīsvīkaraṇasūtramañjarī" : 1600,

    # ── SHASTRA ─────────────────────────────────────────────────────────
    # Grammar
    "Aṣṭādhyāyī"                      : -400,
    "Kāśikāvṛtti"                     : 650,
    "Kāvyālaṃkāra"                    : 700,
    "Kāvyālaṃkāravṛtti"               : 950,
    "Kāvyādarśa"                      : 700,
    # Lexicography
    "Amarakośa"                        : 400,
    "Amaraughaśāsana"                  : 1300,
    "Abhidhānacintāmaṇi"              : 1150,
    "Abhinavacintāmaṇi"               : 1400,
    "Agastīyaratnaparīkṣā"           : 1300,
    "Aṣṭāṅganighaṇṭu"                : 900,
    "Bījanighaṇṭu"                    : 1300,
    "Dhanvantarinighaṇṭu"             : 900,
    "Kaiyadevanighaṇṭu"               : 1500,
    "Madanapālanighaṇṭu"              : 1375,
    "Nighaṇṭuśeṣa"                    : 1300,
    "Rājanighaṇṭu"                    : 1600,
    "Trikāṇḍaśeṣa"                    : 1100,
    # Dharmashastra
    "Arthaśāstra"                      : -300,
    "Manusmṛti"                        : 200,
    "Yājñavalkyasmṛti"                : 300,
    "Nāradasmṛti"                      : 400,
    "Parāśaradharmasaṃhitā"          : 800,
    "Parāśarasmṛtiṭīkā"              : 1000,
    "Viṣṇusmṛti"                      : 300,
    "Kāmasūtra"                        : 300,
    "Kātyāyanasmṛti"                  : 300,
    "Vṛddhayamasmṛti"                 : 500,
    "Gṛhastharatnākara"               : 1300,
    "Nibandhasaṃgraha"                : 1300,
    # Ayurveda
    "Carakasaṃhitā"                   : 100,
    "Suśrutasaṃhitā"                  : 300,
    "Aṣṭāṅgahṛdayasaṃhitā"          : 600,
    "Aṣṭāṅgasaṃgraha"                : 600,
    "Bhāvaprakāśa"                    : 1550,
    "Carakatattvapradīpikā"           : 1100,
    "Āyurvedadīpikā"                  : 1100,
    "Ayurvedarasāyana"                : 1400,
    "Indu (ad AHS)"                   : 900,
    "Nāḍīparīkṣā"                     : 1300,
    "Sarvāṅgasundarā"                 : 1300,
    "Śārṅgadharasaṃhitā"             : 1300,
    "Śārṅgadharasaṃhitādīpikā"      : 1400,
    # Rasa / Alchemy
    "Ānandakanda"                      : 1200,
    "Rasahṛdayatantra"                : 1100,
    "Rasakāmadhenu"                   : 1400,
    "Rasamañjarī"                     : 1300,
    "Rasaprakāśasudhākara"            : 1400,
    "Rasaratnasamuccaya"              : 1400,
    "Rasaratnasamuccayabodhinī"       : 1600,
    "Rasaratnasamuccayadīpikā"       : 1700,
    "Rasaratnasamuccayaṭīkā"         : 1700,
    "Rasaratnākara"                   : 800,
    "Rasasaṃketakalikā"              : 1500,
    "Rasataraṅgiṇī"                  : 1750,
    "Rasendracintāmaṇi"              : 1300,
    "Rasendracūḍāmaṇi"               : 1500,
    "Rasendrasārasaṃgraha"           : 1300,
    "Rasādhyāya"                     : 1400,
    "Rasādhyāyaṭīkā"                 : 1600,
    "Rasārṇava"                      : 1100,
    "Rasārṇavakalpa"                 : 1300,
    "Ratnadīpikā"                    : 1400,
    "Ratnaṭīkā"                      : 1500,
    "Yogaratnākara"                  : 1700,
    # Philosophy
    "Abhidharmakośa"                  : 400,
    "Abhidharmakośabhāṣya"           : 400,
    "Nyāyasūtra"                      : -100,
    "Nyāyabhāṣya"                     : 400,
    "Nyāyabindu"                      : 650,
    "Vaiśeṣikasūtra"                 : 100,
    "Vaiśeṣikasūtravṛtti"            : 900,
    "Mīmāṃsāsūtrabhāṣya"            : 800,
    "Sāṃkhyakārikā"                  : 350,
    "Sāṃkhyakārikābhāṣya"           : 800,
    "Sāṃkhyatattvakaumudī"           : 975,
    "Yogasūtra"                       : 400,
    "Yogasūtrabhāṣya"                : 500,
    "Tattvavaiśāradī"                : 850,
    "Mūlamadhyamakārikāḥ"           : 200,
    "Prasannapadā"                    : 500,
    "Vigrahavyāvartanī"              : 200,
    "Pramāṇasamuccaya"               : 480,
    "Pramāṇavārttika"                : 650,
    "Viṃśatikākārikā"               : 400,
    "Viṃśatikāvṛtti"                : 400,
    "Bodhisattvabhūmi"               : 350,
    "Madhyāntavibhāga"               : 350,
    "Saṃvitsiddhi"                   : 700,
    "Pañcārthabhāṣya"               : 400,
    "Gaṇakārikā"                     : 800,
    "Padārthacandrikā"               : 1300,
    "Spandakārikā"                   : 850,
    "Spandakārikānirṇaya"            : 900,
    "Tarkasaṃgraha"                  : 1600,
    "Sarvadarśanasaṃgraha"          : 1350,
    "Mugdhāvabodhinī"                : 1400,
    "Gūḍhārthadīpikā"               : 1600,
    "Janmamaraṇavicāra"             : 1200,
    "Sphuṭārthāvyākhyā"             : 1100,
    # Poetics
    "Nāṭyaśāstra"                    : 300,
    "Nāṭyaśāstravivṛti"             : 1000,
    "Rājamārtaṇḍa"                  : 950,
    # Astronomy
    "Sūryasiddhānta"                 : 400,
    "Sūryaśatakaṭīkā"               : 1200,
    # Technical
    "Dhanurveda"                     : 400,
    "Kṛṣiparāśara"                  : 1200,
    "Śyainikaśāstra"                 : 900,
    # Tantra
    "Nāṭyaśāstra"                    : 300,
    "Pāśupatasūtra"                  : 200,
    "Sātvatatantra"                  : 800,
    "Śivasūtra"                      : 850,
    "Śivasūtravārtika"               : 900,
    "Spandakārikā"                   : 850,
    "Tantrāloka"                     : 1000,
    "Tantrasāra"                     : 1000,
    "Toḍalatantra"                   : 1400,
    "Uḍḍāmareśvaratantra"           : 1100,
    "Vātūlanāthasūtras"              : 900,
    "Vātūlanāthasūtravṛtti"         : 950,
    "Mṛgendratantra"                 : 900,
    "Mṛgendraṭīkā"                   : 1100,
    "Mahācīnatantra"                 : 1200,
    "Mātṛkābhedatantra"              : 1200,
    "Paraśurāmakalpasūtra"          : 1400,
    "Paramānandīyanāmamālā"         : 1300,
    "Rasahṛdayatantra"               : 1100,
    "Śāktavijñāna"                  : 1200,
    "Gheraṇḍasaṃhitā"               : 1700,
    "Gorakṣaśataka"                  : 1100,
    "Haṭhayogapradīpikā"             : 1450,
    "Yogaratnākara"                  : 1700,
    # Commentaries
    "Commentary on Amaraughaśāsana"                     : 1400,
    "Commentary on the Kādambarīsvīkaraṇasūtramañjarī" : 1700,
    "Commentary on the Kāvyālaṃkāravṛtti"              : 1100,
    "Nibandhasaṃgraha"                                  : 1300,
    "Kādambarīsvīkaraṇasūtramañjarī"                   : 1600,

    # ── PURANIC ─────────────────────────────────────────────────────────
    "Bhāgavatapurāṇa"                : 900,
    "Viṣṇupurāṇa"                    : 400,
    "Matsyapurāṇa"                   : 400,
    "Kūrmapurāṇa"                    : 500,
    "Liṅgapurāṇa"                    : 600,
    "Varāhapurāṇa"                   : 700,
    "Agnipurāṇa"                     : 800,
    "Garuḍapurāṇa"                   : 800,
    "Śivapurāṇa"                     : 800,
    "Skandapurāṇa"                   : 600,
    "Skandapurāṇa (Revākhaṇḍa)"     : 700,
    "Narasiṃhapurāṇa"               : 700,
    "Kālikāpurāṇa"                   : 1000,
    "Devīkālottarāgama"              : 700,
    "Gokarṇapurāṇasāraḥ"            : 1200,
    "Maṇimāhātmya"                   : 1400,
    "Kṛṣṇāmṛtamahārṇava"            : 1600,
}


# =============================================================================
# ASSIGN PERIODS TO FEATURE MATRIX
# =============================================================================

def assign_periods(df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'period' column (approximate CE year, negative = BCE)
    to the feature DataFrame using TEXT_PERIODS lookup.

    Texts not in the lookup are assigned the genre median period.
    """
    df = df_feat.copy()
    df["period"] = df["text_name"].map(TEXT_PERIODS)

    # Fill missing with genre median
    genre_medians = (df.groupby("genre")["period"]
                     .median().to_dict())
    missing = df["period"].isna().sum()
    df["period"] = df.apply(
        lambda r: genre_medians.get(r["genre"], 500)
        if pd.isna(r["period"]) else r["period"],
        axis=1)

    if missing > 0:
        print(f"[correlations] {missing} texts had no period entry — "
              f"filled with genre median.")

    # Check if we have any real period data
    n_matched = df["period"].notna().sum() - missing
    if df["period"].isna().all() or df["period"].nunique() <= len(config.GENRES):
        print("[correlations] WARNING: No text-level period data found. "
              "Period analysis requires real DCS data with TEXT_PERIODS entries.")
        df["period"] = np.nan   # leave as NaN to skip downstream plots
        return df

    print(f"[correlations] Period range: "
          f"{df['period'].min():.0f} to {df['period'].max():.0f} CE")
    return df


# =============================================================================
# SPEARMAN CORRELATIONS
# =============================================================================

def run_spearman(df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman r between text period and key morphological features.
    Also computes within-genre correlations for MCI and compound_rate.

    Returns a DataFrame of results for display.
    """
    features = {
        "MCI"              : "Morphological Complexity Index",
        "compound_rate"    : "Compound Rate",
        "mantra_rate"      : "Mantra Rate (Vedic marker)",
        "nom_verb_ratio"   : "Nominal-to-Verbal Ratio",
        "pos_NOUN"         : "NOUN Rate",
        "pos_VERB"         : "VERB Rate",
        "pos_PRON"         : "PRON Rate",
    }

    rows = []
    print("\n[correlations] Spearman Correlations: Feature ~ Text Period")
    print(f"  {'Feature':<30} {'r':>7} {'p':>10}  {'sig':>4}")
    print("  " + "─" * 55)

    for feat, label in features.items():
        if feat not in df_feat.columns:
            continue
        valid = df_feat[["period", feat]].dropna()
        r, p  = spearmanr(valid["period"], valid[feat])
        sig   = "***" if p < 0.001 else ("**" if p < 0.01
                else ("*" if p < 0.05 else "ns"))
        rows.append({"feature": feat, "label": label,
                     "r": r, "p": p, "sig": sig})
        print(f"  {label:<30} {r:>+7.3f}  {p:>10.4f}  {sig:>4}")

    print("\n[correlations] Within-genre Spearman (MCI ~ Period):")
    for genre, grp in df_feat.groupby("genre"):
        valid = grp[["period", "MCI"]].dropna()
        if len(valid) < 4:
            print(f"  {genre:<12}: n={len(valid)} (too few for reliable r)")
            continue
        r, p = spearmanr(valid["period"], valid["MCI"])
        sig  = "***" if p<0.001 else("**" if p<0.01 else("*" if p<0.05 else "ns"))
        print(f"  {genre:<12}: r={r:+.3f}  p={p:.4f}  {sig}")

    return pd.DataFrame(rows)


# =============================================================================
# PLOT 11 — DIACHRONIC SCATTER (MCI and Compound Rate)
# =============================================================================

def plot_diachronic_scatter(df_feat: pd.DataFrame,
                             save: bool = True) -> None:
    """
    Two-panel scatter: MCI and Compound Rate vs. text period.
    Points coloured by genre. Regression line across all genres.
    """
    if df_feat["period"].isna().all():
        print("[correlations] Skipping diachronic scatter — no period data.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (feat, ylabel) in zip(axes, [
            ("MCI",           "MCI (bits)"),
            ("compound_rate", "Compound Rate (per 1000 tokens)")]):

        # Per-genre scatter
        for genre, grp in df_feat.groupby("genre"):
            col = config.GENRE_PALETTE.get(genre, "#888")
            ax.scatter(grp["period"], grp[feat],
                       color=col, alpha=0.60, s=45, label=genre,
                       edgecolors="white", linewidths=0.4, zorder=3)

        # Overall regression line (Spearman, fitted via polynomial)
        valid = df_feat[["period", feat]].dropna()
        z = np.polyfit(valid["period"], valid[feat], 1)
        p = np.poly1d(z)
        xs = np.linspace(valid["period"].min(),
                         valid["period"].max(), 200)
        r, pval = spearmanr(valid["period"], valid[feat])
        ax.plot(xs, p(xs), color="black", lw=1.5, ls="--",
                alpha=0.6, label=f"Trend (r={r:+.2f})")

        # BCE/CE divider
        ax.axvline(0, color="grey", lw=0.8, ls=":",
                   alpha=0.6, label="BCE | CE")
        ax.set_xlabel("Approximate Text Period (BCE < 0 < CE)",
                      fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{ylabel} over Time", fontsize=11)
        ax.legend(fontsize=7.5, title="Genre", title_fontsize=8,
                  loc="upper right")
        ax.grid(alpha=0.25)

    fig.suptitle(
        "Diachronic Variation in Sanskrit Morphological Features\n"
        "(texts spanning ~500 BCE – 1900 CE)",
        fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save:
        _savefig("11_diachronic_scatter.png")
    plt.show()


# =============================================================================
# PLOT 12 — FEATURE–PERIOD GRID
# =============================================================================

def plot_feature_period_grid(df_feat: pd.DataFrame,
                              save: bool = True) -> None:
    """
    2×3 small-multiple grid: each panel shows one feature vs. period,
    coloured by genre, with per-feature Spearman r annotated.
    """
    if df_feat["period"].isna().all():
        print("[correlations] Skipping feature-period grid — no period data.")
        return
    features = [
        ("MCI",           "MCI (bits)"),
        ("compound_rate", "Compound Rate"),
        ("nom_verb_ratio","Nom/Verb Ratio"),
        ("pos_NOUN",      "NOUN Rate"),
        ("pos_VERB",      "VERB Rate"),
        ("mantra_rate",   "Mantra Rate"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, (feat, label) in zip(axes, features):
        if feat not in df_feat.columns:
            ax.set_visible(False); continue

        for genre, grp in df_feat.groupby("genre"):
            col = config.GENRE_PALETTE.get(genre, "#888")
            ax.scatter(grp["period"], grp[feat],
                       color=col, alpha=0.55, s=30,
                       edgecolors="none", zorder=3)

        valid = df_feat[["period", feat]].dropna()
        r, p  = spearmanr(valid["period"], valid[feat])
        sig   = "***" if p<0.001 else("**" if p<0.01 else("*" if p<0.05 else "ns"))

        # Trend line
        z  = np.polyfit(valid["period"], valid[feat], 1)
        xs = np.linspace(valid["period"].min(),
                         valid["period"].max(), 200)
        ax.plot(xs, np.poly1d(z)(xs), color="black",
                lw=1.2, ls="--", alpha=0.5)
        ax.axvline(0, color="grey", lw=0.6, ls=":", alpha=0.5)
        ax.set_xlabel("Period (BCE<0<CE)", fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(f"{label}", fontsize=10, fontweight="bold")
        ax.annotate(f"r={r:+.2f} {sig}",
                    xy=(0.04, 0.92), xycoords="axes fraction",
                    fontsize=9, color="black",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="white", ec="grey", alpha=0.8))
        ax.grid(alpha=0.2)

    # Shared legend
    handles = [mpatches.Patch(color=config.GENRE_PALETTE.get(g,"#888"),
                              label=g) for g in config.GENRES]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=9, title="Genre", title_fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Feature Trends over Time (Spearman r annotated)",
        fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save:
        _savefig("12_feature_period_grid.png")
    plt.show()


# =============================================================================
# PLOT 13 — GENRE PERIOD DISTRIBUTION
# =============================================================================

def plot_genre_period_box(df_feat: pd.DataFrame,
                           save: bool = True) -> None:
    """
    Boxplot of text periods per genre — shows the temporal coverage
    of each genre in the DCS.
    """
    if df_feat["period"].isna().all():
        print("[correlations] Skipping genre period box — no period data.")
        return
    order = sorted(config.GENRES,
                   key=lambda g: df_feat[df_feat["genre"]==g]["period"]
                                 .median())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bp = ax.boxplot(
        [df_feat[df_feat["genre"]==g]["period"].dropna().values
         for g in order],
        patch_artist=True, labels=order,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker="o", markersize=4,
                        markerfacecolor="white", markeredgewidth=0.8))
    for patch, genre in zip(bp["boxes"], order):
        patch.set_facecolor(config.GENRE_PALETTE.get(genre, "#aaa"))
        patch.set_alpha(0.70)

    # Jitter overlay
    np.random.seed(config.RANDOM_SEED)
    for i, genre in enumerate(order, 1):
        vals   = df_feat[df_feat["genre"]==genre]["period"].dropna()
        jitter = np.random.uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=config.GENRE_PALETTE.get(genre, "#888"),
                   alpha=0.35, s=18, zorder=2, edgecolors="none")

    ax.axhline(0, color="grey", lw=0.8, ls="--",
               alpha=0.6, label="BCE | CE boundary")
    ax.set_xlabel("Genre", fontsize=11)
    ax.set_ylabel("Approximate Period (BCE < 0 < CE)", fontsize=11)
    ax.set_title(
        "Temporal Distribution of Sanskrit Texts by Genre\n"
        "(dots = individual texts; ordered by median period)",
        fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        _savefig("13_genre_period_box.png")
    plt.show()


# =============================================================================
# HELPER
# =============================================================================

def _savefig(filename: str) -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[correlations] Saved → {path}")