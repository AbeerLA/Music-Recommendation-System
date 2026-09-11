"""
clean_data.py
=============

Reproducible preprocessing pipeline for the music-recommender study.

INPUT
-----
dataset.csv  -- Spotify Tracks Dataset by Maharshi Pandya (Kaggle).
               ~114,000 rows, 21 columns including the Spotify audio
               features (energy, valence, danceability, acousticness,
               instrumentalness, tempo, etc.) and a ``track_genre``
               column drawn from 114 micro-genres.

OUTPUT
------
data_clean.csv  -- the cleaned, sampled dataset (one row per track).
data.json       -- the same data shaped for the recommender app.
data.js         -- a tiny JS wrapper exposing window.MUSIC_DATA so the
                   browser app can load it directly from file://.

USAGE
-----
    python3 clean_data.py                # uses defaults
    python3 clean_data.py --sample 5000  # change target sample size

A fixed random seed (42) is used throughout so the sample is
reproducible. Every filtering step prints the row count before and
after so the figures can be quoted directly in the paper.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

RANDOM_SEED   = 42
TARGET_SAMPLE = 5000

# The nine top-level genre buckets the application supports.
# Every micro-genre in the source data is either mapped to one of
# these, or dropped (see UNMAPPED_DROP below).
GENRE_MAP = {
    # The mapping below was produced by a four-test peer-review of the
    # full 114-micro-genre vocabulary. A micro-genre was retained in
    # its bucket only if it satisfied all four tests:
    #   (i)   musicological coherence with the bucket
    #   (ii)  unambiguous reference to one tradition
    #   (iii) musicological rather than descriptor-based
    #   (iv)  non-redundant with the mood and activity axes
    # See GENRE_MAPPING_REVISED.md for the full reasoning.

    # ---- Pop ----  (8 micro-genres)
    "pop": "Pop", "indie-pop": "Pop", "synth-pop": "Pop",
    "power-pop": "Pop", "k-pop": "Pop", "j-pop": "Pop",
    "cantopop": "Pop", "mandopop": "Pop",

    # ---- Rock ----  (17 micro-genres)
    "rock": "Rock", "alt-rock": "Rock", "alternative": "Rock",
    "hard-rock": "Rock", "psych-rock": "Rock",
    "punk": "Rock", "punk-rock": "Rock", "grunge": "Rock",
    "emo": "Rock", "j-rock": "Rock", "rock-n-roll": "Rock",
    "rockabilly": "Rock", "heavy-metal": "Rock", "metal": "Rock",
    "metalcore": "Rock", "black-metal": "Rock",
    "death-metal": "Rock",

    # ---- Hip-Hop ----  (1 micro-genre)
    "hip-hop": "Hip-Hop",

    # ---- EDM ----  (18 micro-genres)
    "edm": "EDM", "dance": "EDM", "electronic": "EDM",
    "electro": "EDM", "house": "EDM", "deep-house": "EDM",
    "progressive-house": "EDM", "chicago-house": "EDM",
    "techno": "EDM", "detroit-techno": "EDM",
    "minimal-techno": "EDM", "trance": "EDM",
    "dubstep": "EDM", "drum-and-bass": "EDM",
    "breakbeat": "EDM", "hardstyle": "EDM", "idm": "EDM",
    "j-dance": "EDM",

    # ---- Chill ----  (2 micro-genres)
    "ambient": "Chill", "new-age": "Chill",

    # ---- Latin ----  (10 micro-genres)
    "latin": "Latin", "latino": "Latin", "reggaeton": "Latin",
    "salsa": "Latin", "samba": "Latin",
    "mpb": "Latin", "forro": "Latin", "sertanejo": "Latin",
    "pagode": "Latin", "tango": "Latin",

    # ---- R&B ----  (4 micro-genres)
    "r-n-b": "R&B", "soul": "R&B", "funk": "R&B",
    "gospel": "R&B",

    # ---- Jazz ----  (1 micro-genre)
    "jazz": "Jazz",

    # ---- Classical ----  (2 micro-genres)
    "classical": "Classical", "opera": "Classical",
}

# Micro-genres deliberately excluded after the strict academic review.
# See GENRE_MAPPING_REVISED.md Section 1 (Table 2) for per-item reasons.
UNMAPPED_DROP = {
    # Mood / activity descriptors masquerading as genres
    "chill", "happy", "party", "sad", "sleep", "study", "romance",

    # Performer-role and production-model descriptors
    "singer-songwriter", "songwriter", "indie",

    # Instrument and musical-quality descriptors
    "acoustic", "guitar", "groove", "piano",

    # Geographic / linguistic / audience tags
    "anime", "comedy", "disney", "kids", "children", "indian",
    "iranian", "malay", "turkish", "french", "german", "swedish",
    "world-music", "british", "brazil", "spanish",

    # Lexically ambiguous tokens
    "hardcore", "garage", "club",

    # Structurally distinct from any of the nine top-level buckets
    "disco", "goth", "industrial", "ska", "trip-hop", "folk",
    "dub", "pop-film", "j-idol", "show-tunes",

    # Geographically misplaced if forced into Latin
    "reggae", "dancehall", "afrobeat",

    # Country-adjacent (no dedicated bucket in the recommender)
    "country", "honky-tonk", "bluegrass",

    # Quality-control exclusion: blues bucket contaminated with
    # Southern rock, metal, R&B, and non-English popular music
    "blues",
}

GENRE_EMOJI = {
    "Pop": "\U0001F3A4", "Rock": "\U0001F3B8",
    "Hip-Hop": "\U0001F3A7", "EDM": "\U0001F4BF",
    "Chill": "\U0001F30A", "Latin": "\U0001F483",
    "R&B": "\U0001F3B6",   "Jazz": "\U0001F3B7",
    "Classical": "\U0001F3BB",
}

MOODS = [
    "Happy", "Sad", "Energetic", "Calm", "Romantic",
    "Dark", "Chill", "Motivated", "Nostalgic",
]

ACTIVITIES = [
    "workout", "studying", "commute", "party",
    "sleep", "cooking", "driving",
]

REQUIRED_FEATURES = [
    "danceability", "energy", "valence",
    "acousticness", "instrumentalness", "tempo",
]


# ---------------------------------------------------------------
# Mood and activity derivation
#
# These rules are documented in PREPROCESSING.md and Table 1 of the
# paper. They were chosen to match common conventions in the
# music-information-retrieval literature: valence/energy quadrants
# for Happy/Sad/Energetic/Calm, and additional axes for the niche
# moods.
# ---------------------------------------------------------------

def derive_moods(row) -> list:
    e, v = row["energy"], row["valence"]
    a, d = row["acousticness"], row["danceability"]
    moods = []

    if v > 0.65 and e > 0.50:                   moods.append("Happy")
    if v < 0.35 and e < 0.55:                   moods.append("Sad")
    if e > 0.75:                                moods.append("Energetic")
    if e < 0.40:                                moods.append("Calm")
    if 0.40 <= v <= 0.75 and 0.30 <= e <= 0.65 and a > 0.30:
        moods.append("Romantic")
    if v < 0.40 and e > 0.50:                   moods.append("Dark")
    if 0.30 < e < 0.60 and d < 0.70:            moods.append("Chill")
    if e > 0.70 and v > 0.50:                   moods.append("Motivated")
    if a > 0.40 and 0.30 <= v <= 0.65 and e < 0.60:
        moods.append("Nostalgic")

    # Fallback so every track is labelled with at least one mood.
    if not moods:
        if v >= 0.5 and e >= 0.5: moods = ["Happy"]
        elif v < 0.5 and e >= 0.5: moods = ["Dark"]
        elif v >= 0.5 and e < 0.5: moods = ["Calm"]
        else:                      moods = ["Sad"]
    return moods


def derive_activities(row) -> list:
    e, v, d = row["energy"], row["valence"], row["danceability"]
    a, i, t = row["acousticness"], row["instrumentalness"], row["tempo"]
    acts = []

    if e > 0.70 and t > 110:                          acts.append("workout")
    if i > 0.40 or (a > 0.50 and e < 0.45):           acts.append("studying")
    if 0.35 <= e <= 0.75:                             acts.append("commute")
    if d > 0.70 and e > 0.60 and v > 0.45:            acts.append("party")
    if e < 0.30 and (a > 0.40 or i > 0.40):           acts.append("sleep")
    if 0.40 <= e <= 0.70 and v > 0.40:                acts.append("cooking")
    if 0.50 <= e <= 0.85 and t > 90:                  acts.append("driving")

    # Fallback so every track has at least one activity.
    if not acts:
        if   e > 0.7:  acts = ["workout"]
        elif e < 0.3:  acts = ["sleep"]
        else:          acts = ["commute"]
    return acts


# ---------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------

def step(label: str, before: int, after: int) -> None:
    delta = before - after
    pct   = (100 * delta / before) if before else 0
    print(f"  {label:<40s} {before:>7,}  →  {after:>7,}   "
          f"(dropped {delta:>6,}, {pct:5.2f}%)")


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop the unnamed row-index column written by pandas when the
    # source CSV was originally exported.
    drop_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    print(f"\n[1] Loaded source dataset:                          {n0:>7,} rows")

    # ----- 2. Drop duplicates (same track + same artist) -----
    n = len(df)
    df = df.drop_duplicates(subset=["track_name", "artists"], keep="first")
    step("Deduplicate (track_name + artists)", n, len(df))

    # ----- 3. Drop rows with missing values in the features we use -----
    n = len(df)
    df = df.dropna(subset=REQUIRED_FEATURES + ["track_genre", "track_name", "artists"])
    step("Drop missing values in core columns", n, len(df))

    # ----- 4. Outlier filtering on tempo and duration -----
    n = len(df)
    df = df[(df["tempo"] >= 30) & (df["tempo"] <= 250)]
    step("Filter tempo to [30, 250] BPM", n, len(df))

    if "duration_ms" in df.columns:
        n = len(df)
        df = df[(df["duration_ms"] >= 30_000) & (df["duration_ms"] <= 900_000)]
        step("Filter duration to [30 s, 15 min]", n, len(df))

    # ----- 5. Clip audio features into [0, 1] (safety net) -----
    for col in ["danceability", "energy", "valence",
                "acousticness", "instrumentalness", "speechiness", "liveness"]:
        if col in df.columns:
            df[col] = df[col].clip(0.0, 1.0)

    # ----- 6. Genre consolidation -----
    n = len(df)
    df = df[~df["track_genre"].isin(UNMAPPED_DROP)]
    df["genre"] = df["track_genre"].map(GENRE_MAP)
    df = df.dropna(subset=["genre"])
    step("Map 114 micro-genres → 9 top-level", n, len(df))

    return df


def stratified_sample(df: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    """Balanced sample across the 9 top-level genres.

    If a genre has fewer rows than ``target / 9`` it contributes all
    of its rows and the deficit is redistributed across the other
    genres proportional to their availability.
    """
    rng = np.random.default_rng(seed)
    per_bucket = target // len(df["genre"].unique())
    parts = []

    counts = df["genre"].value_counts().to_dict()
    deficit = 0

    for g, sub in df.groupby("genre"):
        if len(sub) <= per_bucket:
            parts.append(sub)
            deficit += per_bucket - len(sub)
        else:
            parts.append(sub.sample(per_bucket, random_state=seed))

    # Redistribute deficit by drawing additional rows from the
    # largest genres that still have capacity.
    if deficit > 0:
        already = pd.concat(parts)
        remaining = df.drop(already.index)
        if len(remaining) >= deficit:
            extra = remaining.sample(deficit, random_state=seed)
            parts.append(extra)

    sampled = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"\n[2] Stratified sample target = {target}, actual = {len(sampled)}")
    print("    Final per-genre counts:")
    for g, c in sampled["genre"].value_counts().items():
        print(f"        {g:<10s}  {c:>5d}")
    return sampled


def label(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[3] Deriving mood and activity multi-labels")
    df["mood_list"]     = df.apply(derive_moods, axis=1)
    df["activity_list"] = df.apply(derive_activities, axis=1)

    # Quick distribution summary, useful for the paper.
    print("    Mood label frequency:")
    for m in MOODS:
        c = df["mood_list"].apply(lambda xs: m in xs).sum()
        print(f"        {m:<10s}  {c:>5d}   ({100*c/len(df):5.1f}%)")
    print("    Activity label frequency:")
    for a in ACTIVITIES:
        c = df["activity_list"].apply(lambda xs: a in xs).sum()
        print(f"        {a:<10s}  {c:>5d}   ({100*c/len(df):5.1f}%)")
    return df


def export(df: pd.DataFrame, out_dir: Path) -> None:
    # ---- data_clean.csv : full table for analysis ----
    csv_cols = [
        "track_id", "track_name", "artists", "genre", "track_genre",
        "mood_list", "activity_list", "tempo", "danceability",
        "energy", "valence", "acousticness", "instrumentalness",
        "speechiness", "liveness", "loudness", "duration_ms",
        "popularity",
    ]
    csv_cols = [c for c in csv_cols if c in df.columns]
    df_csv = df[csv_cols].copy()
    df_csv["mood_list"]     = df_csv["mood_list"].apply(lambda xs: ";".join(xs))
    df_csv["activity_list"] = df_csv["activity_list"].apply(lambda xs: ";".join(xs))
    df_csv.to_csv(out_dir / "data_clean.csv", index=False)

    # ---- data.json : shape the app expects ----
    songs = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        songs.append({
            "id":           i,
            "spotify_id":   str(row.track_id),
            "title":        str(row.track_name)[:120],
            "artist":       str(row.artists).split(";")[0][:120],
            "genre":        row.genre,
            "mood":         list(row.mood_list),
            "activity":     list(row.activity_list),
            "bpm":          int(round(row.tempo)),
            "energy":       round(float(row.energy), 3),
            "valence":      round(float(row.valence), 3),
            "danceability": round(float(row.danceability), 3),
            "acousticness": round(float(row.acousticness), 3),
        })
    payload = {"genre_emoji": GENRE_EMOJI, "songs": songs}
    with open(out_dir / "data.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ---- data.js : same JSON wrapped for file:// loading ----
    with open(out_dir / "data.js", "w", encoding="utf-8") as f:
        f.write("window.MUSIC_DATA = ")
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write(";\n")

    print(f"\n[4] Wrote: data_clean.csv, data.json, data.js  ({len(songs):,} tracks)")


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Clean the Kaggle Spotify dataset for the recommender.")
    ap.add_argument("--input",  default="dataset.csv",
                    help="Path to the Kaggle CSV (default: ./dataset.csv).")
    ap.add_argument("--outdir", default=".",
                    help="Where to write outputs (default: ./).")
    ap.add_argument("--sample", type=int, default=TARGET_SAMPLE,
                    help=f"Target sample size after cleaning (default: {TARGET_SAMPLE}).")
    ap.add_argument("--seed",   type=int, default=RANDOM_SEED,
                    help=f"Random seed for reproducibility (default: {RANDOM_SEED}).")
    args = ap.parse_args()

    random.seed(args.seed)

    print("=" * 70)
    print("  Music Recommender -- data cleaning pipeline")
    print(f"  Input  = {args.input}")
    print(f"  Output = {args.outdir}")
    print(f"  Seed   = {args.seed} | Target sample = {args.sample}")
    print("=" * 70)

    in_path  = Path(args.input)
    out_dir  = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(in_path)
    df = clean(df)
    df = stratified_sample(df, target=args.sample, seed=args.seed)
    df = label(df)
    export(df, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
