"""Simulate the recommender for 4 slider configurations and report
mean energy and mean valence of the top-10 recommendations."""
import math
import pandas as pd

# ---- Load the uploaded dataset ----
df = pd.read_csv("/tmp/data_clean.csv")
df["mood_list_parsed"]     = df["mood_list"].fillna("").apply(lambda s: s.split(";") if s else [])
df["activity_list_parsed"] = df["activity_list"].fillna("").apply(lambda s: s.split(";") if s else [])
print(f"Loaded {len(df)} tracks; Pop tracks: {(df['genre']=='Pop').sum()}")

# ---- Recommender constants (copied from script.js) ----
MOODS = ["Happy","Sad","Energetic","Calm","Romantic","Dark","Chill","Motivated","Nostalgic"]
ACTIVITIES = ["workout","studying","commute","party","sleep","cooking","driving"]
BW = {"mood":2.0,"activity":1.5,"energy":2.0,"valence":2.0,
      "danceability":0.7,"acousticness":0.7,"tempo":0.5}

def norm_bpm(b):
    v = (b - 50) / 130
    return max(0.0, min(1.0, v))

def song_vector(row):
    return {
        "mood":         [1 if m in row.mood_list_parsed else 0 for m in MOODS],
        "activity":     [1 if a in row.activity_list_parsed else 0 for a in ACTIVITIES],
        "energy":       [row.energy],
        "valence":      [row.valence],
        "danceability": [row.danceability],
        "acousticness": [row.acousticness],
        "tempo":        [norm_bpm(row.tempo)],
    }

def query_vector(mood, activity, energy, valence):
    return {
        "mood":         [1 if m == mood else 0 for m in MOODS],
        "activity":     [1 if (activity and a == activity) else 0 for a in ACTIVITIES],
        "energy":       [energy],
        "valence":      [valence],
        "danceability": [0.35 + 0.45 * energy],
        "acousticness": [0.65 - 0.45 * energy],
        "tempo":        [0.30 + 0.55 * energy],
    }

def flatten(vec):
    out = []
    for block in BW:
        for x in vec[block]:
            out.append(x * BW[block])
    return out

def cos_sim(a, b):
    dot = sum(a[i]*b[i] for i in range(len(a)))
    ma  = math.sqrt(sum(x*x for x in a))
    mb  = math.sqrt(sum(x*x for x in b))
    return dot / (ma * mb) if ma * mb > 0 else 0.0

def recommend(genre, mood, activity, energy, valence, k=10):
    pool = df[df["genre"] == genre].copy()
    if activity:
        filtered = pool[pool["activity_list_parsed"].apply(lambda xs: activity in xs)]
        if len(filtered) >= 3:
            pool = filtered
    qv = flatten(query_vector(mood, activity, energy, valence))
    scored = []
    for row in pool.itertuples(index=False):
        sv = flatten(song_vector(row))
        sim = cos_sim(sv, qv)
        scored.append((sim, row))
    scored.sort(key=lambda x: -x[0])
    return scored[:k]

# ---- The four test cases ----
CASES = [
    ("S1", "Pop", "Happy", "", 0.00, 0.50),
    ("S2", "Pop", "Happy", "", 1.00, 0.50),
    ("S3", "Pop", "Happy", "", 0.50, 0.00),
    ("S4", "Pop", "Happy", "", 0.50, 1.00),
]

results = []
print()
for case_id, genre, mood, activity, energy_s, valence_s in CASES:
    top10 = recommend(genre, mood, activity, energy_s, valence_s, k=10)
    print(f"=== {case_id}: genre={genre} mood={mood} activity={activity or 'all'} "
          f"energy={energy_s:.2f} positivity={valence_s:.2f} ===")
    energies = []
    valences = []
    for rank, (sim, row) in enumerate(top10, 1):
        energies.append(row.energy)
        valences.append(row.valence)
        title  = row.track_name[:45]
        artist = row.artists.split(";")[0][:30]
        print(f"  {rank:>2}. {title:<45} -- {artist:<30}  energy={row.energy:.3f}  valence={row.valence:.3f}  sim={sim:.3f}")
    mean_e = sum(energies)/len(energies)
    mean_v = sum(valences)/len(valences)
    print(f"  -> Mean energy: {mean_e:.3f}   Mean valence: {mean_v:.3f}")
    print()
    results.append({"case": case_id, "energy_s": energy_s, "valence_s": valence_s,
                    "mean_e": mean_e, "mean_v": mean_v})

print("=" * 70)
print("TABLE 1 -- Recommendation Statistics")
print("=" * 70)
print(f"{'Test ID':<10}{'Energy Slider':<18}{'Positivity Slider':<22}{'Mean Energy':<15}{'Mean Valence':<15}")
for r in results:
    print(f"{r['case']:<10}{r['energy_s']:<18.2f}{r['valence_s']:<22.2f}{r['mean_e']:<15.3f}{r['mean_v']:<15.3f}")

print()
print("=" * 70)
print("TABLE 2 -- Slider Effect Evaluation")
print("=" * 70)
s1, s2, s3, s4 = results

# S1 vs S2: energy slider 0.0 vs 1.0, positivity held at 0.5
energy_delta = s2["mean_e"] - s1["mean_e"]
energy_pass = energy_delta > 0
print(f"S1 vs S2: Mean Energy {s1['mean_e']:.3f} -> {s2['mean_e']:.3f}  "
      f"delta = {energy_delta:+.3f}  "
      f"-> {'PASS (increased)' if energy_pass else 'FAIL (did not increase)'}")

# S3 vs S4: positivity slider 0.0 vs 1.0, energy held at 0.5
valence_delta = s4["mean_v"] - s3["mean_v"]
valence_pass = valence_delta > 0
print(f"S3 vs S4: Mean Valence {s3['mean_v']:.3f} -> {s4['mean_v']:.3f}  "
      f"delta = {valence_delta:+.3f}  "
      f"-> {'PASS (increased)' if valence_pass else 'FAIL (did not increase)'}")
