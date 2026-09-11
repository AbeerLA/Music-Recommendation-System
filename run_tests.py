"""
run_tests.py
============

Black-box test runner for the music recommender.

Methodology
-----------
Two black-box techniques are used:

  * Equivalence Partitioning (EP)
        The input space (genre, mood, activity, energy, valence) is
        partitioned into equivalence classes. One representative case
        is drawn from each class on the assumption that the system
        will behave equivalently within a class.

  * Boundary Value Analysis (BVA)
        Cases at the boundaries of continuous inputs (energy = 0.0,
        energy = 1.0, valence = 0.0, valence = 1.0) are tested
        explicitly, since boundary defects are common.

Each test case has an explicit, measurable pass criterion that
operates only on the system's outputs (no inspection of internal
state) — this is what makes the testing "black box".

Output
------
  * Console summary table.
  * test_results.json with the full structured result of every case.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------
# Cosine-similarity recommender (Python port of script.js)
# ---------------------------------------------------------------

MOODS = ["Happy", "Sad", "Energetic", "Calm", "Romantic",
         "Dark", "Chill", "Motivated", "Nostalgic"]
ACTIVITIES = ["workout", "studying", "commute", "party",
              "sleep", "cooking", "driving"]
BW = {"mood": 2.0, "activity": 1.5, "energy": 2.0, "valence": 2.0,
      "danceability": 0.7, "acousticness": 0.7, "tempo": 0.5}


def norm_bpm(bpm: float) -> float:
    return max(0.0, min(1.0, (bpm - 50) / 130))


def song_vector(s: dict) -> dict:
    return {
        "mood":         [1 if m in s["mood"]     else 0 for m in MOODS],
        "activity":     [1 if a in s["activity"] else 0 for a in ACTIVITIES],
        "energy":       [s["energy"]],
        "valence":      [s["valence"]],
        "danceability": [s["danceability"]],
        "acousticness": [s["acousticness"]],
        "tempo":        [norm_bpm(s["bpm"])],
    }


def query_vector(mood: str, activity: Optional[str],
                 energy: float, valence: float) -> dict:
    return {
        "mood":         [1 if m == mood else 0 for m in MOODS],
        "activity":     [1 if (activity and a == activity) else 0 for a in ACTIVITIES],
        "energy":       [energy],
        "valence":      [valence],
        "danceability": [0.35 + 0.45 * energy],
        "acousticness": [0.65 - 0.45 * energy],
        "tempo":        [0.30 + 0.55 * energy],
    }


def flatten(vec: dict) -> list[float]:
    flat = []
    for block, weight in BW.items():
        for x in vec[block]:
            flat.append(x * weight)
    return flat


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    ma  = math.sqrt(sum(x * x for x in a))
    mb  = math.sqrt(sum(x * x for x in b))
    return dot / (ma * mb) if ma * mb > 0 else 0.0


def recommend(songs: list[dict], genre: str, mood: str,
              activity: Optional[str], energy: float, valence: float,
              limit: int | str = 10) -> dict:
    # Hard filter by genre.
    pool = [s for s in songs if s["genre"] == genre]

    activity_applied = False
    if activity:
        filtered = [s for s in pool if activity in s["activity"]]
        if len(filtered) >= 3:
            pool = filtered
            activity_applied = True

    qv = flatten(query_vector(mood, activity, energy, valence))
    scored = []
    for s in pool:
        sim = cosine(flatten(song_vector(s)), qv)
        scored.append({**s, "similarity": sim,
                       "match_score": round(sim * 100)})

    scored.sort(key=lambda x: -x["similarity"])
    total = len(scored)
    if limit == "all":
        results = scored
    else:
        results = scored[: int(limit)]
    return {"results": results, "total_matches": total,
            "activity_applied": activity_applied}


# ---------------------------------------------------------------
# Test case framework
# ---------------------------------------------------------------

@dataclass
class TestCase:
    id: str
    technique: str            # "EP" or "BVA"
    category: str             # group label (e.g. "Genre filter")
    description: str          # one-line summary
    inputs: dict
    expected: str             # plain-English expectation for the report
    check: Callable[[dict], tuple[bool, str]]   # returns (passed, message)
    actual: str = ""
    passed: bool = False
    message: str = ""


def avg(xs):
    return sum(xs) / len(xs) if xs else 0


# ---- Helper checks ----

def all_in_genre(genre):
    def check(out):
        bad = [r for r in out["results"] if r["genre"] != genre]
        if not bad:
            return True, f"all {len(out['results'])} results in genre={genre}"
        return False, f"{len(bad)} results not in genre={genre}"
    return check


def fraction_with_mood(mood, threshold):
    def check(out):
        n = len(out["results"])
        if n == 0:
            return False, "no results"
        hits = sum(1 for r in out["results"] if mood in r["mood"])
        frac = hits / n
        ok = frac >= threshold
        return ok, f"{hits}/{n} ({frac:.0%}) have mood={mood}, threshold {threshold:.0%}"
    return check


def fraction_with_activity(activity, threshold):
    def check(out):
        n = len(out["results"])
        if n == 0:
            return False, "no results"
        hits = sum(1 for r in out["results"] if activity in r["activity"])
        frac = hits / n
        ok = frac >= threshold
        return ok, f"{hits}/{n} ({frac:.0%}) have activity={activity}, threshold {threshold:.0%}"
    return check


def avg_feature_in_range(feature, lo, hi):
    def check(out):
        if not out["results"]:
            return False, "no results"
        vals = [r[feature] for r in out["results"]]
        a = avg(vals)
        ok = lo <= a <= hi
        return ok, f"avg {feature}={a:.3f}, expected in [{lo}, {hi}]"
    return check


def top_feature(feature, op, threshold):
    def check(out):
        if not out["results"]:
            return False, "no results"
        v = out["results"][0][feature]
        if op == ">":  ok = v >  threshold
        if op == "<":  ok = v <  threshold
        if op == ">=": ok = v >= threshold
        if op == "<=": ok = v <= threshold
        return ok, f"top {feature}={v:.3f} {op} {threshold}"
    return check


def result_count_equals(n):
    def check(out):
        actual = len(out["results"])
        ok = actual == n
        return ok, f"returned {actual} results, expected {n}"
    return check


def result_count_equals_pool(out):
    actual = len(out["results"])
    expected = out["total_matches"]
    ok = actual == expected
    return ok, f"returned {actual} results, pool size {expected}"


def top_similarity_above(threshold):
    def check(out):
        if not out["results"]:
            return False, "no results"
        sim = out["results"][0]["similarity"]
        ok = sim >= threshold
        return ok, f"top similarity={sim:.3f}, expected ≥ {threshold}"
    return check


def score_spread_above(min_spread_pp):
    def check(out):
        if len(out["results"]) < 10:
            return False, "fewer than 10 results"
        top = out["results"][0]["match_score"]
        bot = out["results"][-1]["match_score"]
        spread = top - bot
        ok = spread >= min_spread_pp
        return ok, f"score spread = {spread} pp (top {top}, bottom {bot}), expected ≥ {min_spread_pp}"
    return check


def fallback_triggered(out):
    # For the edge case: we expect activity_applied to be False (genre
    # pool kept) or for results to still be non-empty.
    ok = len(out["results"]) > 0
    return ok, ("results returned, activity_applied=" + str(out["activity_applied"]))


# ---------------------------------------------------------------
# Test plan
# ---------------------------------------------------------------

GENRES = ["Pop", "Rock", "Hip-Hop", "EDM", "Chill",
          "Latin", "R&B", "Jazz", "Classical"]


def build_test_cases() -> list[TestCase]:
    cases: list[TestCase] = []

    # ----- A. Functional: Genre hard filter (EP, 9 cases) -----
    genre_default = {
        "Pop":       ("Energetic", "workout",  0.80, 0.70),
        "Rock":      ("Energetic", "driving",  0.80, 0.50),
        "Hip-Hop":   ("Motivated", "workout",  0.80, 0.60),
        "EDM":       ("Energetic", "party",    0.90, 0.70),
        "Chill":     ("Calm",      "studying", 0.30, 0.50),
        "Latin":     ("Happy",     "party",    0.80, 0.85),
        "R&B":       ("Romantic",  "cooking",  0.50, 0.60),
        "Jazz":      ("Calm",      "studying", 0.30, 0.50),
        "Classical": ("Calm",      "sleep",    0.20, 0.40),
    }
    for i, g in enumerate(GENRES, 1):
        mood, act, e, v = genre_default[g]
        cases.append(TestCase(
            id=f"TC-F{i:02d}", technique="EP", category="Genre hard filter",
            description=f"{g} query returns only {g} tracks",
            inputs={"genre": g, "mood": mood, "activity": act,
                    "energy": e, "valence": v, "limit": 10},
            expected=f"All 10 results have genre = {g}",
            check=all_in_genre(g),
        ))

    # ----- B. Functional: Activity bias (EP, 3 cases) -----
    cases.append(TestCase(
        id="TC-F10", technique="EP", category="Activity bias",
        description="Workout activity biases toward workout-tagged tracks",
        inputs={"genre": "Pop", "mood": "Energetic", "activity": "workout",
                "energy": 0.80, "valence": 0.70, "limit": 10},
        expected="≥ 80% of top-10 carry the 'workout' activity tag",
        check=fraction_with_activity("workout", 0.80),
    ))
    cases.append(TestCase(
        id="TC-F11", technique="EP", category="Activity bias",
        description="Studying activity biases toward studying-tagged tracks",
        inputs={"genre": "Classical", "mood": "Calm", "activity": "studying",
                "energy": 0.25, "valence": 0.45, "limit": 10},
        expected="≥ 80% of top-10 carry the 'studying' activity tag",
        check=fraction_with_activity("studying", 0.80),
    ))
    cases.append(TestCase(
        id="TC-F12", technique="EP", category="Activity bias",
        description="Sleep activity biases toward sleep-tagged tracks",
        inputs={"genre": "Chill", "mood": "Calm", "activity": "sleep",
                "energy": 0.20, "valence": 0.45, "limit": 10},
        expected="≥ 70% of top-10 carry the 'sleep' activity tag",
        check=fraction_with_activity("sleep", 0.70),
    ))

    # ----- C. Functional: Mood bias (EP, 3 cases) -----
    cases.append(TestCase(
        id="TC-F13", technique="EP", category="Mood bias",
        description="Happy mood pushes top-10 average valence above 0.55",
        inputs={"genre": "Pop", "mood": "Happy", "activity": "",
                "energy": 0.70, "valence": 0.80, "limit": 10},
        expected="Average valence of top-10 ≥ 0.55",
        check=avg_feature_in_range("valence", 0.55, 1.0),
    ))
    cases.append(TestCase(
        id="TC-F14", technique="EP", category="Mood bias",
        description="Sad mood pulls top-10 average valence below 0.55",
        inputs={"genre": "Pop", "mood": "Sad", "activity": "",
                "energy": 0.30, "valence": 0.20, "limit": 10},
        expected="Average valence of top-10 ≤ 0.55",
        check=avg_feature_in_range("valence", 0.0, 0.55),
    ))
    cases.append(TestCase(
        id="TC-F15", technique="EP", category="Mood bias",
        description="Energetic mood pushes top-10 average energy above 0.60",
        inputs={"genre": "Rock", "mood": "Energetic", "activity": "",
                "energy": 0.85, "valence": 0.60, "limit": 10},
        expected="Average energy of top-10 ≥ 0.60",
        check=avg_feature_in_range("energy", 0.60, 1.0),
    ))

    # ----- D. Boundary: Slider extremes (BVA, 4 cases) -----
    cases.append(TestCase(
        id="TC-B01", technique="BVA", category="Slider boundary — energy",
        description="Energy slider at 0.00 → top result is low-energy",
        inputs={"genre": "Chill", "mood": "Calm", "activity": "",
                "energy": 0.00, "valence": 0.40, "limit": 10},
        expected="Top result has energy ≤ 0.45",
        check=top_feature("energy", "<=", 0.45),
    ))
    cases.append(TestCase(
        id="TC-B02", technique="BVA", category="Slider boundary — energy",
        description="Energy slider at 1.00 → top result is high-energy",
        inputs={"genre": "EDM", "mood": "Energetic", "activity": "",
                "energy": 1.00, "valence": 0.70, "limit": 10},
        expected="Top result has energy ≥ 0.80",
        check=top_feature("energy", ">=", 0.80),
    ))
    cases.append(TestCase(
        id="TC-B03", technique="BVA", category="Slider boundary — valence",
        description="Valence slider at 0.00 → top result is low-valence",
        inputs={"genre": "Pop", "mood": "Sad", "activity": "",
                "energy": 0.30, "valence": 0.00, "limit": 10},
        expected="Top result has valence ≤ 0.45",
        check=top_feature("valence", "<=", 0.45),
    ))
    cases.append(TestCase(
        id="TC-B04", technique="BVA", category="Slider boundary — valence",
        description="Valence slider at 1.00 → top result is high-valence",
        inputs={"genre": "Latin", "mood": "Happy", "activity": "",
                "energy": 0.80, "valence": 1.00, "limit": 10},
        expected="Top result has valence ≥ 0.70",
        check=top_feature("valence", ">=", 0.70),
    ))

    # ----- E. Boundary: Result-count selector (BVA, 3 cases) -----
    cases.append(TestCase(
        id="TC-B05", technique="BVA", category="Result-count selector",
        description="Limit = 10 returns exactly 10 results",
        inputs={"genre": "Pop", "mood": "Happy", "activity": "",
                "energy": 0.70, "valence": 0.70, "limit": 10},
        expected="Exactly 10 results returned",
        check=result_count_equals(10),
    ))
    cases.append(TestCase(
        id="TC-B06", technique="BVA", category="Result-count selector",
        description="Limit = 50 returns exactly 50 results",
        inputs={"genre": "Pop", "mood": "Happy", "activity": "",
                "energy": 0.70, "valence": 0.70, "limit": 50},
        expected="Exactly 50 results returned (pool ≥ 50)",
        check=result_count_equals(50),
    ))
    cases.append(TestCase(
        id="TC-B07", technique="BVA", category="Result-count selector",
        description='Limit = "all" returns the full match pool',
        inputs={"genre": "Pop", "mood": "Happy", "activity": "",
                "energy": 0.70, "valence": 0.70, "limit": "all"},
        expected="Result count equals total_matches",
        check=result_count_equals_pool,
    ))

    # ----- F. Edge cases (EP, 3 cases) -----
    cases.append(TestCase(
        id="TC-E01", technique="EP", category="Fallback behaviour",
        description="Classical + Party (rare combination) still returns results",
        inputs={"genre": "Classical", "mood": "Energetic", "activity": "party",
                "energy": 0.70, "valence": 0.65, "limit": 10},
        expected="Non-empty result set; fallback triggered if necessary",
        check=fallback_triggered,
    ))
    cases.append(TestCase(
        id="TC-E02", technique="EP", category="Match quality",
        description="Top result similarity is high for a well-specified query",
        inputs={"genre": "Classical", "mood": "Calm", "activity": "studying",
                "energy": 0.20, "valence": 0.40, "limit": 10},
        expected="Top similarity ≥ 0.80",
        check=top_similarity_above(0.80),
    ))
    cases.append(TestCase(
        id="TC-E03", technique="EP", category="Score distribution",
        description="Match-score range between top-10 and bottom-of-pool is wide",
        inputs={"genre": "EDM", "mood": "Calm", "activity": "studying",
                "energy": 0.20, "valence": 0.45, "limit": "all"},
        expected="Score spread (top − bottom) ≥ 25 percentage points",
        check=score_spread_above(25),
    ))

    return cases


# ---------------------------------------------------------------
# Runner
# ---------------------------------------------------------------

def main():
    data_path = Path("data.json")
    if not data_path.exists():
        print("ERROR: data.json not found. Run clean_data.py first.")
        sys.exit(1)

    with open(data_path, encoding="utf-8") as f:
        music = json.load(f)
    songs = music["songs"]

    cases = build_test_cases()

    print("=" * 76)
    print(f"  Black-box test suite — {len(cases)} cases against {len(songs):,} tracks")
    print("=" * 76)
    print(f"  {'ID':<8} {'Tech':<5} {'Category':<28} {'Result'}")
    print("  " + "-" * 72)

    passed = 0
    for tc in cases:
        out = recommend(songs, **tc.inputs)
        ok, msg = tc.check(out)
        tc.passed = ok
        tc.actual = msg
        tc.message = msg
        if ok:
            passed += 1
        flag = "PASS" if ok else "FAIL"
        print(f"  {tc.id:<8} {tc.technique:<5} {tc.category:<28} {flag}  — {msg}")

    print("  " + "-" * 72)
    print(f"  {passed}/{len(cases)} passed   "
          f"({100 * passed / len(cases):.1f}% pass rate)")
    print("=" * 76)

    # Save structured results for the paper.
    report = {
        "total_cases": len(cases),
        "passed": passed,
        "pass_rate": passed / len(cases),
        "cases": [
            {"id": c.id, "technique": c.technique, "category": c.category,
             "description": c.description, "inputs": c.inputs,
             "expected": c.expected, "actual": c.actual, "passed": c.passed}
            for c in cases
        ],
    }
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nFull report written to test_results.json")


if __name__ == "__main__":
    main()
