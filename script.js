// =============================================================
// Music Recommender — content-based filtering with cosine
// similarity over a multi-block feature space, plus per-track
// explainability.
//
// Feature vector blocks (with block weights applied before
// cosine similarity):
//   mood        9 dims   weight 2.0
//   activity    7 dims   weight 1.5
//   energy      1 dim    weight 2.0
//   valence     1 dim    weight 2.0
//   dance       1 dim    weight 0.7
//   acoustic    1 dim    weight 0.7
//   tempo       1 dim    weight 0.5    (normalized BPM)
// =============================================================

const MOODS      = ["Happy","Sad","Energetic","Calm","Romantic","Dark","Chill","Motivated","Nostalgic"];
const ACTIVITIES = ["workout","studying","commute","party","sleep","cooking","driving"];

const BLOCK_WEIGHTS = {
  mood: 2.0,
  activity: 1.5,
  energy: 2.0,
  valence: 2.0,
  danceability: 0.7,
  acousticness: 0.7,
  tempo: 0.5,
};

// Normalize BPM into a 0..1 scale across roughly 50–180 BPM.
function normBpm(bpm) {
  const v = (bpm - 50) / 130;
  return Math.max(0, Math.min(1, v));
}

// Build a song's feature vector as an object keyed by block.
// (We keep blocks as named arrays rather than one flat array so we
//  can compute per-block contributions for the "why this song"
//  explanation.)
function songVector(song) {
  return {
    mood:         MOODS.map(m => song.mood.includes(m) ? 1 : 0),
    activity:     ACTIVITIES.map(a => song.activity.includes(a) ? 1 : 0),
    energy:       [song.energy],
    valence:      [song.valence],
    danceability: [song.danceability],
    acousticness: [song.acousticness],
    tempo:        [normBpm(song.bpm)],
  };
}

// Build the user's query vector from the form inputs.
function queryVector({ mood, activity, energy, valence }) {
  // Target danceability + acousticness are inferred from energy:
  // higher energy -> higher danceability, lower acousticness.
  const targetDance    = 0.35 + 0.45 * energy;        // 0.35–0.80
  const targetAcoustic = 0.65 - 0.45 * energy;        // 0.20–0.65
  const targetTempo    = 0.30 + 0.55 * energy;        // 0.30–0.85

  return {
    mood:         MOODS.map(m => m === mood ? 1 : 0),
    activity:     ACTIVITIES.map(a => activity && a === activity ? 1 : 0),
    energy:       [energy],
    valence:      [valence],
    danceability: [targetDance],
    acousticness: [targetAcoustic],
    tempo:        [targetTempo],
  };
}

// Apply block weights and flatten into a single array.
function flattenWeighted(vec) {
  const out = [];
  for (const block of Object.keys(BLOCK_WEIGHTS)) {
    const w = BLOCK_WEIGHTS[block];
    for (const v of vec[block]) out.push(v * w);
  }
  return out;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function magnitude(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s);
}

function cosineSim(a, b) {
  const m = magnitude(a) * magnitude(b);
  return m === 0 ? 0 : dot(a, b) / m;
}

// Per-block contribution = weighted dot product on that block alone.
// Useful for explaining WHY a song was ranked highly.
function blockContributions(songVec, queryVec) {
  const contribs = {};
  for (const block of Object.keys(BLOCK_WEIGHTS)) {
    const w = BLOCK_WEIGHTS[block];
    let d = 0;
    for (let i = 0; i < songVec[block].length; i++) {
      d += (songVec[block][i] * w) * (queryVec[block][i] * w);
    }
    contribs[block] = d;
  }
  return contribs;
}

// Pick the top reasons a song matched the query.
function buildReasons(song, query, contribs) {
  const reasons = [];

  // Mood reason: did the picked mood actually land on this song?
  if (song.mood.includes(query.moodLabel)) {
    reasons.push({ label: `mood: ${query.moodLabel.toLowerCase()}`, weight: contribs.mood });
  }

  // Activity reason: explicit activity hit.
  if (query.activityLabel && song.activity.includes(query.activityLabel)) {
    reasons.push({ label: `fits ${query.activityLabel}`, weight: contribs.activity });
  }

  // Audio-feature reasons: report whichever is close to the target.
  const closeness = (s, q) => 1 - Math.abs(s - q);
  if (closeness(song.energy,  query.energy)  > 0.85) {
    reasons.push({ label: song.energy > 0.6  ? "high energy match" : "low energy match", weight: 1 });
  }
  if (closeness(song.valence, query.valence) > 0.85) {
    reasons.push({ label: song.valence > 0.6 ? "uplifting feel"     : "melancholic feel", weight: 1 });
  }
  if (closeness(song.danceability, 0.35 + 0.45 * query.energy) > 0.88) {
    reasons.push({ label: "danceability fits", weight: 0.5 });
  }

  // Sort by importance, take top 3.
  return reasons.sort((a, b) => b.weight - a.weight).slice(0, 3).map(r => r.label);
}

// =============================================================
// Recommender entry point
// =============================================================
async function loadData() {
  // Primary: data.js exposes window.MUSIC_DATA (works from file:// too).
  // Fallback: fetch data.json when served over http(s).
  if (window.MUSIC_DATA) return window.MUSIC_DATA;
  const res = await fetch("data.json");
  if (!res.ok) throw new Error("Could not load data.json");
  return await res.json();
}

document.addEventListener("DOMContentLoaded", async () => {
  let musicData;
  const results = document.getElementById("results");

  try {
    musicData = await loadData();
  } catch (e) {
    results.innerHTML = `<div class="error-state">Could not load data.json — open the app via a local web server (e.g. <code>python3 -m http.server</code>) rather than double-clicking the file. Browsers block local fetch() from file:// URLs.</div>`;
    return;
  }

  // ---- Activity chip handling ----
  let selectedActivity = "";
  document.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      selectedActivity = chip.dataset.val;
    });
  });

  // ---- Slider handling ----
  const energyEl    = document.getElementById("energy");
  const energyVal   = document.getElementById("energy-val");
  const valenceEl   = document.getElementById("valence");
  const valenceVal  = document.getElementById("valence-val");

  const syncSlider = (el, label) => {
    label.textContent = (el.value / 100).toFixed(2);
  };
  energyEl.addEventListener("input",  () => syncSlider(energyEl,  energyVal));
  valenceEl.addEventListener("input", () => syncSlider(valenceEl, valenceVal));
  syncSlider(energyEl,  energyVal);
  syncSlider(valenceEl, valenceVal);

  // ---- Recommend ----
  function recommend() {
    const genre    = document.getElementById("genre").value;
    const mood     = document.getElementById("mood").value;
    const activity = selectedActivity;
    const energy   = parseInt(energyEl.value, 10)  / 100;
    const valence  = parseInt(valenceEl.value, 10) / 100;
    const btn      = document.getElementById("recBtn");

    btn.disabled    = true;
    btn.textContent = "⏳ Finding tracks...";

    // Genre = hard filter.
    const genrePool = musicData.songs.filter(s => s.genre === genre);

    // Activity = soft (keeps the pool from going empty).
    let pool = genrePool;
    let activityApplied = false;
    if (activity) {
      const filtered = genrePool.filter(s => s.activity.includes(activity));
      if (filtered.length >= 3) {
        pool = filtered;
        activityApplied = true;
      }
    }

    const query = queryVector({ mood, activity, energy, valence });
    const qFlat = flattenWeighted(query);

    const scored = pool.map(song => {
      const sv      = songVector(song);
      const sFlat   = flattenWeighted(sv);
      const sim     = cosineSim(sFlat, qFlat);                 // 0..1
      const contribs = blockContributions(sv, query);
      const reasons = buildReasons(song, {
        moodLabel: mood, activityLabel: activity, energy, valence
      }, contribs);
      return {
        ...song,
        similarity: sim,
        matchScore: Math.round(sim * 100),
        reasons,
      };
    });

    const ranked = scored.sort((a, b) => b.similarity - a.similarity);

    // How many to show: read from the count selector.
    const countSel = document.getElementById("count").value;
    const limit    = countSel === "all" ? ranked.length : parseInt(countSel, 10);
    const tracks   = ranked.slice(0, limit);
    const totalMatches = ranked.length;

    if (tracks.length === 0) {
      results.innerHTML = `<div class="error-state">No tracks found. Try a different genre.</div>`;
    } else {
      results.innerHTML = renderTracks(tracks, genre, activity, activityApplied, musicData, totalMatches);
    }

    btn.disabled    = false;
    btn.textContent = "♪ Get Recommendations";
  }

  document.getElementById("recBtn").addEventListener("click", recommend);
});

// =============================================================
// Rendering
// =============================================================
function renderTracks(tracks, genre, activity, activityApplied, musicData, totalMatches) {
  const emoji = musicData.genre_emoji[genre] || "🎵";

  // Header with the total match count so the user can see how the
  // displayed slice compares to the full match pool.
  let html = `<div class="results-header">
    <span>Your Playlist</span>
    <span class="match-count">Showing ${tracks.length} of ${totalMatches} matches</span>
  </div>`;

  if (activity && !activityApplied) {
    html += `<div class="insight-box">
      <div class="insight-label">Note</div>
      <div class="insight-text">Not enough tracks matched "${activity}" exactly — showing the closest fits from this genre instead.</div>
    </div>`;
  }

  html += `<div class="insight-box">
    <div class="insight-label">Algorithm</div>
    <div class="insight-text">Content-based filtering · cosine similarity over a 21-dim weighted feature vector (mood, activity, energy, valence, danceability, acousticness, tempo).</div>
  </div>`;

  // Cap the cascading animation delay at 30 cards — past that the
  // staggered effect becomes a long wait, especially in "All" mode.
  const MAX_ANIM = 30;

  // Spotify "wave" logo SVG used as the click affordance on each card.
  const SPOTIFY_ICON = `
    <svg viewBox="0 0 24 24" width="22" height="22" fill="#1DB954"
         xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.42 1.56-.299.421-1.02.599-1.559.3z"/>
    </svg>`;

  tracks.forEach((t, i) => {
    const reasonsHtml = t.reasons.length
      ? `<div class="reasons">${t.reasons.map(r => `<span class="reason-tag">${r}</span>`).join("")}</div>`
      : "";

    const delay = Math.min(i, MAX_ANIM) * 35;

    // If we have a Spotify ID, the whole card opens the Spotify page in a
    // new tab. Otherwise we fall back to a plain card with no link.
    const spotifyUrl  = t.spotify_id
      ? `https://open.spotify.com/track/${t.spotify_id}`
      : null;
    const cardTag     = spotifyUrl ? "a" : "div";
    const linkAttrs   = spotifyUrl
      ? `href="${spotifyUrl}" target="_blank" rel="noopener noreferrer" title="Open on Spotify"`
      : "";
    const spotifyBadge = spotifyUrl
      ? `<div class="spotify-badge">${SPOTIFY_ICON}</div>`
      : "";

    html += `
      <${cardTag} ${linkAttrs} class="track-card" style="animation-delay:${delay}ms">
        <div class="track-num">${i + 1}</div>
        <div class="track-art">${emoji}</div>
        <div class="track-info">
          <div class="track-title">${t.title}</div>
          <div class="track-artist">${t.artist}</div>
          ${reasonsHtml}
        </div>
        <div class="track-meta">
          <span class="match-pill">${t.matchScore}% match</span>
          <span class="track-bpm">${t.bpm} BPM</span>
        </div>
        ${spotifyBadge}
      </${cardTag}>`;
  });

  return html;
}
