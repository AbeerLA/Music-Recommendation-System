# 🎵 Music Recommender

A web-based music recommendation system that suggests tracks using **content-based filtering** with cosine similarity over a weighted, multi-block feature space (mood, activity, energy, valence, danceability, acousticness, tempo). Built as an undergraduate research project.

**[Live Demo](https://music-recommendation-system-qybp69ulz-abeerlas-projects.vercel.app/)**

![Music Recommender Screenshot](/Screenshot.jpg)

---

## Features

- 🎧 Pick a **genre**, **mood**, and optional **activity**
- 🎚️ Fine-tune results with **energy** and **positivity (valence)** sliders
- 🔢 Choose how many recommendations to show (Top 10 / 50 / 100 / all matches)
- 🧠 Ranked by **cosine similarity** over a 21-dimensional weighted feature vector
- 💬 Each track shows **explainable reasons** for why it matched (mood fit, activity fit, energy/valence closeness, danceability)
- 🎵 One-click links out to Spotify for tracks with a known Spotify ID
- ⚡ Runs entirely client-side — no backend or API keys required

## How It Works

The recommender builds a feature vector for every track and for the user's query, then ranks tracks by cosine similarity between the two.

| Block | Dimensions | Weight |
|---|---|---|
| Mood | 9 (one-hot) | 2.0 |
| Activity | 7 (one-hot) | 1.5 |
| Energy | 1 | 2.0 |
| Valence | 1 | 2.0 |
| Danceability | 1 | 0.7 |
| Acousticness | 1 | 0.7 |
| Tempo (normalized BPM) | 1 | 0.5 |

Genre is applied as a **hard filter**; activity is a **soft filter** that only applies if at least 3 tracks in the genre match it (otherwise it falls back to the full genre pool and shows a note). Danceability, acousticness, and tempo targets for the query are inferred from the energy slider.

See [`script.js`](script.js) for the full implementation.

## Tech Stack

- **Frontend:** Vanilla HTML, CSS, JavaScript — no frameworks, no build step
- **Data pipeline:** Python (`pandas`, `numpy`) — cleans and samples the source dataset
- **Testing:** Python black-box test suite (Equivalence Partitioning + Boundary Value Analysis)

## Project Structure

```
.
├── index.html          # App markup
├── style.css            # Styling
├── script.js             # Recommendation engine + UI logic
├── data.js               # Track data as window.MUSIC_DATA (for file:// loading)
├── data.json             # Same track data, fetched when served over http(s)
├── clean_data.py          # Reproducible data-cleaning/sampling pipeline
├── run_tests.py           # Black-box test suite (EP + BVA)
├── slider_eval.py         # Slider-effect evaluation script
├── data_clean.csv         # Cleaned, sampled dataset (analysis-friendly)
└── dataset.csv            # Raw source dataset (Kaggle Spotify Tracks Dataset)
```

## Data

Track data is derived from the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (Maharshi Pandya, Kaggle), ~114,000 tracks across 114 micro-genres.

`clean_data.py` reproduces the full pipeline:
1. Deduplicates and drops rows with missing core audio features
2. Filters tempo/duration outliers
3. Consolidates 114 micro-genres into 9 top-level genres
4. Draws a stratified sample balanced across genres
5. Derives multi-label mood and activity tags from audio features
6. Exports `data_clean.csv`, `data.json`, and `data.js`

```bash
python3 clean_data.py                # uses defaults (sample size 5000, seed 42)
python3 clean_data.py --sample 5000  # customize sample size
```

## Running Locally

Because the app fetches `data.json`, most browsers will block it when the page is opened directly from disk (`file://`). Serve it with a local web server instead:

```bash
python3 -m http.server 8000
```

Then open **http://localhost:8000** in your browser.

> Note: `data.js` also exposes the data as `window.MUSIC_DATA`, so the app will still work even opened directly via `file://` — the local server is only needed if you want the `data.json` fetch path to succeed too.

## Testing

Run the black-box test suite (equivalence partitioning + boundary value analysis, 24 cases covering genre filtering, mood/activity bias, slider boundaries, and result-count edge cases):

```bash
python3 run_tests.py
```

Results are printed to the console and written to `test_results.json`.

## Deployment

Live on **[Vercel](https://music-recommendation-system-qybp69ulz-abeerlas-projects.vercel.app/)**.

This is a fully static site — no server-side code, no environment variables, no build step — so it also deploys cleanly to GitHub Pages, Netlify, or Cloudflare Pages if you want an alternate or custom-domain host.

## License

Add your license here (e.g. MIT).

## Acknowledgments

- Track data: [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) by Maharshi Pandya
- Built as part of an undergraduate research seminar
