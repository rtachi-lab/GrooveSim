# GrooveSim

GrooveSim is a research-oriented Python prototype for estimating `grooveness` from:

- audio files such as WAV / MP3 / FLAC
- MIDI files, rendered to waveform first
- onset-time sequences stored as JSON or CSV
- browser uploads processed in-memory through a local web app
- GitHub Pages frontend + Render-hosted API deployment

The current implementation is a transparent feature-based MVP. It combines:

- auditory-like subband novelty
- beat entrainment and periodicity features
- syncopation and event density
- simple rhythm surprisal
- low-frequency drive

## Install

```bash
pip install -e .
```

## Usage

### Audio input

```bash
groovesim analyze-audio path/to/audio.wav --json-out result.json
```

If you already know the approximate tempo, you can provide a single hint:

```bash
groovesim analyze-audio path/to/audio.wav --tempo-hint 112
```

### Onset input

JSON:

```json
{
  "onsets": [0.12, 0.48, 0.74, 0.99, 1.51]
}
```

CSV:

```csv
time
0.12
0.48
0.74
0.99
1.51
```

Run:

```bash
groovesim analyze-onsets path/to/onsets.json
```

### MIDI input

```bash
groovesim analyze-midi path/to/pattern.mid --json-out result.json
```

Tempo hints are also accepted for MIDI rendering + analysis:

```bash
groovesim analyze-midi path/to/pattern.mid --tempo-hint 160
```

If you also want the rendered audio file:

```bash
groovesim analyze-midi path/to/pattern.mid --rendered-wav-out rendered.wav
```

### Browser app

```bash
groovesim-web
```

Then open `http://127.0.0.1:8000` in your browser.

Notes:

- uploaded files are processed in memory for the current request
- files are not saved on the server by the app
- audio decoding uses `ffmpeg`, so `ffmpeg` must be available on the server
- the browser UI also accepts an optional `tempo hint` input

## GitHub Pages + Render

This repository includes:

- `site/`: static frontend for GitHub Pages
- `render.yaml`: Render deployment config for the FastAPI API
- `Dockerfile`: container image definition with `ffmpeg` included
- `.github/workflows/pages.yml`: GitHub Pages deployment workflow

Recommended deployment:

1. Create a repository under your GitHub account or organization, for example `rtachi-lab/GrooveSim`.
2. Push this project to the repository.
3. On Render, create a new Blueprint or Web Service from that repository.
4. Deploy the API using `render.yaml` and the included `Dockerfile`.
5. Edit `site/config.js` and replace `https://YOUR-RENDER-SERVICE.onrender.com` with your actual Render API URL.
6. Push again.
7. In GitHub repository settings, enable GitHub Pages with GitHub Actions as the source.

After that:

- frontend URL: `https://rtachi-lab.github.io/<repo-name>/`
- API health check: `https://<your-render-service>.onrender.com/api/health`

Important:

- uploaded files are not written to project storage by the app
- audio files are decoded in memory via `ffmpeg`
- Render installs `ffmpeg` through the included container build
- if you use a repository name other than `GrooveSim`, your GitHub Pages path changes accordingly
- detailed deployment steps are in `DEPLOY.md`

## Output

The CLI prints a JSON object that includes:

- `groove_score`
- `tempo_bpm`
- `beat_strength`
- `beat_clarity`
- `periodicity_stability`
- `tempo_alignment_score`
- `syncopation_index`
- `event_density`
- `mean_surprisal`
- `moderate_surprisal_ratio`
- `low_freq_flux`
- `confidence`

## Notes

- This is an MVP for research exploration, not a validated perceptual model.
- `surprisal` is currently estimated from local rhythmic predictability rather than a full cognitive expectation model such as IDyOM.
- MIDI rendering currently uses `pretty_midi`'s built-in waveform synthesis before the normal audio analysis path.
