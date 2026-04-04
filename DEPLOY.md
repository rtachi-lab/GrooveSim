# Deployment Guide

## Recommended topology

- Frontend: GitHub Pages
- API: Render Web Service
- Source code: GitHub repository

This project is already prepared for that setup.

## 1. Create the GitHub repository

Recommended repository name:

- `GrooveSim`

Then your Pages URL will become:

- `https://rtachi-lab.github.io/GrooveSim/`

## 2. Push the project

Example:

```bash
git init
git branch -M main
git add .
git commit -m "Initial GrooveSim web deployment setup"
git remote add origin https://github.com/rtachi-lab/GrooveSim.git
git push -u origin main
```

## 3. Deploy the API to Render

1. Sign in to Render
2. Create a new Web Service or Blueprint from `rtachi-lab/GrooveSim`
3. Render should detect `render.yaml`
4. Deploy

This repository includes a `Dockerfile`, so Render can run the API with
`ffmpeg` installed for audio decoding.

Expected API URL example:

- `https://groovesim-api.onrender.com`

## 4. Configure CORS on Render

Set this environment variable on Render:

```text
GROOVESIM_ALLOWED_ORIGINS=https://rtachi-lab.github.io,http://127.0.0.1:8000,http://localhost:8000
```

If you later use a custom domain for Pages, add it here as well.

## 5. Point the frontend to the API

Edit:

- `site/config.js`

Replace:

```js
window.GROOVESIM_API_BASE = "https://YOUR-RENDER-SERVICE.onrender.com";
```

with:

```js
window.GROOVESIM_API_BASE = "https://groovesim-api.onrender.com";
```

Commit and push again.

## 6. Enable GitHub Pages

In the GitHub repository:

1. Open `Settings`
2. Open `Pages`
3. Set `Source` to `GitHub Actions`

The workflow file is already included:

- `.github/workflows/pages.yml`

## 7. Verify

Frontend:

- `https://rtachi-lab.github.io/GrooveSim/`

API health:

- `https://groovesim-api.onrender.com/api/health`

Expected health response:

```json
{"status":"ok"}
```

## Notes

- The app is designed to process uploads in memory during the request.
- The application does not intentionally persist uploaded files to project storage.
- The Render deployment uses the included `Dockerfile`, which installs `ffmpeg`.
