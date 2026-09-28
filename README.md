# Free Video Translator — Backend (real, free/open-source)

A working `/api/v1` backend that connects the existing frontend to a real
video-translation pipeline. **No paid APIs.** No fake progress.

## Stack (all free/open-source)
- **FFmpeg** — audio extraction, audio sync/mix, final video mux, aspect-ratio crop
- **faster-whisper** (tiny, int8) — speech-to-text with word timestamps + language detection
- **Helsinki-NLP opus-mt** — translation (lazy-loaded per language pair; English pivot fallback)
- **Piper TTS** — open-source neural voices (en/fr/ar/zh downloaded on first use)
- **FastAPI + background worker thread** — long videos run as background jobs

## Run
```bash
pip install -r requirements.txt
python3 server.py          # serves frontend at /, API at /api/v1
# open http://localhost:8000  — the Translate button now works for real
```

## Environment variables
- `PORT` (default 8000)
- `HF_TOKEN` (optional, higher HuggingFace rate limits for model downloads)
- Models cache to `./models`; jobs stored under `./storage/<jobId>/`

## API (v1)
| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/health` | GET | liveness + supported languages |
| `/api/v1/jobs` | POST (multipart `video`) | secure upload → `{jobId}` |
| `/api/v1/jobs/{id}/start` | POST `{sourceLang,targetLang,mode,aspect,volumes}` | enqueue background job |
| `/api/v1/jobs/{id}` | GET | poll `{status, stage, stageIndex, progress, error, downloadUrl}` |
| `/api/v1/jobs/{id}/subtitles` | GET/PUT | fetch / edit translated subtitles |
| `/api/v1/jobs/{id}/download` | GET | download final translated MP4 |

## Verified working
Tested on this host with a real 7.8s video containing real speech:
English → French, mode "Subtitles + Dub". Output MP4 (h264 + AAC dubbed audio
+ soft French subtitle track) produced in ~20s on CPU. See `translated_en_to_fr.mp4`.

## Limitations (honest)
- **CPU-only** (no GPU): works for short videos; long videos are slow but run as
  background jobs. Add a GPU + `WhisperModel("large-v3", device="cuda")` for speed.
- **Translation pairs**: opus-mt covers most en/fr/ar/zh pairs directly; missing
  pairs fall back to English-pivot. For all 12 pairs reliably, install
  `facebook/nllb-200-distilled-600M` (needs ~3GB RAM) in `_get_translator`.
- **Voices**: Arabic/Chinese Piper models download on first dub request.
- **Voice cloning** is not implemented (not claimed).
- Temporary audio/transcript files are auto-deleted after each job.

## Frontend
The existing frontend is served unchanged at `/` (copy in `static/index.html`).
It already uses `API_BASE='/api/v1'` (relative), so it connects automatically
when served from this server. To host the frontend elsewhere, set `API_BASE`
to your backend URL (CORS is open).
