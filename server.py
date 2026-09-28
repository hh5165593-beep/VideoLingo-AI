"""Free Video Translator — real backend API (FastAPI).
Run:  python3 server.py   (serves frontend at /, API at /api/v1)
"""
import os, json, uuid, threading, queue, time
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pipeline

BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "static")
os.makedirs(STATIC, exist_ok=True)

app = FastAPI(title="Free Video Translator API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

job_queue = queue.Queue()
job_configs = {}

def worker_loop():
    while True:
        job_id = job_queue.get()
        cfg = job_configs.get(job_id, {})
        pipeline.process_job(job_id, cfg)
        job_queue.task_done()

threading.Thread(target=worker_loop, daemon=True).start()

ALLOWED_EXT = (".mp4", ".mov", ".webm")
MAX_SIZE = 4 * 1024**3  # 4GB per upload; no time limit

@app.get("/api/v1/health")
def health():
    return {"status": "ok", "languages": pipeline.LANGS,
            "engine": {"stt": "faster-whisper (tiny, int8)", "translation": "Helsinki-NLP opus-mt (lazy)",
                       "tts": "Piper", "video": "FFmpeg"}}

@app.post("/api/v1/jobs")
async def create_job(video: UploadFile = File(...)):
    fn = (video.filename or "video").lower()
    if not fn.endswith(ALLOWED_EXT):
        raise HTTPException(400, "Unsupported format. Use MP4, MOV, or WebM.")
    job_id = uuid.uuid4().hex[:12]
    d = pipeline.job_dir(job_id)
    dest = os.path.join(d, "input" + os.path.splitext(fn)[1])
    size = 0
    with open(dest, "wb") as f:
        while True:
            chunk = await video.read(1024*1024)
            if not chunk: break
            size += len(chunk)
            if size > MAX_SIZE:
                f.close(); os.remove(dest)
                raise HTTPException(413, "File too large (>4GB).")
            f.write(chunk)
    # normalize to mp4 container for the pipeline
    norm = os.path.join(d, "input.mp4")
    if dest != norm:
        pipeline.run(["ffmpeg","-y","-i",dest,"-c","copy",norm])
        os.remove(dest)
    pipeline.set_status(job_id, 0, "Uploading", 100, status="queued")
    return {"jobId": job_id, "size": size}

@app.post("/api/v1/jobs/{job_id}/start")
def start_job(job_id: str, config: dict = Body(...)):
    d = pipeline.job_dir(job_id)
    if not os.path.exists(os.path.join(d, "input.mp4")):
        raise HTTPException(404, "Job not found. Upload video first.")
    job_configs[job_id] = config
    pipeline.set_status(job_id, 1, "Queued", 0, status="processing")
    job_queue.put(job_id)
    return {"ok": True, "jobId": job_id, "queued": job_queue.qsize()}

@app.get("/api/v1/jobs/{job_id}")
def job_status(job_id: str):
    s = pipeline.get_status(job_id)
    d = pipeline.job_dir(job_id)
    out = os.path.join(d, "output.mp4")
    if os.path.exists(out):
        s["downloadUrl"] = f"/api/v1/jobs/{job_id}/download"
        s["subtitlesUrl"] = f"/api/v1/jobs/{job_id}/subtitles"
    return s

@app.get("/api/v1/jobs/{job_id}/subtitles")
def get_subtitles(job_id: str):
    p = os.path.join(pipeline.job_dir(job_id), "subtitles.srt")
    if not os.path.exists(p):
        raise HTTPException(404, "Subtitles not ready.")
    with open(p, encoding="utf-8") as f:
        return PlainTextResponse(f.read(), media_type="text/plain; charset=utf-8")

@app.put("/api/v1/jobs/{job_id}/subtitles")
def put_subtitles(job_id: str, body: dict = Body(...)):
    # accept edited cues [{start,end,text}] and rewrite SRT
    cues = body.get("cues", [])
    pipeline.write_srt(os.path.join(pipeline.job_dir(job_id), "subtitles.srt"), cues)
    return {"ok": True, "cues": len(cues)}

@app.get("/api/v1/jobs/{job_id}/download")
def download(job_id: str):
    p = os.path.join(pipeline.job_dir(job_id), "output.mp4")
    if not os.path.exists(p):
        raise HTTPException(404, "Translated video not ready.")
    return FileResponse(p, media_type="video/mp4", filename=f"translated_{job_id}.mp4")

# serve the (unchanged) frontend
@app.get("/")
def index():
    idx = os.path.join(STATIC, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx, media_type="text/html")
    return JSONResponse({"detail": "frontend not found — place index.html in static/"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), log_level="info")
