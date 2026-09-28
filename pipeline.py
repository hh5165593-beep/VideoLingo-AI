"""Real video-translation pipeline. Free/open-source only.
Stages indices match the frontend:
0 Uploading, 1 Extracting Audio, 2 Transcribing, 3 Translating,
4 Generating Voice, 5 Synchronizing Audio, 6 Rendering Video, 7 Completed.
"""
import os, json, subprocess, wave, math, shutil
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
STORAGE = os.path.join(BASE, "storage")
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)

LANGS = ["ar", "en", "fr", "zh"]
WHISPER_LANG = {"ar": "ar", "en": "en", "fr": "fr", "zh": "zh"}

# Piper voices (medium quality where available)
PIPER_VOICES = {
    "en": ("en_US-lessac-medium", "en/en_US/lessac/medium/en_US-lessac-medium"),
    "fr": ("fr_FR-siwis-medium", "fr/fr_FR/siwis/medium/fr_FR-siwis-medium"),
    "ar": ("ar_JO-kareem-medium", "ar/ar_JO/kareem/medium/ar_JO-kareem-medium"),
    "zh": ("zh_CN-huayan-medium", "zh/zh_CN/huayan/medium/zh_CN-huayan-medium"),
}
PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Opus-MT direct pairs that exist on Helsinki-NLP (lazy download)
OPUS_DIRECT = {
    ("en","fr"):"Helsinki-NLP/opus-mt-en-fr", ("fr","en"):"Helsinki-NLP/opus-mt-fr-en",
    ("en","zh"):"Helsinki-NLP/opus-mt-en-zh", ("zh","en"):"Helsinki-NLP/opus-mt-zh-en",
    ("en","ar"):"Helsinki-NLP/opus-mt-en-ar", ("ar","en"):"Helsinki-NLP/opus-mt-ar-en",
    ("fr","zh"):"Helsinki-NLP/opus-mt-fr-zh",
}

def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)

def job_dir(job_id):
    d = os.path.join(STORAGE, job_id)
    os.makedirs(d, exist_ok=True)
    return d

def set_status(job_id, stage_index, stage, progress=0, status="processing", error=None):
    d = job_dir(job_id)
    s = {"jobId": job_id, "status": status, "stage": stage, "stageIndex": stage_index,
         "progress": progress, "error": error}
    with open(os.path.join(d, "status.json"), "w") as f:
        json.dump(s, f)
    return s

def get_status(job_id):
    p = os.path.join(job_dir(job_id), "status.json")
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return {"jobId": job_id, "status": "unknown", "stage": "Queued", "stageIndex": -1, "progress": 0}

# ---------- stage 1: extract audio ----------
def extract_audio(job_id):
    d = job_dir(job_id)
    inp = os.path.join(d, "input.mp4")
    wav = os.path.join(d, "audio.wav")
    orig = os.path.join(d, "original_audio.wav")
    r = run(["ffmpeg","-y","-i",inp,"-vn","-acodec","pcm_s16le","-ar","16000","-ac","1",wav])
    if r.returncode != 0: raise RuntimeError("ffmpeg extract failed: "+r.stderr[-300:])
    run(["ffmpeg","-y","-i",inp,"-vn","-acodec","pcm_s16le","-ar","22050","-ac","1",orig])
    return wav

def wav_duration(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()

# ---------- stage 2: transcribe (faster-whisper, tiny int8) ----------
_whisper_model = None
def transcribe(job_id, source_lang):
    global _whisper_model
    from faster_whisper import WhisperModel
    if _whisper_model is None:
        _whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8", download_root=MODELS)
    d = job_dir(job_id)
    wav = os.path.join(d, "audio.wav")
    lang = WHISPER_LANG.get(source_lang) if source_lang != "auto" else None
    segments, info = _whisper_model.transcribe(wav, language=lang, beam_size=1, vad_filter=True)
    segs = []
    for seg in segments:
        segs.append({"start": round(seg.start,3), "end": round(seg.end,3), "text": seg.text.strip()})
    detected = info.language
    with open(os.path.join(d, "transcript.json"), "w") as f:
        json.dump({"detected_language": detected, "segments": segs}, f, ensure_ascii=False)
    return segs, detected

# ---------- stage 3: translate (opus-mt, lazy per direction, pivot via en) ----------
_translators = {}
class _Translator:
    def __init__(self, model_name):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        self.tok = AutoTokenizer.from_pretrained(model_name, cache_dir=MODELS)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=MODELS)
        self.torch = torch
    def __call__(self, text, max_length=200):
        inputs = self.tok(text, return_tensors="pt", truncation=True)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_length=max_length)
        return [{"translation_text": self.tok.decode(out[0], skip_special_tokens=True)}]

def _get_translator(src, tgt):
    key = (src, tgt)
    if key in _translators: return _translators[key]
    model_name = OPUS_DIRECT.get(key)
    if model_name:
        _translators[key] = _Translator(model_name)
        return _translators[key]
    return None

def translate_segments(job_id, src, tgt):
    d = job_dir(job_id)
    with open(os.path.join(d, "transcript.json")) as f:
        tr = json.load(f)
    segs = tr["segments"]
    if src == "auto": src = tr.get("detected_language", "en")
    direct = _get_translator(src, tgt)
    pivot1 = pivot2 = None
    if direct is None and src != "en":
        pivot1 = _get_translator(src, "en")
        pivot2 = _get_translator("en", tgt)
    if direct is None and pivot1 is None:
        raise RuntimeError(f"No open-source translation model available for {src}→{tgt}. "
                           f"Install NLLB-200 (needs ~3GB RAM) or add opus-mt pair.")
    out = []
    for i, seg in enumerate(segs):
        text = seg["text"]
        if not text:
            translated = ""
        elif direct:
            translated = direct(text, max_length=200)[0]["translation_text"]
        else:
            mid = pivot1(text, max_length=200)[0]["translation_text"]
            translated = pivot2(mid, max_length=200)[0]["translation_text"] if pivot2 else mid
        out.append({"start": seg["start"], "end": seg["end"], "text": translated})
        set_status(job_id, 3, "Translating", progress=int((i+1)/len(segs)*100))
    with open(os.path.join(d, "translated.json"), "w") as f:
        json.dump({"source_language": src, "target_language": tgt, "segments": out}, f, ensure_ascii=False)
    write_srt(os.path.join(d, "subtitles.srt"), out)
    return out

def write_srt(path, segs):
    def ts(sec):
        h = int(sec//3600); m = int((sec%3600)//60); s = sec%60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segs, 1):
            f.write(f"{i}\n{ts(seg['start'])} --> {ts(seg['end'])}\n{seg['text']}\n\n")

# ---------- stage 4: TTS (piper) ----------
def _ensure_piper_voice(lang):
    name, path = PIPER_VOICES[lang]
    onnx = os.path.join(MODELS, name + ".onnx")
    if not os.path.exists(onnx):
        import urllib.request
        for ext in [".onnx", ".onnx.json"]:
            url = f"{PIPER_BASE}/{path}{ext}"
            dest = os.path.join(MODELS, name + ext)
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception:
                # try low-quality variant
                low_path = path.replace("/medium/", "/low/")
                low_name = name.replace("-medium", "-low")
                url2 = f"{PIPER_BASE}/{low_path}{ext.replace(name, low_name) if name in ext else ext}"
                urllib.request.urlretrieve(url2, dest.replace(name, low_name))
                if ext == ".onnx":
                    onnx = os.path.join(MODELS, low_name + ".onnx")
    return onnx

def generate_voice(job_id, tgt_lang):
    d = job_dir(job_id)
    with open(os.path.join(d, "translated.json")) as f:
        data = json.load(f)
    segs = data["segments"]
    onnx = _ensure_piper_voice(tgt_lang)
    tts_dir = os.path.join(d, "tts"); os.makedirs(tts_dir, exist_ok=True)
    for i, seg in enumerate(segs):
        out = os.path.join(tts_dir, f"seg_{i:04d}.wav")
        text = seg["text"] or "."
        r = run(["piper","--model", onnx, "--output_file", out], input=text)
        if r.returncode != 0:
            # fallback: silent segment
            run(["ffmpeg","-y","-f","lavfi","-i","anullsrc=r=22050:cl=mono","-t","0.5",out])
        set_status(job_id, 4, "Generating Voice", progress=int((i+1)/len(segs)*100))
    return tts_dir

# ---------- stage 5: synchronize + mix ----------
def _read_wav_mono(path, target_sr=22050):
    """Resample to mono target_sr via ffmpeg, return float32 numpy array."""
    tmp = path + ".res.wav"
    run(["ffmpeg","-y","-i",path,"-ac","1","-ar",str(target_sr),"-acodec","pcm_s16le",tmp])
    with wave.open(tmp, "rb") as w:
        n = w.getnframes(); raw = w.readframes(n)
    os.remove(tmp)
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return arr

def sync_and_mix(job_id, volumes):
    SR = 22050
    d = job_dir(job_id)
    with open(os.path.join(d, "translated.json")) as f:
        segs = json.load(f)["segments"]
    total_dur = wav_duration(os.path.join(d, "original_audio.wav"))
    dub = np.zeros(int(total_dur * SR) + SR, dtype=np.float32)
    tts_dir = os.path.join(d, "tts")
    for i, seg in enumerate(segs):
        wav_path = os.path.join(tts_dir, f"seg_{i:04d}.wav")
        if not os.path.exists(wav_path): continue
        slot = max(0.3, seg["end"] - seg["start"])
        # time-stretch TTS to fit slot using atempo (clamped 0.5..2.0)
        tts_dur = wav_duration(wav_path)
        ratio = max(0.5, min(2.0, tts_dur / slot)) if tts_dur > 0 else 1.0
        stretched = wav_path + ".fit.wav"
        if abs(ratio - 1.0) > 0.05:
            run(["ffmpeg","-y","-i",wav_path,"-filter:a",f"atempo={ratio:.3f}","-ac","1","-ar",str(SR),stretched])
        else:
            run(["ffmpeg","-y","-i",wav_path,"-ac","1","-ar",str(SR),stretched])
        arr = _read_wav_mono(stretched, SR)
        os.remove(stretched)
        start_sample = int(seg["start"] * SR)
        end_sample = min(len(dub), start_sample + len(arr))
        dub[start_sample:end_sample] += arr[:end_sample-start_sample] * (volumes.get("trans",100)/100.0)
        set_status(job_id, 5, "Synchronizing Audio", progress=int((i+1)/len(segs)*100))
    # mix with original audio (background)
    orig = _read_wav_mono(os.path.join(d, "original_audio.wav"), SR)
    orig_vol = volumes.get("orig", 100)/100.0 * 0.7  # lower original voice when dubbing
    bg_vol = volumes.get("bg", 80)/100.0
    n = max(len(orig), len(dub))
    mixed = np.zeros(n, dtype=np.float32)
    mixed[:len(orig)] += orig * (orig_vol + bg_vol*0.3)
    mixed[:len(dub)] += dub
    maxv = np.abs(mixed).max()
    if maxv > 0.95: mixed = mixed / maxv * 0.95
    out_wav = os.path.join(d, "dubbed.wav")
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((mixed * 32767).astype(np.int16).tobytes())
    return out_wav

# ---------- stage 6: render final video ----------
def render_video(job_id, mode, aspect):
    d = job_dir(job_id)
    inp = os.path.join(d, "input.mp4")
    out = os.path.join(d, "output.mp4")
    dubbed = os.path.join(d, "dubbed.wav")
    srt = os.path.join(d, "subtitles.srt")
    cmd = ["ffmpeg","-y","-i",inp]
    has_dub = mode in ("dub","both") and os.path.exists(dubbed)
    has_subs = mode in ("subtitles","both") and os.path.exists(srt)
    if has_dub: cmd += ["-i",dubbed]
    if has_subs: cmd += ["-i",srt]
    # video: copy original stream unless aspect change requested
    vf = None
    if aspect == "9:16": vf = "scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,setsar=1"
    elif aspect == "16:9": vf = "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1"
    elif aspect == "1:1": vf = "scale=720:720:force_original_aspect_ratio=decrease,pad=720:720:(ow-iw)/2:(oh-ih)/2,setsar=1"
    if vf: cmd += ["-vf",vf,"-c:v","libx264","-preset","fast","-crf","20"]
    else: cmd += ["-c:v","copy"]
    if has_dub:
        cmd += ["-map","0:v:0","-map","1:a:0","-c:a","aac","-b:a","192k"]
    else:
        cmd += ["-map","0:v:0","-map","0:a:0?","-c:a","aac","-b:a","192k"]
    if has_subs:
        sub_idx = 2 if has_dub else 1
        cmd += ["-map",f"{sub_idx}:0","-c:s","mov_text","-metadata:s:s:0","language=target"]
    cmd += ["-shortest", out]
    r = run(cmd)
    if r.returncode != 0: raise RuntimeError("ffmpeg render failed: "+r.stderr[-400:])
    return out

# ---------- full job ----------
def process_job(job_id, config):
    try:
        src = config.get("sourceLang","auto"); tgt = config.get("targetLang","en")
        mode = config.get("mode","subtitles"); aspect = config.get("aspect","original")
        volumes = config.get("volumes", {"orig":100,"trans":100,"bg":80})
        set_status(job_id, 1, "Extracting Audio", 5)
        extract_audio(job_id)
        set_status(job_id, 2, "Transcribing", 10)
        segs, detected = transcribe(job_id, src)
        if src == "auto": src = detected
        set_status(job_id, 3, "Translating", 30)
        translate_segments(job_id, src, tgt)
        if mode in ("dub","both"):
            set_status(job_id, 4, "Generating Voice", 50)
            generate_voice(job_id, tgt)
            set_status(job_id, 5, "Synchronizing Audio", 75)
            sync_and_mix(job_id, volumes)
        set_status(job_id, 6, "Rendering Video", 90)
        render_video(job_id, mode, aspect)
        set_status(job_id, 7, "Completed", 100, status="completed")
        # cleanup temp files
        d = job_dir(job_id)
        for f in ["audio.wav","original_audio.wav","dubbed.wav"]:
            p=os.path.join(d,f)
            if os.path.exists(p): os.remove(p)
        tts_dir = os.path.join(d,"tts")
        if os.path.isdir(tts_dir): shutil.rmtree(tts_dir, ignore_errors=True)
    except Exception as e:
        set_status(job_id, -1, "Failed", 0, status="failed", error=str(e)[:500])
