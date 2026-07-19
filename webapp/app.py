r"""Local backend: JSON API for the React dashboard + serves the built app.

Run:  .\.venv\Scripts\python.exe webapp/app.py
Then open http://127.0.0.1:5000  (auto-opens).

The React frontend lives in webapp/frontend and is built to
webapp/frontend/dist (see build_ui.ps1). This server exposes:
  GET  /                      -> the React app
  POST /api/process           -> start a batch job {folder | files, language, ...}
  POST /api/record            -> upload a mic recording (multipart) + process it
  GET  /api/status            -> current job progress (poll fallback)
  GET  /api/events            -> Server-Sent Events stream of job progress
  GET  /api/recordings        -> processed recordings (summary)
  GET  /api/recording/<name>  -> full report for one recording
  GET  /media/<name>/<path>   -> isolated audio + spectrogram files

Loaded AI models are cached in-process (pipeline.models.MANAGER), so the
second file/job skips the multi-GB cold loads.
"""
from __future__ import annotations

import sys
import json
import time
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from flask import (Flask, Response, request, jsonify, send_from_directory,
                   abort)

from pipeline.runner import process_file
from pipeline.utils import resolve_device, log

DIST = ROOT / "webapp" / "frontend" / "dist"
UPLOADS = ROOT / "uploads"
app = Flask(__name__, static_folder=str(DIST), static_url_path="")

MEDIA_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v",
             ".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus"}
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def output_root() -> Path:
    return (ROOT / load_config().get("output_dir", "outputs")).resolve()


# ------------------------------------------------------------------ job state
JOB = {"running": False, "total": 0, "done": 0, "current": "", "stage": "",
       "step": 0, "steps": 0, "substep": 0.0, "log": [], "folder": "",
       "version": 0}
LOCK = threading.Lock()


def _bump(**updates):
    with LOCK:
        JOB.update(updates)
        JOB["version"] += 1


def _log(msg: str):
    with LOCK:
        JOB["log"].append(msg)
        JOB["log"] = JOB["log"][-200:]
        JOB["version"] += 1


def _worker(files, cfg, device):
    for f in files:
        _bump(current=Path(f).name)
        _log(f"> {Path(f).name}")

        def progress(stage, step, steps, frac=0.0):
            _bump(stage=stage, step=step, steps=steps,
                  substep=round(float(frac), 3))

        try:
            process_file(f, cfg, device, progress=progress)
            _log(f"[ok] {Path(f).name}")
        except Exception as e:
            _log(f"[fail] {Path(f).name}: {e}")
        with LOCK:
            JOB["done"] += 1
            JOB["version"] += 1
    _bump(running=False, current="", stage="finished")


def _make_cfg(data: dict) -> tuple[dict, str]:
    """Apply per-job overrides from the UI onto the YAML config."""
    cfg = load_config()
    cfg.setdefault("transcription", {})["language"] = data.get("language", "auto")
    if data.get("hf_token"):
        cfg.setdefault("diarization", {})["hf_token"] = str(data["hf_token"]).strip()
    if "separate_overlap" in data:
        cfg.setdefault("speakers", {})["separate_overlap"] = bool(data["separate_overlap"])
    device = resolve_device(data.get("device", cfg.get("device", "auto")))
    return cfg, device


def _start_job(files, cfg, device, folder=""):
    with LOCK:
        JOB.update(running=True, total=len(files), done=0, current="",
                   stage="", step=0, steps=0, substep=0.0, log=[],
                   folder=folder)
        JOB["version"] += 1
    threading.Thread(target=_worker, args=(files, cfg, device),
                     daemon=True).start()


def list_recordings():
    root = output_root()
    out = []
    if root.exists():
        for d in sorted(root.iterdir()):
            rj = d / "report.json"
            if not rj.exists():
                continue
            try:
                r = json.loads(rj.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append({
                "name": d.name,
                "filename": r.get("filename", d.name),
                "num_speakers": r.get("num_speakers", "?"),
                "duration_sec": r.get("duration_sec", 0),
                "languages": r.get("languages_detected", []),
                "warnings": len(r.get("warnings", [])),
                "top_tags": [t["label"] for t in r.get("noise", {}).get("top_tags", [])[:3]],
                "thumb": r.get("spectrograms", {}).get("voice"),
            })
    return out


# ---------------------------------------------------------------------- API
@app.post("/api/process")
def api_process():
    if JOB["running"]:
        return jsonify({"error": "A batch is already running."}), 409
    data = request.get_json(force=True, silent=True) or {}

    files = []
    if data.get("files"):
        for f in data["files"]:
            p = Path(str(f).strip().strip('"')).expanduser()
            if p.is_file() and p.suffix.lower() in MEDIA_EXT:
                files.append(str(p))
        if not files:
            return jsonify({"error": "No valid audio/video files given."}), 400
        folder = ""
    else:
        folder = str(data.get("folder", "")).strip().strip('"')
        fpath = Path(folder).expanduser()
        if not fpath.is_dir():
            return jsonify({"error": f"Not a folder: {folder or '(empty)'}"}), 400
        files = sorted(str(p) for p in fpath.iterdir()
                       if p.suffix.lower() in MEDIA_EXT)
        if not files:
            return jsonify({"error": f"No audio/video files found in {folder}"}), 400

    cfg, device = _make_cfg(data)
    _start_job(files, cfg, device, folder)
    return jsonify({"started": True, "total": len(files)})


@app.post("/api/record")
def api_record():
    """Receive a browser MediaRecorder blob and run the full pipeline on it."""
    if JOB["running"]:
        return jsonify({"error": "A batch is already running."}), 409
    blob = request.files.get("file")
    if blob is None or not blob.filename:
        return jsonify({"error": "No audio uploaded."}), 400

    mimetype = (blob.mimetype or "").lower()
    ext = ".webm"
    for candidate in (".webm", ".ogg", ".mp4", ".m4a", ".wav"):
        if candidate.lstrip(".") in mimetype:
            ext = candidate
            break

    UPLOADS.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS / f"rec_{time.strftime('%Y%m%d_%H%M%S')}{ext}"
    blob.save(str(dest))
    log(f"Mic recording saved: {dest.name} ({dest.stat().st_size} bytes)")

    cfg, device = _make_cfg(request.form.to_dict())
    _start_job([str(dest)], cfg, device)
    return jsonify({"started": True, "name": dest.stem})


@app.get("/api/status")
def api_status():
    with LOCK:
        return jsonify(dict(JOB))


@app.get("/api/events")
def api_events():
    """SSE stream: emits the job dict whenever it changes (500 ms checks)."""
    def stream():
        last = -1
        idle_beats = 0
        while True:
            with LOCK:
                version = JOB["version"]
                snapshot = dict(JOB)
            if version != last:
                last = version
                idle_beats = 0
                yield f"data: {json.dumps(snapshot)}\n\n"
            else:
                idle_beats += 1
                if idle_beats % 30 == 0:      # keep-alive every ~15 s
                    yield ": keep-alive\n\n"
                if not snapshot["running"] and idle_beats > 240:
                    break                     # close idle streams after ~2 min
            time.sleep(0.5)
    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.get("/api/recordings")
def api_recordings():
    return jsonify(list_recordings())


@app.get("/api/recording/<name>")
def api_recording(name):
    rj = output_root() / name / "report.json"
    if not rj.exists():
        abort(404)
    return jsonify(json.loads(rj.read_text(encoding="utf-8")))


@app.get("/media/<name>/<path:relpath>")
def media(name, relpath):
    return send_from_directory(str(output_root() / name), relpath)


@app.get("/")
def index():
    if not (DIST / "index.html").exists():
        return ("<h1>UI not built</h1><p>Run <code>build_ui.ps1</code> first "
                "(needs Node).</p>", 200)
    return send_from_directory(str(DIST), "index.html")


def main():
    url = "http://127.0.0.1:5000"
    log(f"Dashboard: {url}")
    try:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    except Exception:
        pass
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)


if __name__ == "__main__":
    main()
