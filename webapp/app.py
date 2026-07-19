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

from pipeline.runner import process_file, JobCancelled
from pipeline.utils import resolve_device, log, free_cuda

DIST = ROOT / "webapp" / "frontend" / "dist"
UPLOADS = ROOT / "uploads"
app = Flask(__name__, static_folder=str(DIST), static_url_path="")

MEDIA_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpeg", ".mpg",
             ".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus",
             ".wma", ".aiff", ".aif", ".amr", ".3gp"}
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def output_root() -> Path:
    return (ROOT / load_config().get("output_dir", "outputs")).resolve()


# ------------------------------------------------------------------ job state
JOB = {"running": False, "total": 0, "done": 0, "current": "", "stage": "",
       "step": 0, "steps": 0, "substep": 0.0, "log": [], "folder": "",
       "cancel_requested": False, "version": 0}
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


def _cancel_requested() -> bool:
    with LOCK:
        return JOB["cancel_requested"]


def _worker(files, cfg, device):
    for f in files:
        if _cancel_requested():
            _log("[cancelled] batch stopped before next file")
            break
        _bump(current=Path(f).name)
        _log(f"> {Path(f).name}")

        def progress(stage, step, steps, frac=0.0):
            _bump(stage=stage, step=step, steps=steps,
                  substep=round(float(frac), 3))

        try:
            process_file(f, cfg, device, progress=progress,
                        should_cancel=_cancel_requested)
            _log(f"[ok] {Path(f).name}")
        except JobCancelled:
            _log(f"[cancelled] {Path(f).name}")
            free_cuda()
            break
        except Exception as e:
            _log(f"[fail] {Path(f).name}: {e}")
        with LOCK:
            JOB["done"] += 1
            JOB["version"] += 1
    _bump(running=False, current="", stage="finished", cancel_requested=False)


def _truthy(v) -> bool:
    """Robust bool coercion: the JSON path (folder jobs) sends real
    booleans, but the multipart path (mic recordings) round-trips
    everything through FormData as strings, where bool("false") is True."""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() not in ("false", "0", "", "no")


def _make_cfg(data: dict) -> tuple[dict, str]:
    """Apply per-job overrides from the UI onto the YAML config."""
    cfg = load_config()
    cfg.setdefault("transcription", {})["language"] = data.get("language", "auto")
    if data.get("hf_token"):
        cfg.setdefault("diarization", {})["hf_token"] = str(data["hf_token"]).strip()
    if "separate_overlap" in data:
        cfg.setdefault("speakers", {})["separate_overlap"] = _truthy(data["separate_overlap"])
    if "ai_summary" in data:
        cfg.setdefault("summary", {})["enabled"] = _truthy(data["ai_summary"])
    device = resolve_device(data.get("device", cfg.get("device", "auto")))
    return cfg, device


def _start_job(files, cfg, device, folder=""):
    with LOCK:
        JOB.update(running=True, total=len(files), done=0, current="",
                   stage="", step=0, steps=0, substep=0.0, log=[],
                   folder=folder, cancel_requested=False)
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
        if fpath.is_file():
            if fpath.suffix.lower() not in MEDIA_EXT:
                return jsonify({"error": f"Not a supported audio/video file: {folder}"}), 400
            files = [str(fpath)]
            folder = ""
        elif fpath.is_dir():
            files = sorted(str(p) for p in fpath.iterdir()
                           if p.suffix.lower() in MEDIA_EXT)
            if not files:
                return jsonify({"error": f"No audio/video files found in {folder}"}), 400
        else:
            return jsonify({"error": f"Not a folder or file: {folder or '(empty)'}"}), 400

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


@app.post("/api/cancel")
def api_cancel():
    """Best-effort cancel: takes effect at the next stage boundary (or
    before the next file in a batch), not mid-inference."""
    with LOCK:
        if not JOB["running"]:
            return jsonify({"error": "No job is running."}), 400
        JOB["cancel_requested"] = True
        JOB["version"] += 1
    _log("Cancel requested…")
    return jsonify({"cancelling": True})


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


_browse_lock = threading.Lock()


@app.post("/api/browse-folder")
def api_browse_folder():
    """Open the native Windows folder picker on the server machine and
    return the chosen path. Runs as a separate PowerShell/WinForms process
    (not in-process Tkinter) so it reliably comes to the front and never
    races with itself if the button is clicked more than once."""
    import subprocess
    if not _browse_lock.acquire(blocking=False):
        return jsonify({"error": "A folder dialog is already open."}), 409
    try:
        # Repurposes the modern Explorer-style OpenFileDialog (WinForms' own
        # FolderBrowserDialog is the old tree-view style). Recordings are
        # visible and directly pickable; keeping the default "Select this
        # folder" pseudo-filename and clicking Open returns the folder
        # instead, so one dialog covers both cases.
        exts = ";".join(f"*{e}" for e in sorted(MEDIA_EXT))
        ps_script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "$dlg = New-Object System.Windows.Forms.OpenFileDialog;"
            "$dlg.ValidateNames = $false;"
            "$dlg.CheckFileExists = $false;"
            "$dlg.CheckPathExists = $true;"
            "$dlg.FileName = 'Select this folder';"
            "$dlg.Title = 'Pick a recording, or click Open to use the whole folder';"
            f"$dlg.Filter = 'Audio & video|{exts}|All files|*.*';"
            "$owner = New-Object System.Windows.Forms.Form -Property @{TopMost=$true};"
            "if ($dlg.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) "
            "{ if (Test-Path -LiteralPath $dlg.FileName -PathType Leaf) "
            "{ Write-Output $dlg.FileName } else { Split-Path $dlg.FileName -Parent } }"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-STA", "-Command", ps_script],
            capture_output=True, text=True, timeout=300,
        )
        path = result.stdout.strip()
        return jsonify({"folder": path})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Folder dialog timed out."}), 504
    finally:
        _browse_lock.release()


@app.get("/api/recordings")
def api_recordings():
    return jsonify(list_recordings())


@app.get("/api/recording/<name>")
def api_recording(name):
    rj = output_root() / name / "report.json"
    if not rj.exists():
        abort(404)
    return jsonify(json.loads(rj.read_text(encoding="utf-8")))


@app.delete("/api/recording/<name>")
def api_delete_recording(name):
    """Permanently remove one processed result's output folder."""
    root = output_root()
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        abort(400)
    target = root / name
    if not target.is_dir() or target.resolve().parent != root:
        abort(404)
    import shutil
    shutil.rmtree(target)
    return jsonify({"deleted": name})


@app.get("/media/<name>/<path:relpath>")
def media(name, relpath):
    return send_from_directory(str(output_root() / name), relpath)


@app.get("/")
def index():
    if not (DIST / "index.html").exists():
        return ("<h1>UI not built</h1><p>Run <code>build_ui.ps1</code> first "
                "(needs Node).</p>", 200)
    return send_from_directory(str(DIST), "index.html")


class _Tee:
    """Mirror a stream into the server log so output survives a crash or a
    closed console window."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self._streams[0], name)


def main():
    logf = open(ROOT / "server.log", "a", encoding="utf-8",
                errors="replace", buffering=1)
    import faulthandler
    faulthandler.enable(file=logf)      # captures hard native crashes too
    sys.stdout = _Tee(sys.stdout, logf)
    sys.stderr = _Tee(sys.stderr, logf)
    log(f"--- server start {time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    url = "http://127.0.0.1:5000"
    log(f"Dashboard: {url}")
    try:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    except Exception:
        pass
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)


if __name__ == "__main__":
    main()
