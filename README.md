# Voice Isolator

Fully **local / offline** pipeline that takes a video, audio file, or a live
microphone recording and:

1. **Isolates the speech** from noise at full 48 kHz band (ClearVoice
   MossFormer2 speech enhancement) — you get a clean `voice.wav` and a
   `noise.wav` that is the *same model's* residual, so the voice does not
   bleed into the noise track.
2. **Identifies the noise** — what each sound is and when it occurs
   (PANNs / AudioSet, 527 classes).
3. **Counts the speakers** and labels who spoke when (pyannote diarization),
   **un-mixing simultaneous speech** into per-speaker tracks (MossFormer2
   separation + ECAPA speaker matching).
4. **Transcribes** each speaker (faster-whisper large-v3) with **word-level
   timestamps + confidence**, auto-detecting **Farsi / Hebrew / Arabic /
   English** (or forced). Decoding is gated to detected speech regions and
   filtered with multiple anti-hallucination checks.
5. **Translates** everything into **English** (Whisper) and **Arabic**
   (NLLB-200-1.3B, translating the *source* text directly so a bad English
   guess can't poison the Arabic).
6. **Sentiment** per speaker from **vocal tone** (language-agnostic).
7. Writes a **`report.md` + `report.json`** (with per-word timings).

No cloud APIs are used at runtime. The only network step is a **one-time model
download** (including a free HuggingFace token for diarization). After that it
runs with no internet.

---

## What this can and cannot do (read this)

- **"Where" the noise is** = *what it is* + *when in the timeline*. Compass direction
  is **not** recoverable from a normal mono/stereo recording — it needs a multichannel
  mic array. This tool reports the sound type and its time range.
- **Distant, overlapping speech in two languages is the hard case.** Everything runs,
  but transcription/translation quality is bounded by how audible the speech is, not by
  the code. Segments the pipeline distrusts are marked **⚠ low confidence** in the UI
  instead of being silently kept — treat those as leads, not ground truth.
- **Sentiment is tone-based**, so it survives translation and works on Farsi/Hebrew.

---

## Setup (Windows, one time)

```powershell
cd "C:\Users\Waleed.Alawneh\Voice Isolator Code"
./setup.ps1
```

This creates `.venv`, installs a CUDA build of PyTorch (for your RTX A1000 6GB),
installs the rest, and runs `check_env.py`.

### HuggingFace token (REQUIRED for speaker counting / overlap un-mixing)

Only the pyannote diarization models are gated; everything else downloads
without a token. One time, free, while logged in at hf.co:

1. Accept the terms at **both** pages (forgetting the second one causes a 401):
   - https://hf.co/pyannote/speaker-diarization-3.1
   - https://hf.co/pyannote/segmentation-3.0
2. Create a token at https://hf.co/settings/tokens — either a classic **read**
   token, or a fine-grained token with *"Read access to contents of all public
   gated repos you can access"*.
3. Provide it any of these ways:
   ```powershell
   $env:HF_TOKEN = "hf_xxx"          # per session
   ```
   or paste it into `config.yaml` under `diarization.hf_token`, or into the
   token field in the dashboard.

Without a token the tool still runs, but treats the audio as a **single
speaker** and shows a warning banner on every report.

### Optional: DeepFilterNet fallback enhancer
`deepfilternet` is a lighter 48 kHz fallback for the separation stage, but its
wheel needs a Rust toolchain to build on Python 3.12. The pipeline already
falls back to SpeechBrain/noisereduce automatically, so this is optional.

---

## Usage — React dashboard (recommended)

```powershell
./dashboard.ps1
```
On first launch this builds the React UI (needs **Node.js** — one time), then
starts the server and opens `http://127.0.0.1:5000`.

- **📁 Folder tab** — paste a folder of recordings and batch-process it.
- **🎙️ Microphone tab** — record straight from the mic (raw, no browser
  processing); pressing **Stop** runs the full-quality pipeline automatically.
- Language: **Auto** detects Farsi / Hebrew / Arabic / English, or force one.
  Arabic = transcribe only (no translation).
- Progress streams live (SSE); loaded models are cached in the server process,
  so the second file/batch skips the multi-GB cold loads.
- Per recording you get: mel spectrograms (full 48 kHz band), playable
  isolated voice/noise, per-speaker cleaned tracks, the noise timeline, and a
  transcript with **Original / English / Arabic / Words** views — in the Words
  view every word shows its confidence and clicking it plays the isolated
  voice from that instant.

The UI is a lightweight Vite + React app in `webapp/frontend` (pure CSS
design, no WebGL/animation libraries — it stays off the GPU the models need).
Rebuild after UI edits with `./build_ui.ps1`.

## Usage — command line

```powershell
.\.venv\Scripts\python.exe main.py "path\to\recording.mp4" --language fa
```

Options:
```
--device cpu|cuda|auto     force compute device (default: auto)
--outdir results           output folder (default: outputs)
--language auto|fa|he|ar|en
--hf-token hf_xxx          diarization token inline
--skip noise emotion       skip specific stages
```

Output goes to `outputs/<filename>/`:
```
voice.wav        isolated speech (48 kHz)
voice_16k.wav    16 kHz copy used by the speech models
noise.wav        isolated noise (the enhancer's residual)
report.md        human-readable report (incl. word-level table)
report.json      structured data (per-segment speaker/lang/text/english/
                 arabic/words/quality/sentiment)
```

---

## Configuration

Everything is tunable in `config.yaml`: separation method and chunking,
loudness target (EBU R128), model sizes, device, hallucination-rejection
thresholds, expected languages, Whisper `compute_type` (use `int8_float16` on
the 6 GB GPU, `int8` on CPU), translation model/beams, etc.

## First run is slow
Models download on first use (Whisper large-v3 ~3 GB, NLLB-1.3B ~5.5 GB,
MossFormer2 ~1 GB, PANNs ~300 MB, pyannote ~30 MB). Subsequent runs are
offline and much faster — and within a dashboard session, models stay loaded
between files.

## VRAM note (6 GB GPUs)
The pipeline keeps **one heavy model at a time** on the GPU and evicts Whisper
before the NLLB translator loads — they cannot coexist in 6 GB. This is
automatic; forcing `--device cpu` avoids it entirely (much slower).
