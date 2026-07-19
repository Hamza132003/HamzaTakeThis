# THIRD_PARTY_NOTICES.md

This project depends on third-party software and model weights. Licenses are recorded
per component in MODEL_LICENSES.yaml (models, code/weights separately) and in
docs/MODEL_LICENSE_MATRIX.md (analysis). Python package licenses are those declared by
each distribution in `requirements.lock.txt`.

Key notices:

- **NLLB-200 distilled weights (Meta AI)** — CC-BY-NC-4.0, **non-commercial**. Used only
  in the research/local-evaluation profile. Attribution: NLLB Team et al., "No Language
  Left Behind: Scaling Human-Centered Machine Translation", 2022.
- **Whisper large-v3 (OpenAI, via Systran/faster-whisper-large-v3)** — MIT.
- **pyannote.audio / speaker-diarization-community-1** — MIT (HF user agreement applies
  to the gated weights).
- **ClearVoice / MossFormer2 (Alibaba Speech Lab)** — code Apache-2.0; weight license
  verification pending (see MODEL_LICENSES.yaml).
- **SpeechBrain models (SepFormer, ECAPA-TDNN)** — Apache-2.0.
- **PANNs (CNN14)** — code MIT; checkpoint license verification pending.
- **Qwen3-4B-Instruct-2507 (Alibaba)** — Apache-2.0.
- **Silero VAD** (bundled in faster-whisper) — MIT.
- **superb/wav2vec2-base-superb-er** — Apache-2.0.
- **ffmpeg (via imageio-ffmpeg)** — LGPL/GPL depending on build flags; exact build
  configuration must be reviewed before any binary redistribution (release is currently
  blocked regardless — see RELEASE_BLOCKED.md).
- Python libraries: torch (BSD-3), transformers (Apache-2.0), faster-whisper (MIT),
  CTranslate2 (MIT), librosa (ISC), soundfile (BSD-3), Flask (BSD-3), numpy (BSD-3),
  scipy (BSD-3), and transitive dependencies as declared in their distributions.

This file is a notice inventory, not a grant of license for this repository itself
(project license: deferred — docs/LICENSE_DECISION.md).
