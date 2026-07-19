# MODEL LICENSE MATRIX — AEGIS-X PRIME

Rule (spec §1): code license, weight license, dataset restrictions, attribution, and commercial-use
status are recorded **separately**. "Downloadable" ≠ "open source".

Verification key:
- **VERIFIED-LOCAL** — license file present in the local cache / package metadata inspected on this machine.
- **KNOWN** — license widely documented; must still be re-verified against the pinned revision at acquisition time (Phase 0/4).
- **UNVERIFIED** — must be checked before any download or enablement. Do not enable until verified.

No model terms are accepted on the operator's behalf (spec §26). Gated models remain the operator's action.

## A. Models currently used by the pipeline

| Component | Code license | Weight license | Gated? | Commercial use | Status / notes |
|---|---|---|---|---|---|
| faster-whisper (CTranslate2) | MIT | — | no | yes | KNOWN |
| Whisper large-v3 weights (Systran/faster-whisper-large-v3) | — | MIT | no | yes | KNOWN |
| ClearVoice code | Apache-2.0 | — | no | yes | KNOWN |
| MossFormer2_SE_48K / MossFormer2_SS_16K weights | — | Apache-2.0 (ModelScope publication) | no | likely yes | **UNVERIFIED** — confirm per-checkpoint license at pinned revision |
| pyannote.audio code | MIT | — | no | yes | KNOWN |
| pyannote speaker-diarization-community-1 weights | — | MIT (with HF gating/user-agreement step) | **yes** | yes per license; gating terms apply | KNOWN; token stays user-held |
| **NLLB-200-distilled (1.3B & 600M) weights** | HF transformers Apache-2.0 | **CC-BY-NC-4.0** | no | **NO — non-commercial only** | KNOWN. **Headline gate: the shipped translation stage is non-commercial as-is.** Deployment posture decision required (R-2) |
| PANNs (panns_inference code) | MIT | CNN14 checkpoints — publication states MIT/Apache lineage | no | probably yes | **UNVERIFIED** — also note insecure HTTP label download in the package (fix in Phase 13) |
| SpeechBrain (sepformer-wham16k/whamr16k, ECAPA voxceleb) | Apache-2.0 | Apache-2.0 (VoxCeleb-trained ECAPA: dataset terms worth review) | no | yes (dataset caveat) | KNOWN / dataset caveat UNVERIFIED |
| noisereduce | MIT | — | no | yes | KNOWN |
| superb/wav2vec2-base-superb-er (emotion) | — | Apache-2.0 | no | yes | KNOWN; scientific validity, not license, is the blocker (defect 3.11) |
| Qwen3-4B-Instruct-2507 | — | Apache-2.0 | no | yes | KNOWN |
| Silero VAD (bundled inside faster-whisper) | MIT | MIT | no | yes | KNOWN |
| imageio-ffmpeg (bundled ffmpeg build) | BSD-2 wrapper | ffmpeg LGPL/GPL build flags | no | depends on build | **UNVERIFIED** — record exact build config in THIRD_PARTY_NOTICES |

## B. Planned additions (Phase 4+), all acquisition-gated

| Component | Expected code / weight license | Commercial | Status |
|---|---|---|---|
| Omnilingual ASR CTC 300M (Meta) | code Apache-2.0 / weights — reported permissive | TBD | **UNVERIFIED — verify he+fa coverage AND weight license before download; operator approves download** |
| Omnilingual ASR LLM 7B (rescue) | same family | TBD | UNVERIFIED; hardware-excluded on 8 GB anyway |
| OWSM-CTC v4 1B | ESPnet Apache-2.0 / weights CC-BY-4.0 (reported) | yes if CC-BY | UNVERIFIED; challenger only, benchmark-gated |
| SeamlessM4T v2 | code MIT-ish / weights **CC-BY-NC-4.0** (reported) | **no** | UNVERIFIED; research-profile only if confirmed NC |
| DiariZen | weights **non-commercial** (stated by spec) | **no** | research-profile gate mandatory; disabled elsewhere |
| NVIDIA NeMo Sortformer/MSDD | NeMo Apache-2.0 / weight terms vary | TBD | UNVERIFIED; optional |
| DeepFilterNet3 | MIT/Apache dual | yes | KNOWN (code); **build blocker**: no py3.12 Windows wheel without Rust toolchain |
| Hebrew specialist (ivrit.ai-family fine-tune) | model-dependent | TBD | UNVERIFIED; also a DATA license question |
| Persian specialist (PSRB-family) | model-dependent | TBD | UNVERIFIED; also a DATA license question |

## C. Project-level

- **The repository itself has NO license.** All redistribution/packaging blocked until the owner decides (license decision record to be created in Phase 0; owner choice required — spec §3.13).
- THIRD_PARTY_NOTICES.md, MODEL_LICENSES.yaml, MODEL_MANIFEST.yaml (with pinned revisions + SHA-256 of weight files) are Phase 0 deliverables.
- Any model whose license conflicts with likely deployment gets its adapter implemented behind a disabled license gate with documented alternatives (spec §26).
