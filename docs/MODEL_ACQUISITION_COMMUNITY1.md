# ACQUISITION AUDIT — pyannote/speaker-diarization-community-1

Status: **AUDIT ONLY — NOT DOWNLOADED.** Awaiting explicit operator approval.
All figures below were obtained from Hugging Face's **public metadata API**
(no token required, no weights fetched). File *content* is gated; metadata is not.

## 1. Identity and pinned revision

| Field | Value |
|---|---|
| Repository | `pyannote/speaker-diarization-community-1` |
| **Exact revision (commit)** | `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee` |
| Last modified | 2025-09-29T08:43:24Z |
| Private | no |
| Gated | `auto` — access granted automatically once the conditions are accepted |
| Library | pyannote-audio (pipeline) |
| Popularity (30 d downloads) | ~4.42 M |

This revision is what `config.yaml` must pin as `diarization.model_revision`
so future runs cannot silently drift to a different upstream commit.

## 2. Required files, individual and total sizes

| File | Bytes | Storage |
|---|---:|---|
| `segmentation/pytorch_model.bin` | 5,906,507 | LFS (sha256 recorded) |
| `embedding/pytorch_model.bin` | 26,646,242 | LFS (sha256 recorded) |
| `plda/plda.npz` | 133,852 | LFS (sha256 recorded) |
| `plda/xvec_transform.npz` | 134,376 | LFS (sha256 recorded) |
| `config.yaml` | 444 | git blob |
| `README.md` | 9,978 | git blob |
| `embedding/README.md` | 938 | git blob |
| `plda/README.md` | 220 | git blob |
| `.gitattributes` | 1,571 | git blob |
| `diarization.gif` (documentation image, not needed at runtime) | 861,445 | LFS |
| **TOTAL (whole repo)** | **33,695,573 (~33.7 MB)** | |
| **Weights + config only** (excluding the GIF and READMEs) | **~32.8 MB** | |

**Key finding:** the pipeline is **self-contained**. Segmentation, speaker
embedding and PLDA weights all live inside this one repository. Unlike
pyannote 3.1 — which additionally required `pyannote/segmentation-3.0` — **no
second gated repo needs to be acquired**. Total acquisition is one repo, ~34 MB.

## 3. License and gated-access terms

- **License: CC-BY-4.0** (permissive, commercial use allowed, attribution
  required). This is materially better than the NLLB CC-BY-NC-4.0 constraint
  and imposes no new non-commercial restriction.
- Gating is `auto`: acceptance of the conditions grants access immediately.
- The gate collects **Company/university** and **Use case** and includes an
  opt-in style notice that pyannote may email occasionally about models and
  premium offerings. The card states the pipeline "is released under the
  CC-BY-4.0 license and will always remain freely accessible."
- Attribution obligation → to be recorded in `THIRD_PARTY_NOTICES.md` when the
  model is enabled (papers: arXiv:2104.03603, arXiv:2111.14448).

## 4. Cache destination and current state

| Field | Value |
|---|---|
| `HF_HOME` set | no |
| Cache root | `C:\Users\USER\.cache\huggingface` |
| Hub dir | `C:\Users\USER\.cache\huggingface\hub` (exists) |
| Target dir | `models--pyannote--speaker-diarization-community-1` |
| **Currently present** | **NO — no pyannote model is cached at all** |
| Free disk on `C:\` | 736.8 GB (ample for 34 MB) |

Note: the cache is on `C:\`, outside the OneDrive tree — no sync risk.

## 5. Integrity metadata

Captured **before** download into
`workers/diarization_worker/model_integrity_community1.json`:
- 10 files recorded; **5 LFS files carry upstream SHA-256 digests**; 5 small
  text files carry git blob OIDs.
- `huggingface_hub` verifies LFS SHA-256 during download and fails the fetch on
  mismatch; the post-download verification step re-checks the cached bytes
  against this manifest independently.

## 6. RAM / VRAM estimates

These are engineering estimates for planning, to be replaced by measurements
from the first real run (the worker already instruments
`torch.cuda.max_memory_allocated`).

| Resource | Estimate | Basis |
|---|---|---|
| Disk | ~34 MB (plus ~34 MB transient blob staging) | measured file sizes |
| Model weights in VRAM | ~35–60 MB | 32.6 MB of fp32 parameters + CUDA overhead |
| Peak VRAM during inference | ~0.7–1.5 GB for a few-minute file | activations dominate; scales with window batching, not file length |
| Host RAM | waveform ~3.8 MB per audio-minute (16 kHz mono float32) + ~1–2 GB torch/CUDA context | pyannote requires the full waveform in memory for one call |
| 8 GB RTX 5060 headroom | comfortable — the worker runs **alone**; ClearVoice is evicted first | existing VRAM policy |

## 7. Secure download procedure (to run only after approval)

1. Token is read from `hf_token.txt` (gitignored) or `HF_TOKEN`; it is passed
   to the child process **via environment only** — never argv, never JSON,
   never logged, never committed.
2. Download **the pinned revision only**, over HTTPS, in the isolated worker
   environment:
   `huggingface_hub.snapshot_download(repo_id=..., revision="3533c8cf…", token=<env>)`
   with `allow_patterns` excluding `diarization.gif`.
3. `huggingface_hub` verifies each LFS object's SHA-256 during transfer.
4. No other repository is fetched (the pipeline is self-contained — §2).
5. Nothing is written into the git working tree; artifacts land in the HF cache.

## 8. Offline verification procedure

1. Re-hash every cached file and compare against
   `model_integrity_community1.json`; any mismatch fails loudly.
2. Pin `diarization.model_revision: 3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`
   in `config.yaml`.
3. Re-run with the network disabled at the process level
   (`HF_HUB_OFFLINE=1`, already set by the worker when `offline: true`) and
   confirm a real inference succeeds — only then may the model be described as
   working offline (per the standing "do not claim offline until proven" rule).
4. `python -m aegis doctor` should then report `model cache: present (… @ 3533c8cf…)`
   and `offline readiness: cached weights present`.

## 9. Rollback / removal

```powershell
# remove the cached model (frees ~34 MB); nothing else depends on it
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub\models--pyannote--speaker-diarization-community-1"
```
Then set `diarization.enabled: false` (or leave it enabled to fall back).
The pipeline reverts to `fallback_single_speaker` with an explicit warning —
no code change required, no other component affected. To remove the entire
worker environment as well:
`Remove-Item -Recurse -Force "$env:LOCALAPPDATA\AegisXPrime\envs\diarization"`.

## 10. ACCEPTANCE RESULTS (2026-07-20)

Download: pinned revision `3533c8cf…`, GIF excluded, 5.86 s. Cache verified
independently against the integrity manifest: **9 files verified, 0 mismatched,
0 missing** — all four LFS weight objects SHA-256 match.

Network-disabled inference (`HF_HUB_OFFLINE=1`) on a synthetic two-speaker
fixture (two distinct Windows TTS voices, 34.14 s, deliberate overlap):

| Metric | Raw fixture | ClearVoice-enhanced (`voice_16k.wav`) |
|---|---|---|
| genuine_pyannote | **true** | **true** |
| speakers detected | **2** ✅ | **1** ❌ |
| regular turns | 14 | 12 |
| exclusive turns | 12 | 12 |
| overlap regions | 2 | 0 |
| device / time / peak VRAM | cuda / 11.65 s / 2723.6 MB | cuda / 9.22 s / 2723.6 MB |

Structural checks on the raw-fixture result: non-empty, monotonically ordered,
all turns within [0, duration], no duplicate turns, exclusive ≠ regular,
overlap preserved — **all pass**.

### Open finding: enhancement destroys speaker separability

The pipeline feeds diarization the **ClearVoice-enhanced** `voice_16k.wav`.
With identical model, revision, settings and offline path, that input yields
**1 speaker instead of 2, and loses both overlap regions**. The raw audio
yields the correct 2 speakers. Enhancement — not diarization — is the cause.

This is precisely the effect the specification warns about (§8.1: enhancement
artifacts degrading downstream tasks) and is the motivation for Phase 2's
evidence-gated branch selection. It is recorded here, **not fixed**, because
changing which audio branch diarization consumes is Phase 2 routing work and
was not in scope for this pass.

Consequence for the acceptance gate: real diarization **is functional and
genuine**, but the "≥2 speakers where the reference has two" criterion is met
only on the raw branch, not through the current pipeline routing.

## 11. Residual risks

- The gate collects company/use-case data and permits occasional emails — an
  operator/privacy decision, not a technical one.
- `torchcodec`'s native decoder remains unavailable in the worker environment;
  irrelevant here because the worker passes pyannote a preloaded waveform, but
  it means pyannote could not decode a file path directly if that path were
  ever used.
- Accuracy is **not** established by acquisition; a real two-speaker fixture
  must still pass the functional gate before diarization may be called working.
