# INGEST.md — Phase 1 immutable ingest: guarantees, layout, policies

## Immutability guarantees

- Sources are opened **read-only**; nothing in ingest writes, renames, resamples, truncates
  or normalizes an original (regression-tested byte equality before/after).
- SHA-256 (primary content ID) + SHA-512 (secondary evidentiary digest) are computed in ONE
  streaming pass (8 MiB chunks) over the raw container bytes **before any decoding**.
- Content ID: `src-<sha256[:16]>`; derived: `drv-<sha256[:16]>` (content-addressed).
- Derived artifacts are written temp-file-then-`os.replace` (atomic on NTFS/POSIX) and
  re-hashed from disk *after* the final rename — an artifact without its final hash does not
  exist as far as the manifest is concerned.

## Duplicate semantics (Phase 1)

- **Source duplicate** = exact-byte SHA-256 equality, detected against the store's
  `source_index.json`; the manifest records `duplicate_of=<first-seen asset_id>` plus a
  warning ("not an independent recording"). Same bytes at a different path → duplicate.
- **Different container bytes with acoustically equivalent content are NOT duplicates in
  Phase 1** (acoustic fingerprinting is explicitly out of scope; no scaffold enabled).
- **Duplicate derived artifacts**: content addressing makes them self-identifying — e.g. for a
  stereo source, `mid` = (L+R)/2 and `mono_mix` = mean(L,R) are the same samples, so they
  legitimately share sha256/asset_id while remaining distinct roles/files.

## Canonical store location (risk R-8 — resolved in the Phase 1 repair pass)

Canonical artifacts live in the artifact store, **never** in the recording output tree.
Resolution order:
1. `storage.artifact_dir` from config, if set;
2. else `%LOCALAPPDATA%\AegisXPrime\store` (on this machine
   `C:\Users\<user>\AppData\Local\AegisXPrime\store`) — a non-synced location; the default is
   never inside OneDrive.

Canonical layout is content-addressed by source SHA-256 (`canonical_dirs()` validates the
digest and rejects anything that would resolve outside the store root — traversal attempts,
malformed or non-hex ids):

```
<store>/sources/<sha[:2]>/<sha>/
    ingest_manifest.json     (IngestManifest, schema v2, atomic)
    assets/
      channel_0.wav …        (per source channel, native rate, pcm_f32le)
      mono_mix.wav           (mean of channels, scale 1/C)
      mid.wav, side.wav      (stereo only: (L±R)/2)
```

Identical source bytes resolve to the same canonical directory, so duplicates share one
canonical source dir rather than creating a second copy.

## Recording output directory (compatibility path, unchanged for the dashboard)

```
outputs/<stem>/
  audio_16k_mono.wav      (legacy, unchanged, pcm_s16le)
  audio_48k_mono.wav      (legacy, unchanged, pcm_s16le)
  ingest_pointer.json     (small IngestPointer, ~460 bytes)
  …existing report/spectrogram/voice artifacts…
```

The pointer holds **store-relative** paths only (`sources/<ab>/<sha>/…`) — deliberately no
absolute path, so it is portable and leaks no user-profile path. Readers resolve it against
`storage.artifact_dir` at read time. Four path kinds are kept distinct:

| Kind | Where | Portable? |
|---|---|---|
| canonical manifest path | `<store>/<store_relative_dir>/ingest_manifest.json` | yes (store-relative) |
| canonical artifact path | `DerivedAudio.rel_path` (`assets/…`), relative to the canonical dir | yes |
| local source path | `AudioAsset.source_path_local` | **no** — non-portable by contract |
| compatibility output path | `outputs/<stem>/…` legacy wavs + pointer | local |

## Schema history

- **v1** — first Phase 1 implementation; `rel_path` was relative to the recording output dir.
- **v2** (Phase 1 repair) — canonical store-relative artifacts, `store_relative_dir`,
  `stages_completed`, structured `failures`, `analysis_block_frames`, `analysis_policy`,
  and `IngestPointer`. v1 manifests remain readable and are never rewritten or deleted;
  there is no automatic migration because no v1 manifest exists outside development runs.

PCM format `pcm_f32le` (float32 WAV): lossless for the float matrix math, cannot clip on
write, readable by soundfile/librosa/ffmpeg. Trade-off: 2× the bytes of 16-bit PCM. Written
via `scipy.io.wavfile` deliberately — libsndfile embeds a wall-clock timestamp in float-WAV
PEAK chunks, which would break reproducible hashing.

Mid/side convention: mid=(L+R)/2, side=(L−R)/2 — the /2 bounds outputs at max(|L|,|R|), so
matrixing can never introduce clipping. Mono sources get NO fabricated mid/side. >2-channel
sources keep every channel plus a documented equal-weight downmix; nothing is discarded.
No gain or normalization is ever applied (`gain_adjusted=false` on every contract).

## Media probing

PyAV (structured libav bindings) — imageio-ffmpeg does **not** bundle ffprobe, so PyAV is the
reliable structured probe in this environment. Unavailable fields are `null` with an entry in
`unavailable_reasons`; nothing is guessed. Stream selection rule (recorded on the contract):
first audio stream in index order. Malformed inputs produce warnings + reasons, not fake
metadata.

## Contract policies

- Schema: `INGEST_SCHEMA_VERSION = 1`; every model carries `schema_version`.
- Unknown keys: **rejected** (`extra="forbid"`). Adding a field = schema-version decision +
  migration note here.
- No calibrated-confidence fields exist in ingest contracts (enforced by test).
- Portability/privacy: manifests reference artifacts by RELATIVE path + hash. The absolute
  source path appears only in `source_path_local`, explicitly non-portable — it may contain a
  user-profile path; strip it before sharing a manifest. Display names are sanitized
  (basename + control-character removal).
- Symlink policy: paths are used as given by the operator; ingest does not dereference or
  follow links beyond what the OS does for a normal file open, and never enumerates
  directories itself (the webapp's folder scan is unchanged from Phase 0).

## Bounded memory

The decoded recording is never materialized as one array.

- **Channel extraction** streams `BLOCK_FRAMES = 48_000` frames (1 s at 48 kHz) via
  `soundfile.blocks()` through the split/matrix into per-track incremental writers. Peak RAM
  is O(block × channels), independent of duration. Block size is recorded in the manifest
  (`analysis_block_frames`) and provably does not change artifact bytes (tested).
- **Diagnostics** (`analyse_path`) read only what they analyse: the whole file below
  `max_full_analysis_sec` (120 s → a documented ~46 MB bound at 48 kHz stereo float32),
  otherwise exactly three deterministic seek-read windows (start/middle/end,
  `sample_window_sec` each). `analysis_mode` and the policy string are persisted.

Measured on this machine (identical fixtures, one subprocess per measurement):

| Fixture | Before (whole-file load) | After (streaming) |
|---|---|---|
| 12 min stereo (276 MB) | 1158 MB peak | **69 MB** |
| 40 min stereo (922 MB) | 3738 MB peak | **71 MB** |

Peak stays flat as duration grows — bounded, not proportional.

## Failure behavior

Ingest never raises and never reports failure only as a log line. Each stage
(`hash`, `store`, `probe`, `channels`, `diagnostics`, `manifest_write`, `pointer_write`)
records a structured `IngestFailure`: stage, exception type, sanitized single-line message
(bounded length; never secrets or environment dumps), UTC timestamp, source content id,
recoverability, and whether the legacy path continued. `IngestManifest.failures` plus
`failure_reason` summarize them, and the compatibility façade always returns
`ingest_status` (`ok` | `degraded` | `failed` | `disabled`) which the runner carries into the
report for later phases. The legacy transcript pipeline continues in every case, and the log
states explicitly that evidence-grade ingest did not complete.

`ingest.enabled: false` disables the whole additive stage (status `disabled`).
