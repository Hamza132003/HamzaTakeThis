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

## Store location (risk R-8)

The duplicate index lives in the artifact store, resolved as:
1. `storage.artifact_dir` from config, if set;
2. else `%LOCALAPPDATA%\AegisXPrime\store` (on this machine:
   `C:\Users\<user>\AppData\Local\AegisXPrime\store`) — a non-synced location; the default is
   **never** inside OneDrive. Channel assets and manifests are written next to the legacy
   outputs (`outputs/<stem>/ingest/`) in Phase 1 so the dashboard's relative-path serving keeps
   working; NOTE: if the repo lives in OneDrive, those outputs are synced — operators handling
   sensitive audio should set `output_dir`/`--outdir` to a non-synced path (see PRIVACY.md).
   The Phase 12 DAG store will move all artifacts to the non-synced store.

## Derived asset layout and format

```
outputs/<stem>/
  audio_16k_mono.wav      (legacy, unchanged, pcm_s16le)
  audio_48k_mono.wav      (legacy, unchanged, pcm_s16le)
  ingest_manifest.json    (IngestManifest, schema v1, atomic)
  ingest/
    channel_0.wav …       (per source channel, native rate, pcm_f32le)
    mono_mix.wav          (mean of channels, scale 1/C)
    mid.wav, side.wav     (stereo only: (L±R)/2)
```

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

## Failure behavior

Ingest failures (decode errors, malformed media) set `failure_reason` on the manifest and log
a warning; the legacy extraction path is unaffected. `ingest.enabled: false` disables the
whole additive stage.
