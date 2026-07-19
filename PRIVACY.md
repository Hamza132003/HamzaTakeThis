# PRIVACY.md — AEGIS-X PRIME

- All inference is local. After one-time model downloads the pipeline runs offline; no
  audio, transcript, or metadata leaves the machine as part of processing.
- Only legally acquired, documented audio may be processed for evaluation. Private,
  intercepted, classified, or unauthorized audio is prohibited, including in tests
  (specification §1).
- Processed outputs (`outputs/`), uploads (`uploads/`), and raw recordings are gitignored
  and stay on the local machine. They may contain personal data: treat the output
  directories as sensitive.
- **OneDrive caveat (active risk R-8):** this repository currently resides in a
  OneDrive-synced folder, so gitignored outputs under the repo tree may still be synced
  to Microsoft's cloud by OneDrive itself. Until the artifact store moves to a non-synced
  path (`storage.artifact_dir`, Phase 12), operators handling sensitive audio should
  process it to a non-synced output directory (`--outdir`) or pause OneDrive sync.
- Retention: no automatic retention policy exists yet (Phase 12 adds configurable
  retention and secure deletion). Deleting a result from the UI removes its output folder.
- Credentials: HF token is user-held; never committed; redact from logs.
