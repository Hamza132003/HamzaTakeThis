# SECURITY.md — AEGIS-X PRIME security posture

Phase 0 statement; hardened in Phase 13.

## Current posture (verified 2026-07-19)

- The dashboard listens on **127.0.0.1:5000 only**, no authentication. It must not be
  exposed beyond localhost without adding authentication and CSRF protection first.
- Secrets: the Hugging Face token lives in `hf_token.txt` (gitignored), the `HF_TOKEN`
  env var, or the UI field — never in Git. `tools/secret_scan.py` scans tracked files
  and runs in CI.
- Private audio: `outputs/`, `uploads/`, `recording/`, and common audio extensions are
  gitignored. Personal recordings never enter Git or evaluation sets without consent.
- Delete endpoint (`DELETE /api/recording/<name>`) validates against path traversal and
  only removes direct children of the configured output root.
- Model acquisition: currently via HF hub over HTTPS, except the `panns_inference` label
  CSV which the upstream package fetches over plain HTTP — known issue, replaced by a
  checksummed HTTPS acquisition step in Phase 13.
- `trust_remote_code=True` is not used anywhere and remains prohibited without a reviewed,
  pinned, allowlisted revision.
- Serialization: prefer `safetensors`; quarantine other formats (Phase 13 enforcement).
- GPU-compatibility failures are now **loud** (see `aegis doctor` and
  `pipeline.utils.resolve_device`): an installed torch build that cannot execute on the
  detected GPU aborts before pipeline execution instead of silently degrading.

## Threat-model headlines (full model in Phase 13)

Supply chain (floating revisions → pinned in MODEL_MANIFEST.yaml; lockfile added),
credential leakage (scan + gitignore + redaction), evidence integrity (hashing lands
Phase 1/13), local-host exposure (bind guard), malicious media inputs (decode hardening
Phase 13).

## Reporting

This is a private research prototype; report issues to the repository owner directly.
