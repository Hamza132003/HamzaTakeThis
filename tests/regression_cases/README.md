# Private regression cases

Case definitions live here as JSON. **Recordings never do.** Each case points
at an audio path on the operator's machine and is identified by SHA-256; the
audio itself is excluded from Git (`.gitignore`), never copied into the repo,
and never uploaded.

A case file records what the recording is *expected* to produce, so a
regression is detected by content rather than by eyeballing the dashboard.

Run them with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_private_regressions.py -q
```

Cases whose audio is absent on this machine **skip** rather than fail, so CI
and clean clones stay green.
