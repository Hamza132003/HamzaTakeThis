"""Isolated pyannote 4 diarization worker.

Runs in its OWN Python environment (numpy 2 + pyannote.audio 4), invoked as a
subprocess by the main pipeline. Only `protocol.py` is safe to import from the
main environment — `main.py` imports pyannote and must never run in-process.
"""
