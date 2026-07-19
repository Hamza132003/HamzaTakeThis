"""Serialize the report dict to JSON + a human-readable Markdown file."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .utils import fmt_ts

LANG_NAME = {"fa": "Farsi", "he": "Hebrew", "iw": "Hebrew", "ar": "Arabic",
             "en": "English"}


def write(report: dict, out_dir: Path) -> dict:
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(r: dict) -> str:
    L = ["# Voice Isolator report\n"]
    L.append(f"- **Input:** `{r['filename']}`")
    L.append(f"- **Duration:** {r['duration_sec']:.1f}s   ·   **Device:** {r['device']}")
    L.append(f"- **Speakers:** {r['num_speakers']} ({r['diarization_method']})")
    langs = ", ".join(LANG_NAME.get(x, x) for x in r["languages_detected"]) or "—"
    L.append(f"- **Languages:** {langs}   ·   **Setting:** {r['language_setting']}")
    L.append(f"- **Separation:** {r['separation_method']}   ·   "
             f"**Overlap:** {r['overlap_method']}\n")

    if r.get("warnings"):
        L.append("## ⚠ Warnings\n")
        for w in r["warnings"]:
            L.append(f"- ⚠ {w}")
        L.append("")

    if r.get("summary"):
        L.append("## AI analysis\n")
        L.append(r["summary"].get("english", "").strip())
        if r["summary"].get("arabic"):
            L.append("\n### التحليل بالعربية\n")
            L.append(r["summary"]["arabic"].strip())
        L.append(f"\n_(model: {r['summary'].get('model', '?')})_\n")

    # Noise
    L.append("## Noise (what & when)\n")
    tags = r["noise"].get("top_tags", [])
    if tags:
        L.append("**Prominent sounds:** " +
                 ", ".join(f"{t['label']} ({t['confidence']:.2f})" for t in tags) + "\n")
        L.append("| Time | Sound | Confidence |")
        L.append("|---|---|---|")
        for e in r["noise"].get("events", []):
            L.append(f"| {fmt_ts(e['start'])}–{fmt_ts(e['end'])} | {e['label']} "
                     f"| {e['confidence']:.2f} |")
    else:
        L.append("_No distinct non-speech sounds detected._")
    L.append("")

    if r.get("overlaps"):
        L.append("## Overlapping speech\n")
        for o in r["overlaps"]:
            L.append(f"- {fmt_ts(o['start'])}–{fmt_ts(o['end'])}: "
                     f"{', '.join(o['speakers'])}")
        L.append("")

    # Speech per speaker
    L.append("## Speech (transcript · Arabic · sentiment)\n")
    by_spk = defaultdict(list)
    for s in r["speech"]:
        by_spk[s["speaker"]].append(s)
    if not r["speech"]:
        L.append("_No intelligible speech transcribed._\n")
    for spk in sorted(by_spk):
        segs = by_spk[spk]
        tone = Counter(s.get("sentiment", "?") for s in segs).most_common(1)[0][0]
        talk = sum(s["end"] - s["start"] for s in segs)
        L.append(f"### {spk} · {talk:.1f}s · tone: **{tone}**\n")
        for s in segs:
            lang = LANG_NAME.get(s.get("language"), s.get("language", "?"))
            flag = " ⚠ low-confidence" if s.get("quality", {}).get("flagged") else ""
            L.append(f"- **[{fmt_ts(s['start'])}–{fmt_ts(s['end'])}] "
                     f"({lang}, {s.get('emotion','?')}/{s.get('sentiment','?')})**{flag}")
            L.append(f"  - orig: {s['text']}")
            if s.get("english"):
                L.append(f"  - EN: {s['english']}")
            if s.get("arabic") and s.get("language") not in ("ar",):
                L.append(f"  - عربي: {s['arabic']}")
        L.append("")

    # Word-level transcript (capped so the markdown stays readable; the full
    # word list is always in report.json).
    words = [(s["speaker"], w) for s in r["speech"] for w in (s.get("words") or [])]
    if words:
        L.append("## Word-level transcript\n")
        L.append("| Time | Speaker | Word | Confidence |")
        L.append("|---|---|---|---|")
        for spk, w in words[:400]:
            L.append(f"| {fmt_ts(w['start'])} | {spk} | {w['word']} "
                     f"| {w.get('probability', 0):.2f} |")
        if len(words) > 400:
            L.append(f"| … | | _{len(words) - 400} more words in report.json_ | |")
        L.append("")

    L.append("## Files\n")
    L.append(f"- voice: `{r['audio']['voice']}`  ·  noise: `{r['audio']['noise']}`")
    for spk, p in r.get("speaker_tracks", {}).items():
        L.append(f"- {spk}: `{p}`")
    return "\n".join(L) + "\n"
