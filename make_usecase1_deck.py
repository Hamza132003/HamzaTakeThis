"""Generate "Use Case 1.pptx" — a simple, graph-first deck explaining the
Voice Isolator pipeline step by step, using REAL output images from
outputs_v2/Clip3 (waveforms, spectrograms, speaker timeline).

Uses only the Python standard library (python-pptx needs lxml, whose wheel
is blocked on this network) — a .pptx is just a zip of OOXML parts.

Run:  .\\.venv\\Scripts\\python.exe make_usecase1_deck.py
"""
from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "outputs_v2" / "Clip3" / "spectrograms"
OUT = ROOT / "Use Case 1.pptx"

EMU_IN = 914400
SLIDE_W = 12192000   # 13.33 in (16:9)
SLIDE_H = 6858000    # 7.5 in

BG = "0B1019"
FG = "EAF0F7"
TEAL = "35D2CB"
AMBER = "F2B45A"
MUTED = "93A1B5"
CARD = "141C2E"


# ------------------------------------------------------------------ helpers
def png_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as f:
        head = f.read(24)
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


_id = [1]


def nid() -> int:
    _id[0] += 1
    return _id[0]


def textbox(x, y, w, h, runs, align="l", anchor="t", line_spc=None) -> str:
    """runs: list of paragraphs; each paragraph = list of (text, size_pt,
    color, bold, font?) tuples."""
    paras = []
    spc = f'<a:lnSpc><a:spcPct val="{line_spc}"/></a:lnSpc>' if line_spc else ""
    for para in runs:
        rs = []
        for t in para:
            text, sz, color, bold = t[0], t[1], t[2], t[3]
            font = t[4] if len(t) > 4 else "Segoe UI"
            rs.append(
                f'<a:r><a:rPr lang="en-US" sz="{int(sz * 100)}" b="{1 if bold else 0}" dirty="0">'
                f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
                f'<a:latin typeface="{font}"/><a:cs typeface="{font}"/></a:rPr>'
                f'<a:t>{esc(text)}</a:t></a:r>')
        paras.append(f'<a:p><a:pPr algn="{align}">{spc}</a:pPr>{"".join(rs)}</a:p>')
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{nid()}" name="tb{_id[0]}"/><p:cNvSpPr txBox="1"/>'
        f'<p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="{x}" y="{y}"/>'
        f'<a:ext cx="{w}" cy="{h}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f'<a:noFill/></p:spPr><p:txBody><a:bodyPr wrap="square" anchor="{anchor}"/>'
        f'<a:lstStyle/>{"".join(paras)}</p:txBody></p:sp>')


def chip(x, y, w, h, text, color=TEAL, fill=CARD, sz=13, bold=True) -> str:
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{nid()}" name="chip{_id[0]}"/><p:cNvSpPr/>'
        f'<p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="{x}" y="{y}"/>'
        f'<a:ext cx="{w}" cy="{h}"/></a:xfrm>'
        f'<a:prstGeom prst="roundRect"><a:avLst><a:gd name="adj" fmla="val 22000"/></a:avLst></a:prstGeom>'
        f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
        f'<a:ln w="12700"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:ln></p:spPr>'
        f'<p:txBody><a:bodyPr anchor="ctr" lIns="45720" rIns="45720"/><a:lstStyle/>'
        f'<a:p><a:pPr algn="ctr"/><a:r><a:rPr lang="en-US" sz="{int(sz * 100)}" b="{1 if bold else 0}">'
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
        f'<a:latin typeface="Segoe UI"/><a:cs typeface="Segoe UI"/></a:rPr>'
        f'<a:t>{esc(text)}</a:t></a:r></a:p></p:txBody></p:sp>')


def picture(rid, x, y, w, h) -> str:
    return (
        f'<p:pic><p:nvPicPr><p:cNvPr id="{nid()}" name="pic{_id[0]}"/>'
        f'<p:cNvPicPr/><p:nvPr/></p:nvPicPr>'
        f'<p:blipFill><a:blip r:embed="rId{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
        f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'
        f'<a:prstGeom prst="roundRect"><a:avLst><a:gd name="adj" fmla="val 4000"/></a:avLst></a:prstGeom>'
        f'<a:ln w="9525"><a:solidFill><a:srgbClr val="2C3A52"/></a:solidFill></a:ln></p:spPr></p:pic>')


def fit_image(path: Path, x, y, max_w, max_h):
    """Return (x, y, w, h) EMU keeping aspect, centered in the box."""
    pw, ph = png_size(path)
    scale = min(max_w / pw, max_h / ph)
    w, h = int(pw * scale), int(ph * scale)
    return x + (max_w - w) // 2, y + (max_h - h) // 2, w, h


def slide_xml(shapes: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        f'<p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="{BG}"/></a:solidFill>'
        '<a:effectLst/></p:bgPr></p:bg>'
        '<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
        '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        f'{shapes}</p:spTree></p:cSld><p:clrMapOvr><a:overrideClrMapping bg1="dk1" tx1="lt1" '
        'bg2="dk2" tx2="lt2" accent1="accent1" accent2="accent2" accent3="accent3" '
        'accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" '
        'folHlink="folHlink"/></p:clrMapOvr></p:sld>')


def header(title: str, subtitle: str = "") -> str:
    s = textbox(548640, 274320, 11094720, 640080,
                [[(title, 28, FG, True)]])
    if subtitle:
        s += textbox(548640, 868680, 11094720, 411480,
                     [[(subtitle, 14, MUTED, False)]])
    return s


# ------------------------------------------------------------------ slides
def build_slides():
    """Returns list of (shapes_xml, [image_paths_in_rid_order])."""
    slides = []
    IN = EMU_IN

    # -- 1. Title ---------------------------------------------------------
    s = textbox(548640, 2194560, 11094720, 1005840,
                [[("Use Case 1", 54, FG, True)]], align="ctr")
    s += textbox(548640, 3291840, 11094720, 548640,
                 [[("One noisy recording  →  clean voice · transcript · translation · AI analysis",
                    18, TEAL, False)]], align="ctr")
    s += textbox(548640, 3931920, 11094720, 411480,
                 [[("Runs 100% on this laptop — no cloud, nothing leaves the machine.",
                    13, MUTED, False)]], align="ctr")
    slides.append((s, []))

    # -- 2. The input -----------------------------------------------------
    ow = IMG_DIR / "original_wave.png"
    og = IMG_DIR / "original.png"
    s = header("The input — a voice buried in noise",
               "The microphone hears everything at once: speech and noise share one signal.")
    x, y, w, h = fit_image(ow, 548640, 1508760, 11094720, 2011680)
    s += picture(1, x, y, w, h)
    x, y, w, h = fit_image(og, 548640, 3703320, 11094720, 2560320)
    s += picture(2, x, y, w, h)
    s += textbox(548640, 6355080, 11094720, 365760,
                 [[("Top: the raw waveform (loudness over time).   Bottom: its spectrogram (which frequencies, when).",
                    12, MUTED, False)]])
    slides.append((s, [ow, og]))

    # -- 3. Step 1: isolate -----------------------------------------------
    vw = IMG_DIR / "voice_wave.png"
    nw = IMG_DIR / "noise_wave.png"
    s = header("Step 1 — Isolate the voice",
               "A neural network (MossFormer2, 48 kHz) has learned what human speech looks like — "
               "it keeps that, and everything else becomes the noise track.")
    s += textbox(548640, 1554480, 5486400, 365760, [[("ISOLATED VOICE", 13, TEAL, True)]])
    x, y, w, h = fit_image(vw, 548640, 1920240, 11094720, 1874520)
    s += picture(1, x, y, w, h)
    s += textbox(548640, 3931920, 5486400, 365760, [[("ISOLATED NOISE  (what was removed)", 13, AMBER, True)]])
    x, y, w, h = fit_image(nw, 548640, 4297680, 11094720, 1874520)
    s += picture(2, x, y, w, h)
    s += textbox(548640, 6263640, 11094720, 365760,
                 [[("One signal in, two signals out — subtracting the clean voice from the original leaves pure noise.",
                    12, MUTED, False)]])
    slides.append((s, [vw, nw]))

    # -- 4. Proof in the spectrum ------------------------------------------
    vg = IMG_DIR / "voice.png"
    ng = IMG_DIR / "noise.png"
    s = header("The same split, seen in the spectrum",
               "Speech lives in harmonic stripes; noise is smeared everywhere. The model separates the two patterns.")
    x, y, w, h = fit_image(vg, 548640, 1554480, 11094720, 2286000)
    s += picture(1, x, y, w, h)
    x, y, w, h = fit_image(ng, 548640, 3977640, 11094720, 2286000)
    s += picture(2, x, y, w, h)
    slides.append((s, [vg, ng]))

    # -- 5. Step 2: speakers ------------------------------------------------
    tl = IMG_DIR / "timeline.png"
    s = header("Step 2 — Who spoke, and when",
               "Every voice has a fingerprint. Grouping fingerprints over time gives the speakers, "
               "their turns — and the moments they talk over each other.")
    x, y, w, h = fit_image(tl, 548640, 1737360, 11094720, 3200400)
    s += picture(1, x, y, w, h)
    s += textbox(548640, 5303520, 11094720, 731520,
                 [[("Overlapping moments are un-mixed by a separation model, then each stream is matched ",
                    13, MUTED, False)],
                  [("back to its owner by comparing voice fingerprints.", 13, MUTED, False)]])
    slides.append((s, [tl]))

    # -- 6. Step 3: words ----------------------------------------------------
    s = header("Step 3 — Every word, with a timestamp and a confidence",
               "Whisper reads the spectrogram like a language. It only decodes where speech was found — "
               "so background noise can never invent sentences.")
    words = _sample_words()
    cx = 548640
    for wtext, wtime, prob in words:
        wide = 1005840 + 137160 * max(0, len(wtext) - 4)
        color = TEAL if prob >= 0.5 else AMBER
        s += chip(cx, 2377440, wide, 731520, wtext, color=color, sz=16)
        s += textbox(cx, 3154680, wide, 320040,
                     [[(f"{wtime}  ·  {int(prob * 100)}%", 10, MUTED, False)]], align="ctr")
        cx += wide + 182880
        if cx > 10972800:
            break
    s += textbox(548640, 4114800, 11094720, 731520,
                 [[("Real words from this recording — each knows exactly when it was said and how sure the model is. ",
                    13, MUTED, False)],
                  [("Low-confidence and hallucination-pattern segments are flagged, never silently trusted.",
                    13, MUTED, False)]])
    slides.append((s, []))

    # -- 7. Step 4: two languages -------------------------------------------
    s = header("Step 4 — Two languages out, safely",
               "Two independent paths: if one guess goes wrong, it cannot poison the other.")
    y0 = 2011680
    s += chip(548640, y0, 2377440, 822960, "Source speech", FG)
    s += textbox(3017520, y0, 731520, 822960, [[("→", 28, MUTED, False)]], align="ctr", anchor="ctr")
    s += chip(3840480, y0, 2606040, 822960, "Whisper (translate)", TEAL)
    s += textbox(6537960, y0, 731520, 822960, [[("→", 28, MUTED, False)]], align="ctr", anchor="ctr")
    s += chip(7361110, y0, 2148840, 822960, "English", TEAL)
    y1 = 3383280
    s += chip(548640, y1, 2377440, 822960, "Source words", FG)
    s += textbox(3017520, y1, 731520, 822960, [[("→", 28, MUTED, False)]], align="ctr", anchor="ctr")
    s += chip(3840480, y1, 2606040, 822960, "NLLB-200 (1.3B)", AMBER)
    s += textbox(6537960, y1, 731520, 822960, [[("→", 28, MUTED, False)]], align="ctr", anchor="ctr")
    s += chip(7361110, y1, 2148840, 822960, "العربية", AMBER)
    s += textbox(548640, 4754880, 11094720, 731520,
                 [[("Arabic is translated directly from the source-language words — never from the English guess.",
                    13, MUTED, False)]])
    slides.append((s, []))

    # -- 8. Step 5: AI analysis ----------------------------------------------
    s = header("Step 5 — A local AI reads the whole picture",
               "A small on-device language model combines the transcript, the background sounds and "
               "the vocal tones — and infers the story.")
    rows = [("Setting", "where they probably are — inferred from the background sounds"),
            ("Topic", "what the conversation is about"),
            ("Speakers", "who each voice might be — role, relationship, state of mind"),
            ("Activity", "what they are probably doing while talking"),
            ("Key points", "names, places, times and intentions worth attention")]
    y0 = 1783080
    for label, desc in rows:
        s += chip(548640, y0, 2011680, 640080, label, TEAL, sz=14)
        s += textbox(2743200, y0, 8900160, 640080, [[(desc, 14, FG, False)]], anchor="ctr")
        y0 += 822960
    s += textbox(548640, 6081840, 11094720, 411480,
                 [[("Everything is phrased as evidence-based inference (“probably / possibly”) — never invented facts.",
                    12, MUTED, False)]])
    slides.append((s, []))

    # -- 9. The whole pipeline ------------------------------------------------
    s = header("The whole pipeline", "Ten seconds of theory, end to end.")
    steps = ["Audio", "Isolate", "Name the noise", "Speakers", "Words", "Translate", "AI analysis", "Report"]
    colors = [FG, TEAL, AMBER, TEAL, TEAL, AMBER, TEAL, FG]
    cx, cy = 548640, 2560320
    for i, (st, c) in enumerate(zip(steps, colors)):
        wide = 868680 + 100584 * len(st)
        if cx + wide > 11887200:
            cx = 548640
            cy += 1188720
        s += chip(cx, cy, wide, 731520, st, c, sz=14)
        cx += wide + 128016
        if i < len(steps) - 1 and cx + 231648 <= 11887200:
            s += textbox(cx - 100584, cy, 274320, 731520, [[("→", 20, MUTED, False)]], align="ctr", anchor="ctr")
            cx += 219456
    s += textbox(548640, 5486400, 11094720, 548640,
                 [[("Every model runs on this machine. The only thing the recording ever touches is your own GPU.",
                    14, MUTED, False)]], align="ctr")
    slides.append((s, []))

    return slides


def _sample_words():
    """Pull a few real words (text, time, confidence) from the Clip3 report."""
    try:
        r = json.loads((ROOT / "outputs_v2" / "Clip3" / "report.json").read_text(encoding="utf-8"))
        out = []
        for seg in r["speech"]:
            for w in seg.get("words", []):
                token = w["word"].strip()
                if len(token) >= 3:
                    m, sec = divmod(w["start"], 60)
                    out.append((token, f"{int(m)}:{sec:04.1f}", w.get("probability", 0)))
            if len(out) >= 6:
                break
        if out:
            return out[:6]
    except Exception:
        pass
    return [("مرحبا", "0:01.2", 0.94), ("كيف", "0:02.0", 0.88), ("حالك", "0:02.4", 0.91)]


# ------------------------------------------------------------------ OOXML
THEME = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="VI"><a:themeElements><a:clrScheme name="VI"><a:dk1><a:srgbClr val="0B1019"/></a:dk1><a:lt1><a:srgbClr val="EAF0F7"/></a:lt1><a:dk2><a:srgbClr val="141C2E"/></a:dk2><a:lt2><a:srgbClr val="93A1B5"/></a:lt2><a:accent1><a:srgbClr val="35D2CB"/></a:accent1><a:accent2><a:srgbClr val="F2B45A"/></a:accent2><a:accent3><a:srgbClr val="6ED69A"/></a:accent3><a:accent4><a:srgbClr val="8FA8FF"/></a:accent4><a:accent5><a:srgbClr val="E78A7B"/></a:accent5><a:accent6><a:srgbClr val="D98FFF"/></a:accent6><a:hlink><a:srgbClr val="35D2CB"/></a:hlink><a:folHlink><a:srgbClr val="93A1B5"/></a:folHlink></a:clrScheme><a:fontScheme name="VI"><a:majorFont><a:latin typeface="Segoe UI"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont><a:minorFont><a:latin typeface="Segoe UI"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme><a:fmtScheme name="Office"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="19050"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>"""

MASTER = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="0B1019"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMap bg1="dk1" tx1="lt1" bg2="dk2" tx2="lt2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle><a:lvl1pPr><a:defRPr sz="2800"/></a:lvl1pPr></p:titleStyle><p:bodyStyle><a:lvl1pPr><a:defRPr sz="1400"/></a:lvl1pPr></p:bodyStyle><p:otherStyle><a:lvl1pPr><a:defRPr sz="1400"/></a:lvl1pPr></p:otherStyle></p:txStyles></p:sldMaster>"""

LAYOUT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>"""


def build_pptx():
    slides = build_slides()
    n = len(slides)

    ct = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/>'
          '<Default Extension="png" ContentType="image/png"/>'
          '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
          '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>'
          '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>'
          '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>'
          '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
          '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>']
    for i in range(1, n + 1):
        ct.append(f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>')
    ct.append('</Types>')

    root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                 '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                 '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>'
                 '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
                 '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
                 '</Relationships>')

    sld_ids = "".join(f'<p:sldId id="{255 + i}" r:id="rId{1 + i}"/>' for i in range(1, n + 1))
    presentation = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                    '<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
                    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
                    'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
                    '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
                    f'<p:sldIdLst>{sld_ids}</p:sldIdLst>'
                    f'<p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}"/><p:notesSz cx="{SLIDE_H}" cy="{SLIDE_W}"/></p:presentation>')

    pres_rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                 '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                 '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>']
    for i in range(1, n + 1):
        pres_rels.append(f'<Relationship Id="rId{1 + i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>')
    pres_rels.append(f'<Relationship Id="rId{n + 2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>')
    pres_rels.append('</Relationships>')

    master_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
                   '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>'
                   '</Relationships>')

    layout_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>'
                   '</Relationships>')

    core = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            '<dc:title>Use Case 1</dc:title><dc:creator>Voice Isolator</dc:creator></cp:coreProperties>')
    app = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
           '<Application>Voice Isolator</Application></Properties>')

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", "".join(ct))
        z.writestr("_rels/.rels", root_rels)
        z.writestr("docProps/core.xml", core)
        z.writestr("docProps/app.xml", app)
        z.writestr("ppt/presentation.xml", presentation)
        z.writestr("ppt/_rels/presentation.xml.rels", "".join(pres_rels))
        z.writestr("ppt/slideMasters/slideMaster1.xml", MASTER)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", master_rels)
        z.writestr("ppt/slideLayouts/slideLayout1.xml", LAYOUT)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", layout_rels)
        z.writestr("ppt/theme/theme1.xml", THEME)

        media_index = {}
        for i, (shapes, images) in enumerate(slides, start=1):
            rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                    '<Relationship Id="rId100" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>']
            for j, img in enumerate(images, start=1):
                key = str(img)
                if key not in media_index:
                    media_index[key] = f"image{len(media_index) + 1}.png"
                    z.writestr(f"ppt/media/{media_index[key]}", img.read_bytes())
                rels.append(f'<Relationship Id="rId{j}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{media_index[key]}"/>')
            rels.append('</Relationships>')
            z.writestr(f"ppt/slides/slide{i}.xml", slide_xml(shapes))
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", "".join(rels))

    print(f"Deck written: {OUT} ({OUT.stat().st_size // 1024} KB, {n} slides)")


if __name__ == "__main__":
    build_pptx()
