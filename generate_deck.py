"""Generate 'Voice Isolator - How It Works.pptx'.

A theory deck explaining every stage of the local pipeline: source separation,
noise identification, diarization, Whisper transcription, translation and
emotion. Run:  python generate_deck.py
"""
from __future__ import annotations

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE

# ---------------------------------------------------------------- palette
BG      = RGBColor(0x0F, 0x15, 0x23)   # deep navy ground
PANEL   = RGBColor(0x19, 0x22, 0x34)   # raised panel
PANEL2  = RGBColor(0x22, 0x2E, 0x45)   # brighter panel
ACCENT  = RGBColor(0x35, 0xD2, 0xCB)   # teal (primary signal)
ACCENT2 = RGBColor(0xF2, 0xB4, 0x5A)   # amber (secondary / highlight)
TEXT    = RGBColor(0xEA, 0xF0, 0xF7)   # near-white
MUTED   = RGBColor(0x93, 0xA1, 0xB5)   # muted slate
LINE    = RGBColor(0x2C, 0x3A, 0x52)   # hairline
GOOD    = RGBColor(0x6E, 0xD6, 0x9A)
WARN    = RGBColor(0xE7, 0x8A, 0x7B)

BODY = "Segoe UI"
MONO = "Consolas"

EMU_W, EMU_H = Inches(13.333), Inches(7.5)


# ---------------------------------------------------------------- helpers
def style(run, size, color, bold=False, italic=False, font=BODY):
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = font


def textbox(slide, l, t, w, h, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def para(tf, text, size, color, bold=False, italic=False, font=BODY,
         first=False, space_after=6, space_before=0, align=PP_ALIGN.LEFT):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.space_after = Pt(space_after)
    p.space_before = Pt(space_before)
    p.alignment = align
    r = p.add_run()
    r.text = text
    style(r, size, color, bold, italic, font)
    return p


def rich(tf, segments, size, bold=False, first=False, space_after=6,
         space_before=0, font=BODY):
    """A paragraph built from (text, color) segments for inline emphasis."""
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.space_after = Pt(space_after)
    p.space_before = Pt(space_before)
    for text, color in segments:
        r = p.add_run()
        r.text = text
        style(r, size, color, bold, font=font)
    return p


def slide_bg(slide, color=BG):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def rect(slide, l, t, w, h, fill=PANEL, line=None, radius=0.06,
         shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    sp = slide.shapes.add_shape(shape, Inches(l), Inches(t), Inches(w), Inches(h))
    sp.fill.solid()
    sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(1)
    sp.shadow.inherit = False
    try:
        sp.adjustments[0] = radius
    except Exception:
        pass
    return sp


def chip(slide, l, t, w, h, label, sub=None, fill=PANEL, accent=ACCENT):
    sp = rect(slide, l, t, w, h, fill=fill, radius=0.14)
    tf = sp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.08); tf.margin_right = Inches(0.08)
    tf.margin_top = Inches(0.04); tf.margin_bottom = Inches(0.04)
    para(tf, label, 11.5, accent, bold=True, first=True, space_after=1,
         align=PP_ALIGN.CENTER)
    if sub:
        para(tf, sub, 8.5, MUTED, align=PP_ALIGN.CENTER)
    return sp


def arrow(slide, l, t, w, h=0.3, color=ACCENT):
    sp = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(l), Inches(t),
                                Inches(w), Inches(h))
    sp.fill.solid(); sp.fill.fore_color.rgb = color
    sp.line.fill.background(); sp.shadow.inherit = False
    return sp


def header(slide, eyebrow, title, accent=ACCENT):
    rect(slide, 0.7, 0.62, 0.12, 0.42, fill=accent, radius=0.4)
    tf = textbox(slide, 0.95, 0.55, 11.6, 1.1)
    para(tf, eyebrow.upper(), 12, accent, bold=True, first=True, space_after=3)
    para(tf, title, 27, TEXT, bold=True, space_after=0)


def footer(slide, page, total=15):
    ln = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(7.02),
                                Inches(11.93), Pt(1.2))
    ln.fill.solid(); ln.fill.fore_color.rgb = LINE; ln.line.fill.background()
    ln.shadow.inherit = False
    tf = textbox(slide, 0.7, 7.08, 11.93, 0.3)
    p = tf.paragraphs[0]
    p.space_after = 0
    for text, color in [("VOICE ISOLATOR", MUTED), ("   ·   fully local, offline", RGBColor(0x5C,0x6A,0x80))]:
        r = p.add_run(); r.text = text; style(r, 9, color, font=MONO)
    tf2 = textbox(slide, 11.9, 7.08, 0.73, 0.3)
    para(tf2, f"{page:02d} / {total}", 9, MUTED, font=MONO, first=True,
         align=PP_ALIGN.RIGHT)


def panel_with_text(slide, l, t, w, h, heading, lines, accent=ACCENT,
                    fill=PANEL, hsize=13, bsize=11.5):
    """lines: list of strings; a leading '·' becomes an accent bullet."""
    rect(slide, l, t, w, h, fill=fill)
    tf = textbox(slide, l + 0.28, t + 0.24, w - 0.56, h - 0.48)
    para(tf, heading.upper(), 11, accent, bold=True, first=True, space_after=8)
    for i, line in enumerate(lines):
        if line.startswith("· "):
            rich(tf, [("●  ", accent), (line[2:], TEXT)], bsize, space_after=6)
        elif line.startswith("> "):
            para(tf, line[2:], bsize - 0.5, MUTED, italic=True, space_after=6)
        else:
            para(tf, line, bsize, TEXT, space_after=6)


def code_panel(slide, l, t, w, h, title, code_lines):
    rect(slide, l, t, w, h, fill=RGBColor(0x0B, 0x10, 0x1B), line=LINE)
    tf = textbox(slide, l + 0.24, t + 0.2, w - 0.48, h - 0.4)
    para(tf, title, 9.5, ACCENT2, bold=True, first=True, space_after=6, font=MONO)
    for text, color in code_lines:
        para(tf, text, 10.5, color, font=MONO, space_after=2)


# ---------------------------------------------------------------- build
prs = Presentation()
prs.slide_width = EMU_W
prs.slide_height = EMU_H
BLANK = prs.slide_layouts[6]


def new_slide():
    s = prs.slides.add_slide(BLANK)
    slide_bg(s)
    return s


STAGES = ["Extract", "Separate", "Enhance", "Identify\nnoise",
          "Diarize", "Transcribe", "Translate", "Sentiment"]


def pipeline_strip(slide, top, active=None):
    l = 0.7
    n = len(STAGES)
    gap = 0.12
    total = 11.93
    w = (total - gap * (n - 1)) / n
    for i, name in enumerate(STAGES):
        is_active = (active == i)
        fill = PANEL2 if is_active else PANEL
        acc = ACCENT2 if is_active else ACCENT
        sp = rect(slide, l, top, w, 0.62, fill=fill, radius=0.16)
        tf = sp.text_frame; tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = Inches(0.04); tf.margin_right = Inches(0.04)
        for j, ln in enumerate(name.split("\n")):
            para(tf, ln, 9.5, acc, bold=True, first=(j == 0),
                 space_after=0, align=PP_ALIGN.CENTER)
        l += w + gap


# ---- Slide 1: title ------------------------------------------------------
s = new_slide()
# ambient waveform bars across the hero
import math
base_y = 3.05
for i in range(46):
    x = 0.7 + i * 0.265
    amp = 0.18 + 1.35 * abs(math.sin(i * 0.5)) * (0.4 + 0.6 * abs(math.cos(i * 0.27)))
    col = ACCENT if i % 3 else ACCENT2
    bar = slide_bg  # noop guard
    r = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x),
                           Inches(base_y - amp / 2), Inches(0.11), Inches(amp))
    r.fill.solid(); r.fill.fore_color.rgb = col
    r.line.fill.background(); r.shadow.inherit = False
    r.fill.fore_color.rgb = col
    try:
        r.adjustments[0] = 0.5
    except Exception:
        pass
    # fade via transparency not trivial; use muted tone for depth
    if i % 3:
        r.fill.fore_color.rgb = RGBColor(0x1E, 0x4A, 0x52)
    else:
        r.fill.fore_color.rgb = RGBColor(0x5A, 0x45, 0x2A)

tf = textbox(s, 0.7, 0.7, 11.9, 1.4)
para(tf, "LOCAL · OFFLINE · NO CLOUD APIs", 13, ACCENT, bold=True, first=True,
     space_after=10, font=MONO)
tf = textbox(s, 0.7, 1.15, 11.9, 1.6)
para(tf, "Voice Isolator", 54, TEXT, bold=True, first=True, space_after=2)
para(tf, "How it works — from raw video to per-speaker Arabic", 22, MUTED)
tf = textbox(s, 0.7, 4.9, 11.9, 1.8)
rich(tf, [("Isolate the voices from the noise, figure out ", TEXT),
          ("what the noise is", ACCENT),
          (", ", TEXT), ("count the speakers", ACCENT),
          (", transcribe ", TEXT), ("Farsi & Hebrew", ACCENT2),
          (", translate to ", TEXT), ("Arabic", ACCENT2),
          (", and read the ", TEXT), ("emotion", ACCENT), (".", TEXT)],
     17, first=True, space_after=10)
para(tf, "Everything below runs on your own machine (NVIDIA RTX A1000).",
     13, MUTED, italic=True)
footer(s, 1)

# ---- Slide 2: overview ---------------------------------------------------
s = new_slide()
header(s, "The big picture", "One command, eight stages, two output files")
tf = textbox(s, 0.7, 1.7, 11.9, 0.9)
rich(tf, [("You hand the pipeline a ", TEXT), ("30–60s recording", ACCENT2),
          (". It runs each stage in sequence, freeing GPU memory between them "
           "so everything fits in 6 GB, and writes a readable report plus "
           "structured JSON.", TEXT)], 15, first=True)
pipeline_strip(s, 2.75)
# data flow row
panel_with_text(s, 0.7, 3.75, 3.75, 2.9, "Input",
                ["· A video or audio file",
                 "· Distant speech, one or many talkers",
                 "· Loud static / mechanical noise",
                 "· Two languages: Farsi or Hebrew",
                 "> the messy real-world clip"], accent=ACCENT2)
panel_with_text(s, 4.79, 3.75, 3.75, 2.9, "What each stage adds",
                ["· Clean voice + isolated noise",
                 "· Noise labelled on a timeline",
                 "· Speaker count & turns",
                 "· Transcript per speaker",
                 "· Arabic translation + tone"], accent=ACCENT)
panel_with_text(s, 8.88, 3.75, 3.75, 2.9, "Output",
                ["· voice.wav  /  noise.wav",
                 "· report.md  (human-readable)",
                 "· report.json  (per-segment data)",
                 "> speaker · time · language ·",
                 "  text · arabic · emotion"], accent=ACCENT2)
footer(s, 2)

# ---- Slide 3: signal basics ---------------------------------------------
s = new_slide()
header(s, "Foundation", "How a computer 'sees' sound")
panel_with_text(s, 0.7, 1.75, 3.85, 4.9, "1 · Waveform",
                ["Sound is air pressure over time.",
                 "We sample it thousands of times per second.",
                 "· 16,000 samples/sec (16 kHz) is the",
                 "  standard for speech models — enough",
                 "  to capture the human voice.",
                 "> Just a long list of numbers."], accent=ACCENT)
panel_with_text(s, 4.74, 1.75, 3.85, 4.9, "2 · Spectrogram",
                ["A waveform hides which frequencies",
                 "are present. A Fourier transform over",
                 "short windows turns it into a picture:",
                 "· time on the x-axis",
                 "· frequency on the y-axis",
                 "· brightness = energy",
                 "> Now patterns become visible."], accent=ACCENT)
panel_with_text(s, 8.78, 1.75, 3.85, 4.9, "3 · Mel spectrogram",
                ["Human hearing is not linear — we",
                 "resolve low pitches far better than high.",
                 "The mel scale warps frequency to match",
                 "the ear, then we take the log of energy.",
                 "· This 'log-mel' image is the input to",
                 "  Whisper, PANNs and the emotion model.",
                 "> The shared language of the pipeline."],
                accent=ACCENT2)
footer(s, 3)

# ---- Slide 4: stage 1 audio ---------------------------------------------
s = new_slide()
header(s, "Stage 1 · Extraction", "Pulling clean audio out of the video")
pipeline_strip(s, 1.7, active=0)
panel_with_text(s, 0.7, 2.7, 5.9, 3.9, "What happens",
                ["· ffmpeg strips the video track away,",
                 "  keeping only the audio.",
                 "· Resampled to 16 kHz, mono.",
                 "· Two versions are saved: a 16 kHz mono",
                 "  working copy, and a 44.1 kHz stereo",
                 "  copy that the separator prefers.",
                 "> The ffmpeg binary is bundled — no",
                 "  system install needed."], accent=ACCENT)
panel_with_text(s, 6.74, 2.7, 5.9, 3.9, "Why these choices",
                ["· 16 kHz captures speech (voice energy",
                 "  lives below 8 kHz) while keeping files",
                 "  small and models fast.",
                 "· Mono, because a single distant mic has",
                 "  no useful stereo information.",
                 "· Higher-quality stereo is kept only for",
                 "  separation, which benefits from it."], accent=ACCENT2)
footer(s, 4)

# ---- Slide 5: stage 2 separation ----------------------------------------
s = new_slide()
header(s, "Stage 2 · Separation", "Splitting voice from noise with Demucs")
pipeline_strip(s, 1.7, active=1)
# diagram
rect(s, 0.7, 2.75, 2.4, 1.0, fill=PANEL2)
tf = textbox(s, 0.7, 2.75, 2.4, 1.0, anchor=MSO_ANCHOR.MIDDLE)
para(tf, "Mixed audio", 13, TEXT, bold=True, first=True, align=PP_ALIGN.CENTER, space_after=1)
para(tf, "voice + noise", 10, MUTED, align=PP_ALIGN.CENTER)
arrow(s, 3.25, 3.05, 0.7)
rect(s, 4.1, 2.6, 2.7, 1.3, fill=PANEL, line=ACCENT)
tf = textbox(s, 4.1, 2.6, 2.7, 1.3, anchor=MSO_ANCHOR.MIDDLE)
para(tf, "Demucs U-Net", 13, ACCENT, bold=True, first=True, align=PP_ALIGN.CENTER, space_after=1)
para(tf, "learns a mask for", 9.5, MUTED, align=PP_ALIGN.CENTER, space_after=0)
para(tf, "each source", 9.5, MUTED, align=PP_ALIGN.CENTER)
arrow(s, 6.95, 3.05, 0.7)
chip(s, 7.8, 2.5, 2.3, 0.62, "vocals → voice.wav", accent=ACCENT)
chip(s, 7.8, 3.25, 2.3, 0.62, "drums+bass+other", accent=ACCENT2)
tf = textbox(s, 10.25, 2.5, 2.4, 1.4, anchor=MSO_ANCHOR.MIDDLE)
para(tf, "→ speech", 11, ACCENT, bold=True, first=True, space_after=8)
para(tf, "→ noise.wav", 11, ACCENT2, bold=True)
panel_with_text(s, 0.7, 4.2, 5.9, 2.5, "The theory",
                ["Demucs is a deep U-Net trained on huge",
                 "audio mixtures. It encodes the sound, then",
                 "decodes one output per source, learning a",
                 "time-frequency mask that keeps each part.",
                 "· We repurpose its 'vocals' stem as speech."],
                accent=ACCENT)
panel_with_text(s, 6.74, 4.2, 5.9, 2.5, "Why this is the real win",
                ["Removing noise raises the signal-to-noise",
                 "ratio — the single biggest factor in whether",
                 "transcription succeeds.",
                 "· Bonus: the leftover 'noise' track is exactly",
                 "  what Stage 4 analyses.",
                 "> If Demucs is missing, a spectral-gate",
                 "  fallback runs instead."], accent=ACCENT2)
footer(s, 5)

# ---- Slide 6: enhancement (the dB question) -----------------------------
s = new_slide()
header(s, "Stage 3 · Enhancement", "Why 'just turn up the volume' isn't the fix",
       accent=ACCENT2)
pipeline_strip(s, 1.7, active=2)
panel_with_text(s, 0.7, 2.7, 5.9, 3.9, "The misconception",
                ["Whisper does NOT hear loudness. Its first",
                 "step normalises the log-mel spectrogram,",
                 "so -30 dB and -6 dB look almost identical",
                 "to it.",
                 "· Raising gain scales the noise by the exact",
                 "  same factor as the speech, so the",
                 "  signal-to-noise ratio never improves.",
                 "> Louder ≠ clearer."], accent=WARN)
panel_with_text(s, 6.74, 2.7, 5.9, 3.9, "What we actually do",
                ["· High-pass filter: cut sub-80 Hz rumble",
                 "  and handling thud.",
                 "· Loudness-normalise to a consistent target",
                 "  (RMS ≈ −20 dBFS), with a −1 dBFS peak",
                 "  ceiling so it never clips.",
                 "· Optional deep denoise (DeepFilterNet).",
                 "> This helps the level-sensitive stages —",
                 "  voice activity, diarization, emotion —",
                 "  and separation is what helps Whisper."],
                accent=ACCENT)
footer(s, 6)

# ---- Slide 7: noise identification --------------------------------------
s = new_slide()
header(s, "Stage 4 · Noise ID", "What is the noise, and when? (PANNs)")
pipeline_strip(s, 1.7, active=3)
panel_with_text(s, 0.7, 2.7, 5.9, 3.9, "How it works",
                ["The isolated noise track goes into PANNs —",
                 "a CNN trained on Google's AudioSet.",
                 "· It knows 527 sound classes: engine,",
                 "  fan, static, footsteps, wind, machinery…",
                 "· Run in 'detection' mode, it scores every",
                 "  class in short frames across time.",
                 "· Contiguous high-confidence frames become",
                 "  a timestamped event."], accent=ACCENT)
panel_with_text(s, 6.74, 2.7, 3.0, 3.9, "Output shape",
                ["A ranked list of the",
                 "loudest sound types,",
                 "plus a timeline:",
                 "> engine hum",
                 "   0:00–0:44",
                 "> metallic bang",
                 "   0:12–0:13"], accent=ACCENT2, bsize=11)
code_panel(s, 9.88, 2.7, 2.75, 3.9, "report.json (noise)",
           [("\"label\":", ACCENT),
            ("  \"Engine\",", TEXT),
            ("\"start\": 0.0,", MUTED),
            ("\"end\":  44.2,", MUTED),
            ("\"conf\": 0.71", ACCENT2),
            ("", TEXT),
            ("# what + when,", RGBColor(0x5C,0x6A,0x80)),
            ("# not direction", RGBColor(0x5C,0x6A,0x80))])
footer(s, 7)

# ---- Slide 8: diarization -----------------------------------------------
s = new_slide()
header(s, "Stage 5 · Diarization", "Counting speakers and who spoke when")
pipeline_strip(s, 1.7, active=4)
# mini flow
labels = [("Detect speech", "voice activity"), ("Embed voice", "a fingerprint\nper slice"),
          ("Cluster", "group similar\nfingerprints"), ("Label turns", "SPEAKER_00,\n01, 02…")]
x = 0.7
for i, (t, sub) in enumerate(labels):
    rect(s, x, 2.72, 2.6, 1.15, fill=PANEL, line=(ACCENT if i == 2 else None))
    tf = textbox(s, x, 2.72, 2.6, 1.15, anchor=MSO_ANCHOR.MIDDLE)
    para(tf, t, 12.5, TEXT, bold=True, first=True, align=PP_ALIGN.CENTER, space_after=2)
    for j, ln in enumerate(sub.split("\n")):
        para(tf, ln, 9.5, MUTED, align=PP_ALIGN.CENTER, space_after=0)
    if i < 3:
        arrow(s, x + 2.62, 3.12, 0.32, color=ACCENT)
    x += 2.98
panel_with_text(s, 0.7, 4.25, 5.9, 2.45, "The theory (pyannote)",
                ["Each short voice slice is turned into an",
                 "embedding — a vector that captures a",
                 "person's vocal signature. Slices from the",
                 "same person land near each other, so",
                 "clustering reveals how many people spoke",
                 "and which turn belongs to whom."], accent=ACCENT)
panel_with_text(s, 6.74, 4.25, 5.9, 2.45, "Practical notes",
                ["· Needs a one-time free HuggingFace token",
                 "  to download the model — then fully offline.",
                 "· No token? It falls back to single-speaker.",
                 "· Overlapping distant voices are the hard",
                 "  case; counts are best-effort, not exact."],
                accent=ACCENT2)
footer(s, 8)

# ---- Slide 9: Whisper deep dive -----------------------------------------
s = new_slide()
header(s, "Stage 6 · Transcription", "Whisper, up close")
pipeline_strip(s, 1.7, active=5)
# architecture diagram
rect(s, 0.7, 2.72, 2.2, 1.15, fill=PANEL2)
tf = textbox(s, 0.7, 2.72, 2.2, 1.15, anchor=MSO_ANCHOR.MIDDLE)
para(tf, "Log-mel", 13, ACCENT, bold=True, first=True, align=PP_ALIGN.CENTER, space_after=1)
para(tf, "of the voice", 9.5, MUTED, align=PP_ALIGN.CENTER)
arrow(s, 2.95, 3.12, 0.4)
rect(s, 3.45, 2.72, 2.5, 1.15, fill=PANEL, line=ACCENT)
tf = textbox(s, 3.45, 2.72, 2.5, 1.15, anchor=MSO_ANCHOR.MIDDLE)
para(tf, "Transformer", 13, TEXT, bold=True, first=True, align=PP_ALIGN.CENTER, space_after=1)
para(tf, "ENCODER", 9.5, ACCENT, align=PP_ALIGN.CENTER, space_after=0)
para(tf, "'understands' the audio", 8.5, MUTED, align=PP_ALIGN.CENTER)
arrow(s, 6.0, 3.12, 0.4)
rect(s, 6.5, 2.72, 2.5, 1.15, fill=PANEL, line=ACCENT2)
tf = textbox(s, 6.5, 2.72, 2.5, 1.15, anchor=MSO_ANCHOR.MIDDLE)
para(tf, "Transformer", 13, TEXT, bold=True, first=True, align=PP_ALIGN.CENTER, space_after=1)
para(tf, "DECODER", 9.5, ACCENT2, align=PP_ALIGN.CENTER, space_after=0)
para(tf, "writes text, token by token", 8.5, MUTED, align=PP_ALIGN.CENTER)
arrow(s, 9.05, 3.12, 0.4)
rect(s, 9.55, 2.72, 3.08, 1.15, fill=PANEL2)
tf = textbox(s, 9.55, 2.72, 3.08, 1.15, anchor=MSO_ANCHOR.MIDDLE)
para(tf, "متن / טקסט", 15, TEXT, bold=True, first=True, align=PP_ALIGN.CENTER, space_after=1)
para(tf, "transcribed words", 9.5, MUTED, align=PP_ALIGN.CENTER)
panel_with_text(s, 0.7, 4.25, 6.05, 2.45, "Why it's so capable",
                ["· Trained on 680,000 hours of multilingual",
                 "  audio scraped from the web.",
                 "· The decoder is prompted with special",
                 "  tokens: <|language|> then <|transcribe|>,",
                 "  so one model handles 99 languages.",
                 "· It predicts the next text token from both",
                 "  the audio and the words so far — like a",
                 "  language model that can listen."], accent=ACCENT)
panel_with_text(s, 6.9, 4.25, 5.73, 2.45, "Our Farsi/Hebrew trick  [fa, he]",
                ["Noisy distant audio fools auto language",
                 "detection. So per speaker turn we decode",
                 "TWICE — forcing Farsi, then Hebrew —",
                 "and keep whichever Whisper is more",
                 "confident about (higher avg log-prob).",
                 "> Reliable language pick on hard audio."],
                accent=ACCENT2)
footer(s, 9)

# ---- Slide 10: translation ----------------------------------------------
s = new_slide()
header(s, "Stage 7 · Translation", "Farsi / Hebrew → Arabic with NLLB-200")
pipeline_strip(s, 1.7, active=6)
panel_with_text(s, 0.7, 2.7, 6.05, 3.9, "How it works",
                ["NLLB-200 ('No Language Left Behind') is a",
                 "sequence-to-sequence transformer from Meta",
                 "that translates directly between 200",
                 "languages — no English pivot needed.",
                 "· Whisper only translates TO English, so we",
                 "  use a dedicated model for Arabic.",
                 "· We tell it the source and force the target",
                 "  by seeding the decoder with a language",
                 "  token."], accent=ACCENT)
code_panel(s, 6.95, 2.7, 5.68, 3.9, "language codes (FLORES-200)",
           [("Farsi   →  pes_Arab", TEXT),
            ("Hebrew  →  heb_Hebr", TEXT),
            ("Arabic  →  arb_Arab   (target)", ACCENT2),
            ("", TEXT),
            ("# forced_bos_token_id =", RGBColor(0x5C,0x6A,0x80)),
            ("#   tokenizer(\"arb_Arab\")", RGBColor(0x5C,0x6A,0x80)),
            ("", TEXT),
            ("distilled-600M: runs happily", MUTED),
            ("on CPU or the 6 GB GPU", MUTED)])
footer(s, 10)

# ---- Slide 11: emotion --------------------------------------------------
s = new_slide()
header(s, "Stage 8 · Sentiment", "Reading emotion from the voice, not the words")
pipeline_strip(s, 1.7, active=7)
panel_with_text(s, 0.7, 2.7, 6.05, 3.9, "Why tone, not text",
                ["Sentiment from translated text would be",
                 "doubly unreliable — noisy transcription",
                 "then cross-language translation.",
                 "· Tone survives both. Anger, calm and",
                 "  distress live in pitch, energy and rhythm,",
                 "  which are language-agnostic.",
                 "> So we read the sound, not the sentence."],
                accent=ACCENT)
panel_with_text(s, 6.95, 2.7, 5.68, 3.9, "The model",
                ["· A wav2vec2 network, self-supervised on",
                 "  raw audio, fine-tuned for emotion.",
                 "· Runs on each speaker turn and returns a",
                 "  label — neutral / happy / sad / angry —",
                 "  which we fold into a coarse polarity.",
                 "· Reported per turn AND summarised as each",
                 "  speaker's dominant tone."], accent=ACCENT2)
footer(s, 11)

# ---- Slide 12: putting it together --------------------------------------
s = new_slide()
header(s, "The record", "What one line of the report contains")
tf = textbox(s, 0.7, 1.7, 11.9, 0.7)
para(tf, "Every speaker turn becomes one structured record — this is the "
         "payload all eight stages were building toward.", 15, TEXT, first=True)
fields = [("speaker", "SPEAKER_01", ACCENT), ("time", "0:12.4 – 0:16.1", TEXT),
          ("language", "Farsi", ACCENT2), ("emotion", "angry", WARN),
          ("sentiment", "negative", WARN)]
x = 0.7
w = 2.3
for name, val, col in fields:
    rect(s, x, 2.55, w, 1.0, fill=PANEL)
    tf = textbox(s, x, 2.55, w, 1.0, anchor=MSO_ANCHOR.MIDDLE)
    para(tf, name.upper(), 9.5, MUTED, bold=True, first=True, align=PP_ALIGN.CENTER, space_after=3, font=MONO)
    para(tf, val, 12.5, col, bold=True, align=PP_ALIGN.CENTER)
    x += w + 0.09
rect(s, 0.7, 3.75, 11.93, 1.5, fill=PANEL2)
tf = textbox(s, 1.0, 3.95, 11.4, 1.2)
para(tf, "TEXT (original)", 9.5, ACCENT, bold=True, first=True, font=MONO, space_after=3)
para(tf, "چرا اینجا ایستاده‌ای؟", 15, TEXT, space_after=8)
para(tf, "عربي (translation)", 9.5, ACCENT2, bold=True, font=MONO, space_after=3)
para(tf, "لماذا تقف هنا؟", 15, TEXT)
panel_with_text(s, 0.7, 5.45, 11.93, 1.25, "And the two audio files",
                ["voice.wav — the isolated, enhanced speech    ·    "
                 "noise.wav — the isolated background, already labelled by Stage 4"],
                accent=ACCENT, hsize=11)
footer(s, 12)

# ---- Slide 13: limitations ----------------------------------------------
s = new_slide()
header(s, "Honesty", "What it can and cannot do", accent=WARN)
panel_with_text(s, 0.7, 1.8, 5.95, 4.9, "Real limits",
                ["· 'Where' means WHAT + WHEN. Compass",
                 "  direction needs a multi-mic array — it",
                 "  cannot be recovered from one mic.",
                 "· Distant, overlapping speech is the hard",
                 "  case for every stage.",
                 "· Speaker counts are estimates.",
                 "· Treat low-confidence transcripts as leads,",
                 "  not ground truth."], accent=WARN, bsize=12)
panel_with_text(s, 6.68, 1.8, 5.95, 4.9, "What it does well",
                ["· Genuinely isolates voice from steady noise.",
                 "· Labels the noise on a timeline.",
                 "· Handles Farsi and Hebrew robustly by",
                 "  testing both.",
                 "· Direct-to-Arabic translation, offline.",
                 "· Tone-based sentiment that survives",
                 "  translation.",
                 "· 100% local — nothing leaves the machine."],
                accent=GOOD, bsize=12)
footer(s, 13)

# ---- Slide 14: run it ---------------------------------------------------
s = new_slide()
header(s, "Run it", "Two commands")
code_panel(s, 0.7, 1.9, 11.93, 2.2, "PowerShell — one-time setup",
           [("cd \"C:\\Users\\Waleed.Alawneh\\Voice Isolator Code\"", TEXT),
            ("./setup.ps1        # venv + CUDA torch + all models", MUTED),
            ("$env:HF_TOKEN = \"hf_xxx\"   # for speaker counting", ACCENT2)])
code_panel(s, 0.7, 4.25, 11.93, 2.2, "PowerShell — analyse a recording",
           [(".\\.venv\\Scripts\\Activate.ps1", TEXT),
            ("python main.py \"path\\to\\recording.mp4\"", ACCENT),
            ("", TEXT),
            ("# → outputs\\recording\\report.md  +  report.json", MUTED)])
footer(s, 14)

# ---- Slide 15: closing --------------------------------------------------
s = new_slide()
tf = textbox(s, 0.7, 2.6, 11.9, 2.2, anchor=MSO_ANCHOR.MIDDLE)
para(tf, "From noise to meaning —", 40, TEXT, bold=True, first=True, space_after=4)
rich(tf, [("entirely on your own machine.", ACCENT)], 40, bold=True)
tf = textbox(s, 0.7, 4.7, 11.9, 1.0)
para(tf, "Separate · Enhance · Identify · Diarize · Transcribe · Translate · Feel",
     15, MUTED, first=True, font=MONO)
footer(s, 15)

out = "Voice Isolator - How It Works.pptx"
prs.save(out)
print(f"Saved {out} with {len(prs.slides._sldIdLst)} slides.")
