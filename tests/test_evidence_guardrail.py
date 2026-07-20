"""Interim evidence-safety guardrail and the replacement branch score.

These are pure-logic tests (no models, no audio) so they run everywhere.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("numpy")

from pipeline.transcription import (  # noqa: E402
    UNRELIABLE_PLACEHOLDER,
    annotate_cross_branch_support,
    apply_evidence_guardrail,
    branch_utility,
    segment_utility,
)


def _seg(start, end, alp=-0.3, nsp=0.0, wp=0.9, text="hello world"):
    return {"start": start, "end": end, "alp": alp, "nsp": nsp, "text": text,
            "words": [{"probability": wp}]}


# --------------------------------------------------------- branch scoring
def test_longer_correct_transcript_is_not_penalised():
    """The defect that motivated this: sum(avg_logprob * len(text)) rewarded
    SHORTER transcripts because avg_logprob is negative."""
    short = [_seg(0, 2, text="hi")]
    long = [_seg(0, 2, text="hi"), _seg(2, 8, text="a much longer continuation")]
    legacy_short = sum(s["alp"] * len(s["text"]) for s in short)
    legacy_long = sum(s["alp"] * len(s["text"]) for s in long)
    assert legacy_short > legacy_long, "documents the old defective ordering"
    assert branch_utility(long) > branch_utility(short), \
        "new utility must prefer the longer, equally-confident transcript"


def test_utility_is_non_negative_and_monotone_in_duration():
    assert segment_utility(_seg(0, 1)) >= 0
    assert segment_utility(_seg(0, 4)) > segment_utility(_seg(0, 1))


def test_utility_rewards_confidence_not_character_count():
    confident = _seg(0, 3, alp=-0.1, wp=0.95, text="short")
    unsure = _seg(0, 3, alp=-1.5, wp=0.3, text="a very long piece of text here")
    assert segment_utility(confident) > segment_utility(unsure)


def test_no_speech_probability_suppresses_utility():
    assert segment_utility(_seg(0, 3, nsp=0.9)) < segment_utility(_seg(0, 3, nsp=0.0))


# ---------------------------------------------------- cross-branch support
def test_agreeing_branches_give_support_and_keep_alternatives():
    results = [{"start": 0.0, "end": 3.0, "text": "the meeting starts now",
                "quality": {"avg_logprob": -0.3}}]
    others = [{"name": "voice_16k",
               "segments": [{"start": 0.1, "end": 3.1,
                             "text": "the meeting starts now"}]}]
    annotate_cross_branch_support(results, others)
    q = results[0]["quality"]
    assert q["cross_branch_support"] == 1
    assert results[0]["alternatives"][0]["similarity"] >= 0.9


def test_disagreement_is_exposed_not_resolved():
    results = [{"start": 0.0, "end": 3.0, "text": "thank you for watching",
                "quality": {"avg_logprob": -0.6}}]
    others = [{"name": "voice_16k",
               "segments": [{"start": 0.0, "end": 3.0,
                             "text": "completely different words spoken"}]}]
    annotate_cross_branch_support(results, others)
    assert results[0]["quality"]["cross_branch_support"] == 0
    assert results[0]["alternatives"], "the losing hypothesis must stay visible"


# ------------------------------------------------------------- guardrail
def test_flagged_hallucination_is_withheld_and_not_translated():
    r = {"start": 0.0, "end": 3.0, "text": "متن فارسی",
         "english": "Thank you for watching the video.", "arabic": "شكرا",
         "quality": {"flagged": True, "reason": "signature noise phrase in English"}}
    apply_evidence_guardrail([r])
    assert r["text"] == UNRELIABLE_PLACEHOLDER
    assert r["english"] == "" and r["arabic"] == ""
    assert r["translation_suppressed"] is True
    assert r["text_rejected"] == "متن فارسی"          # nothing deleted
    assert r["english_rejected"] == "Thank you for watching the video."
    assert r["quality"]["unreliable"] is True
    assert r["quality"]["unreliable_reasons"]


def test_uncorroborated_weak_segment_is_withheld():
    r = {"start": 0.0, "end": 3.0, "text": "invented text", "english": "x",
         "quality": {"avg_logprob": -0.9, "cross_branch_support": 0,
                     "branches_compared": 1}}
    apply_evidence_guardrail([r])
    assert r["quality"]["unreliable"] is True
    assert r["text"] == UNRELIABLE_PLACEHOLDER


def test_corroborated_segment_survives_untouched():
    r = {"start": 0.0, "end": 3.0, "text": "real speech", "english": "real speech",
         "arabic": "كلام", "quality": {"avg_logprob": -0.3,
                                        "cross_branch_support": 1,
                                        "branches_compared": 1}}
    apply_evidence_guardrail([r])
    assert r["text"] == "real speech"
    assert r["quality"]["unreliable"] is False
    assert "translation_suppressed" not in r


def test_confident_uncorroborated_segment_is_kept():
    """Only WEAK uncorroborated text is withheld; a confident single-branch
    segment must not be destroyed (that would delete real speech)."""
    r = {"start": 0.0, "end": 3.0, "text": "clear speech", "english": "clear",
         "quality": {"avg_logprob": -0.2, "cross_branch_support": 0,
                     "branches_compared": 1}}
    apply_evidence_guardrail([r])
    assert r["quality"]["unreliable"] is False
    assert r["text"] == "clear speech"


def test_known_outro_phrases_trigger_the_guardrail():
    from pipeline.transcription import _flag_inconsistent
    for phrase in ("Thank you for watching the video.",
                   "Don't forget to subscribe to my channel.",
                   "This is the end of this video."):
        r = {"start": 0.0, "end": 2.0, "text": "برخی متن", "english": phrase,
             "quality": {"avg_logprob": -0.5}}
        _flag_inconsistent([r])
        apply_evidence_guardrail([r])
        assert r["quality"]["unreliable"] is True, phrase
        assert r["english"] == "" and r["arabic"] == ""


def test_translation_stage_skips_suppressed_segments():
    from pipeline.translation import translate_segments
    segs = [{"language": "fa", "text": UNRELIABLE_PLACEHOLDER,
             "translation_suppressed": True,
             "quality": {"unreliable": True}}]
    out = translate_segments(segs, {"enabled": True}, "cpu", models=None)
    assert out[0]["arabic"] == ""


def test_arabic_source_does_not_copy_placeholder_into_arabic():
    """Regression: the 'source already Arabic' shortcut copied `text` into
    `arabic`, which published the UNRELIABLE placeholder as if it were the
    Arabic transcript."""
    from pipeline.translation import translate_segments
    segs = [{"language": "ar", "text": UNRELIABLE_PLACEHOLDER,
             "translation_suppressed": True, "quality": {"unreliable": True}},
            {"language": "ar", "text": "كلام حقيقي", "quality": {}}]
    out = translate_segments(segs, {"enabled": True}, "cpu", models=None)
    assert out[0]["arabic"] == "", "placeholder must never appear as Arabic"
    assert out[1]["arabic"] == "كلام حقيقي", "real Arabic must pass through"


# --------------------------------------------------------------- coverage
def test_coverage_accounts_for_every_second():
    from pipeline.runner import _coverage
    segs = [{"start": 0.0, "end": 5.0, "quality": {"unreliable": False}},
            {"start": 10.0, "end": 12.0, "quality": {"unreliable": True}}]
    diar = {"speech_regions": [{"start": 0.0, "end": 12.0}]}
    c = _coverage(42.0, segs, diar)
    assert c["transcribed_sec"] == 5.0
    assert c["withheld_unreliable_sec"] == 2.0
    assert c["not_decoded_sec"] == 35.0
    assert c["duration_sec"] == 42.0
    assert "not evidence that nothing was said" in c["note"]
