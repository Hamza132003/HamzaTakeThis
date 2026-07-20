# Branch selection and the evidence guardrail

## The deleted whole-file score (defect 3.2)

The original transcription stage chose ONE audio branch for the entire
recording with:

    score = sum(avg_logprob * len(text) for segment in hypothesis)

`avg_logprob` is negative, so this score *decreases* as a transcript grows.
A truncated or hallucinated branch could therefore beat a longer, equally
confident, correct one. It is deleted; `tests/test_evidence_guardrail.py`
pins the old ordering as a documented defect and asserts the new behaviour.

## Replacement: segment utility

    U(segment) = duration * (1 + mean_word_prob) * exp(avg_logprob) * (1 - no_speech_prob)
    U(branch)  = sum of segment utilities

Properties: non-negative, monotonically non-decreasing in accepted speech
duration, rewards acoustic confidence rather than character count, and
penalises segments the model itself considers non-speech. Uncalibrated by
design — this is routing only, never a confidence claim (Phase 6 owns
calibration).

## Branches are retained, not discarded

Every branch is decoded and kept. The highest-utility branch becomes the
primary transcript; the others are compared segment-by-segment and stored as
`alternatives` with a token-overlap similarity. Disagreements are exposed in
the report and UI rather than resolved silently in favour of the more fluent
option.

## Interim evidence guardrail (NOT calibrated confidence)

A segment is marked `unreliable` when:
1. a hallucination signature fires (video-outro phrases, subtitle credits,
   music/noise tags, wild source/English length mismatch); or
2. no independent branch corroborates it AND `avg_logprob < -0.5`.

For those segments the pipeline:
- replaces the displayed source text with `[UNRELIABLE — REVIEW AUDIO]`;
- keeps the original in `text_rejected` / `english_rejected` (nothing is
  deleted — the interval and the rejected hypothesis stay recorded);
- clears and suppresses translation, so a hallucination is never laundered
  into English and Arabic;
- surfaces the raw scores and the reasons in the report and dashboard.

A *confident* single-branch segment is NOT withheld: withholding it would
delete real speech, which is the opposite failure.

## Coverage accounting

`report.json.coverage` accounts for the whole recording:
`transcribed_sec`, `withheld_unreliable_sec`, `diarization_gated_speech_sec`
and `not_decoded_sec`. `not_decoded_sec` includes silence AND any speech
excluded by diarization gating — it is explicitly not evidence that nothing
was said.
