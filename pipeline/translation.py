"""Stage 6: translate each transcribed segment into Arabic with NLLB-200.

Translates the SOURCE-LANGUAGE transcript directly (fa/he/en -> ar) rather
than pivoting through Whisper's English: if Whisper hallucinated the English,
a pivot would propagate that hallucination into the Arabic. The English view
in the UI still comes from Whisper's own translate task.

Quality-first: nllb-200-distilled-1.3B in fp16 on the GPU (fits 6 GB alone;
the runner evicts Whisper first), falling back to the 600M model and then
CPU on failure. Segments are batched through the tokenizer/generate.
"""
from __future__ import annotations

from .utils import (log, free_cuda, guard_speechbrain_lazy, check_cancel,
                    JobCancelled)

# Whisper language code -> NLLB (FLORES-200) code
NLLB_SRC = {
    "fa": "pes_Arab",   # Western Persian / Farsi
    "he": "heb_Hebr",   # Hebrew
    "iw": "heb_Hebr",   # legacy Hebrew code
    "ar": "arb_Arab",
    "en": "eng_Latn",
}


def translate_segments(segments: list, cfg: dict, device: str,
                       models=None) -> list:
    if not cfg.get("enabled", True) or not segments:
        return segments
    if models is None:
        from .models import MANAGER as models

    # Nothing to do if every segment is already Arabic (source language = ar).
    tgt = cfg.get("target_lang", "arb_Arab")
    if all(NLLB_SRC.get(s.get("language")) == tgt for s in segments):
        for s in segments:
            s["arabic"] = s["text"]
        log("Source already Arabic - skipping translation.")
        return segments

    try:
        guard_speechbrain_lazy()
        tok, model, dev, model_id = _load(cfg, device, models)
        tgt_id = tok.convert_tokens_to_ids(tgt)
        use_pivot = str(cfg.get("source", "native")).lower() == "pivot"
        batch_size = max(1, int(cfg.get("batch_size", 8)))
        num_beams = max(1, int(cfg.get("num_beams", 5)))

        # Group by source language so each batch shares tok.src_lang.
        todo = []
        for seg in segments:
            if NLLB_SRC.get(seg.get("language")) == tgt:   # already Arabic
                seg["arabic"] = seg["text"]
                continue
            english = (seg.get("english") or "").strip()
            if use_pivot and english:
                src, text = "eng_Latn", english
            else:
                src = NLLB_SRC.get(seg.get("language"), "eng_Latn")
                text = seg["text"]
            if not text.strip():
                seg["arabic"] = ""
                continue
            todo.append((seg, src, text))

        for i in range(0, len(todo), batch_size):
            check_cancel()
            chunk = todo[i:i + batch_size]
            # Split further if the batch mixes source languages.
            by_src = {}
            for item in chunk:
                by_src.setdefault(item[1], []).append(item)
            for src, items in by_src.items():
                _translate_batch(tok, model, dev, src, tgt_id, items, num_beams)

        free_cuda()
        log(f"Translated {len(segments)} segment(s) to Arabic "
            f"[{model_id.split('/')[-1]}, "
            f"{'english pivot' if use_pivot else 'native source'}].")
        return segments

    except JobCancelled:
        raise
    except Exception as e:
        log(f"WARNING: translation failed ({e}).")
        for seg in segments:
            seg.setdefault("arabic", "")
        return segments


def _translate_batch(tok, model, device, src, tgt_id, items, num_beams):
    import torch
    tok.src_lang = src
    texts = [t for _, _, t in items]
    enc = tok(texts, return_tensors="pt", truncation=True,
              max_length=512, padding=True)
    if device == "cuda":
        enc = {k: v.to("cuda") for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, forced_bos_token_id=tgt_id,
                             max_length=512, num_beams=num_beams)
    decoded = tok.batch_decode(out, skip_special_tokens=True)
    for (seg, _, _), ar in zip(items, decoded):
        seg["arabic"] = ar.strip()


def _load(cfg: dict, device: str, models):
    """Load NLLB with quality-first fallbacks: 1.3B fp16 GPU -> 600M -> CPU."""
    primary = cfg.get("model", "facebook/nllb-200-distilled-1.3B")
    fallback = cfg.get("fallback_model", "facebook/nllb-200-distilled-600M")

    attempts = []
    if device == "cuda":
        attempts += [(primary, "cuda"), (fallback, "cuda")]
    attempts += [(fallback, "cpu"), (primary, "cpu")]

    last_err = None
    for model_id, dev in attempts:
        try:
            def loader(model_id=model_id, dev=dev):
                import torch
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                log(f"Loading translator '{model_id}' on {dev} ...")
                tok = AutoTokenizer.from_pretrained(model_id)
                kwargs = {"torch_dtype": torch.float16} if dev == "cuda" else {}
                m = AutoModelForSeq2SeqLM.from_pretrained(model_id, **kwargs)
                if dev == "cuda":
                    m = m.to("cuda")
                m.eval()
                return (tok, m)

            tok, model = models.get(f"nllb:{model_id}:{dev}", loader, dev)
            return tok, model, dev, model_id
        except Exception as e:
            last_err = e
            log(f"WARNING: could not load '{model_id}' on {dev} ({e}); "
                f"trying next option.")
            models.evict(f"nllb:{model_id}:{dev}")
            free_cuda()
    raise RuntimeError(f"no NLLB model could be loaded: {last_err}")
