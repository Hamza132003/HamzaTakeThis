"""Stage 8: AI analysis of the whole recording (fully local LLM).

Feeds the complete transcript plus the pipeline's context signals (background
sounds, speaker count, overlaps, vocal emotions) to a small instruction-tuned
LLM and asks for an analyst-style summary:

  - Setting: where the speakers probably are (inferred from background noise)
  - Topic: what they are talking about
  - Speakers: who they might be (roles / relationship)
  - Activity: what they are probably doing
  - Key points worth attention

Produced in BOTH English and Arabic by the same model. Default model is
Qwen3-4B-Instruct (strong multilingual incl. Arabic), loaded 4-bit on the GPU
(~2.7 GB, everything else is evicted first) with a CPU fallback.
"""
from __future__ import annotations

from .utils import log, free_cuda, fmt_ts, guard_speechbrain_lazy

_EN_MARK = "### ENGLISH"
_AR_MARK = "### ARABIC"


def summarize(segments: list, noise: dict, diar: dict, overlaps: list,
              duration: float, cfg: dict, device: str, models=None) -> dict | None:
    if not cfg.get("enabled", True):
        return None
    if not segments:
        return {"english": "No intelligible speech was transcribed, so no "
                           "conversation analysis is possible.",
                "arabic": "لم يتم تفريغ أي كلام مفهوم، لذا لا يمكن تحليل المحادثة.",
                "model": "none"}
    if models is None:
        from .models import MANAGER as models

    try:
        guard_speechbrain_lazy()
        tok, model, dev, model_id = _load(cfg, device, models)
        prompt = _build_prompt(segments, noise, diar, overlaps, duration)
        text = _generate(tok, model, dev, prompt,
                         int(cfg.get("max_new_tokens", 600)))
        english, arabic = _split(text)
        log(f"AI summary generated ({model_id.split('/')[-1]} on {dev}).")
        return {"english": english, "arabic": arabic,
                "model": model_id.split("/")[-1]}
    except Exception as e:
        log(f"WARNING: AI summary failed ({e}).")
        return None


# ------------------------------------------------------------------ prompt
def _build_prompt(segments, noise, diar, overlaps, duration) -> str:
    lines = []
    for s in segments:
        who = s.get("speaker", "?")
        lang = s.get("language", "?")
        emo = s.get("emotion", "")
        flag = " [LOW CONFIDENCE — may be misheard]" if s.get("quality", {}).get("flagged") else ""
        line = f"[{fmt_ts(s['start'])}] {who} ({lang}"
        line += f", tone: {emo}" if emo else ""
        line += f"): {s.get('text', '')}"
        if s.get("english"):
            line += f"  | English: {s['english']}"
        lines.append(line + flag)
    # Cap prompt size: long prompts dominate generation time on a small GPU.
    transcript = "\n".join(lines[:120])[:7000]

    tags = ", ".join(f"{t['label']} ({t['confidence']:.2f})"
                     for t in (noise or {}).get("top_tags", [])[:8]) or "none detected"
    n_spk = diar.get("num_speakers", 1)
    ov = (f"{len(overlaps)} moment(s) of people talking over each other"
          if overlaps else "no overlapping speech")

    return f"""You are an audio-intelligence analyst. A {duration:.0f}-second recording was processed by a voice-isolation pipeline. Analyze it.

BACKGROUND SOUNDS DETECTED (from the noise track): {tags}
NUMBER OF SPEAKERS DETECTED: {n_spk}
OVERLAP: {ov}

TRANSCRIPT (speaker, language, vocal tone, original text, English translation):
{transcript}

Write your analysis in TWO languages, using EXACTLY these two section markers:

{_EN_MARK}
**Setting** — where the speakers most likely are, inferred from the background sounds and content (1-2 sentences).
**Topic** — what they are talking about (2-3 sentences).
**Speakers** — who each speaker might be: their possible role, relationship, and state of mind, based on what they say and their vocal tone (1-2 sentences per speaker).
**Activity** — what they are probably doing while speaking (1-2 sentences).
**Key points** — up to 3 bullet points worth attention (names, places, times, intentions).

{_AR_MARK}
(The same full analysis written natively in Modern Standard Arabic — not a word-for-word translation.)

Rules: base every claim ONLY on the evidence above; clearly say "possibly/probably" for inferences; if the transcript is too short or unreliable for a section, say so honestly rather than inventing details. Treat LOW CONFIDENCE lines as unreliable."""


# ------------------------------------------------------------------ model
def _has(mod: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(mod) is not None


def _load(cfg: dict, device: str, models):
    model_id = cfg.get("model", "Qwen/Qwen3-4B-Instruct-2507")

    # Quantization availability differs per machine (corporate networks block
    # some wheels), so try every viable path in quality/speed order.
    attempts = []
    if device == "cuda":
        if _has("torchao"):
            attempts.append(("int8-torchao", "cuda"))
        if _has("bitsandbytes"):
            attempts.append(("4bit-bnb", "cuda"))
        # NO plain-fp16 attempt: an 8 GB model on a 6 GB card doesn't OOM on
        # Windows — WDDM silently spills into shared memory and generation
        # crawls at 10-50x slower. CPU bf16 is faster than that.
    attempts.append(("bf16", "cpu"))

    last = None
    for quant, dev in attempts:
        try:
            def loader(quant=quant, dev=dev):
                import torch
                from transformers import (AutoModelForCausalLM, AutoTokenizer)
                log(f"Loading summarizer '{model_id}' ({quant}, {dev}) ...")
                tok = AutoTokenizer.from_pretrained(model_id)
                if quant == "int8-torchao":
                    from transformers import TorchAoConfig
                    m = AutoModelForCausalLM.from_pretrained(
                        model_id, quantization_config=TorchAoConfig("int8_weight_only"),
                        dtype=torch.bfloat16, device_map="cuda")
                elif quant == "4bit-bnb":
                    from transformers import BitsAndBytesConfig
                    bnb = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4")
                    m = AutoModelForCausalLM.from_pretrained(
                        model_id, quantization_config=bnb, device_map="cuda")
                else:
                    m = AutoModelForCausalLM.from_pretrained(
                        model_id, dtype=torch.bfloat16)
                m.eval()
                return (tok, m)

            tok, model = models.get(f"llm:{model_id}:{quant}:{dev}", loader, dev)
            return tok, model, dev, model_id
        except Exception as e:
            last = e
            log(f"WARNING: summarizer load failed ({quant}/{dev}): {e}")
            models.evict(f"llm:{model_id}:{quant}:{dev}")
            free_cuda()
    raise RuntimeError(f"no summarizer could be loaded: {last}")


def _generate(tok, model, dev, prompt: str, max_new_tokens: int) -> str:
    import torch
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=4096)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=True, temperature=0.4, top_p=0.9,
                             pad_token_id=tok.eos_token_id)
    gen = out[0][enc["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def _split(text: str) -> tuple[str, str]:
    english, arabic = text, ""
    if _AR_MARK in text:
        head, arabic = text.split(_AR_MARK, 1)
        english = head
    if _EN_MARK in english:
        english = english.split(_EN_MARK, 1)[1]
    return english.strip(), arabic.strip()
