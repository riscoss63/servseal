"""Running a live model over the probes. The only module that imports torch.

The measured object is the full softmax at every probe position, float32, greedy-free:
no sampling is involved in weights-mode snapshots, so on CPU the snapshot of a given
model is bit-deterministic -- an unchanged deployment attests at Hellinger distance
exactly zero, and anything above the threshold is signal, not luck.

Perplexity over the probes is recorded into the metadata not because it is a good
change detector -- the point of this tool is that it is not (a top-p 0.95 filter moves
the distribution by 0.149 while moving perplexity by 0.00) -- but because showing the
two numbers side by side in the report is the honest way to make that argument.
"""
from __future__ import annotations

import time

import numpy as np

from .probes import load_probes, probe_id
from .snapshot import Snapshot

__all__ = ["snapshot_model", "softmax_over_probes"]

_DTYPES = {"float32": None, "bfloat16": "bfloat16", "float16": "float16"}


def softmax_over_probes(model, tokenizer, texts, *, max_positions=1500,
                        max_length=96, template=None):
    """Full next-token distributions (positions x vocab, float32) plus realised-token
    log-probabilities for perplexity. One forward pass per probe text."""
    import torch

    rows, logprobs = [], []
    for t in texts:
        if sum(r.shape[0] for r in rows) >= max_positions:
            break
        text = template.replace("{text}", t) if template else t
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_length).input_ids
        with torch.no_grad():
            logits = model(ids).logits[0].float()
        p = torch.softmax(logits, -1).numpy().astype(np.float32)
        rows.append(p)
        tg = ids[0, 1:].numpy()
        n = min(len(tg), p.shape[0] - 1)
        logprobs.append(np.log(np.clip(p[np.arange(n), tg[:n]], 1e-300, None)))
    P = np.concatenate(rows)[:max_positions]
    ppl = float(np.exp(-np.mean(np.concatenate(logprobs)))) if logprobs else float("nan")
    return P, ppl


def snapshot_model(model_id, *, probes=None, D=256, seed=0, max_positions=1500,
                   max_length=96, template=None, dtype="float32", device="cpu",
                   label=None, return_P=False):
    """Load a model, run the probes, return the Snapshot (and optionally the raw P)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
    texts = load_probes(probes)
    tok = AutoTokenizer.from_pretrained(model_id)
    torch_dtype = getattr(torch, dtype) if _DTYPES[dtype] else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
    if dtype != "float32":
        # measure in float32 arithmetic what the narrowed weights produce, which is the
        # deployment-relevant object; keeping reduced-precision *arithmetic* as well
        # would measure the host's kernel quirks along with the model
        model = model.to(torch.float32)
    model = model.to(device).eval()

    t0 = time.perf_counter()
    P, ppl = softmax_over_probes(model, tok, texts, max_positions=max_positions,
                                 max_length=max_length, template=template)
    snap = Snapshot.from_distributions(
        P, probe=probe_id(texts), model=label or str(model_id), D=D, seed=seed,
        template=template, dtype=dtype,
        extra={"perplexity": round(ppl, 4), "probe_file": probes or "default-v1",
               "max_length": int(max_length),
               "snapshot_seconds": round(time.perf_counter() - t0, 1)})
    return (snap, P) if return_P else snap
