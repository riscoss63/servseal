"""The battery: known perturbations of real models against the verdicts they must get.

Seven deployment events whose ground truth is known by construction, applied to GPT-2
where a deployment would apply them, each pushed through the full product path --
Snapshot.from_distributions -> compare -> classify -- plus a second architecture
(pythia-160m) to check none of it is a GPT-2 accident. The battery asserts:

  unchanged model .......... SEALED, at Hellinger distance ~0 (weights mode is
                             deterministic; a false positive here is a product bug)
  bfloat16 weights ......... CHANGED/minor, precision signature
  int8 weights ............. CHANGED/major, weight-level signature
  top-p 0.95 at serving .... CHANGED/major, serving-layer signature -- the change
                             perplexity and greedy diffs cannot see
  temperature 1.05 ......... CHANGED/minor
  wrong prompt template .... CHANGED, template/substitution class
  distilgpt2 substituted ... CHANGED/major (same tokenizer, different model)

The perturbation helpers are the ones used in the sqsketch paper's LLM benchmark,
copied here so this repository stands alone.

    python e2e_real_models.py          # writes outputs/e2e_output.txt numbers
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servseal.probes import load_probes, probe_id           # noqa: E402
from servseal.runner import softmax_over_probes             # noqa: E402
from servseal.snapshot import Snapshot                      # noqa: E402
from servseal.verdict import classify                       # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
POSITIONS, MAXLEN, D = 1500, 96, 256


# ------------------------------------------------- perturbations (as in the paper)

def round_weights(model, dtype):
    """What a deployment does when it serves in a narrower float type."""
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.to(dtype).to(torch.float32))
    return model


def quantise_weights(model, bits=8):
    """Per-tensor symmetric integer quantisation, the cheapest realistic scheme."""
    lv = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() < 2:
                continue
            s = p.abs().max() / lv
            if s > 0:
                p.copy_(torch.round(p / s) * s)
    return model


def serve_top_p(P, p_keep=0.95):
    srt = -np.sort(-P, 1)
    thr = srt[np.arange(len(P)), np.argmax(np.cumsum(srt, 1) >= p_keep, 1)][:, None]
    Q = np.where(P >= thr, P, 0.0)
    return (Q / Q.sum(1, keepdims=True)).astype(np.float32)


def serve_temperature(P, t=1.05):
    lg = np.log(np.clip(P.astype(np.float64), 1e-300, None)) / t
    lg -= lg.max(1, keepdims=True)
    Q = np.exp(lg)
    return (Q / Q.sum(1, keepdims=True)).astype(np.float32)


# ----------------------------------------------------------------------- helpers

def load(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    return tok, mdl


def snap_from_P(P, texts, model_label, ppl, template=None):
    return Snapshot.from_distributions(
        P, probe=probe_id(texts), model=model_label, D=D, template=template,
        extra={"perplexity": round(ppl, 4), "probe_file": "default-v1"})


def run(model, tok, texts, template=None):
    t0 = time.perf_counter()
    P, ppl = softmax_over_probes(model, tok, texts, max_positions=POSITIONS,
                                 max_length=MAXLEN, template=template)
    return P, ppl, time.perf_counter() - t0


FAILURES = []


def show(name, ref, cand, expect_status, expect_severity=None, expect_sig=None):
    m = ref.compare(cand)
    v = classify(m)
    t1 = m.get("top1_agreement")
    ppl_r = m["ref_meta"].get("perplexity")
    ppl_c = m["cand_meta"].get("perplexity")
    klb = m.get("mean_kl_lower_bound", max(0.0, m["aggregate_kl_lower_bound"]))
    print(f"  {name:<26} mean_h={m['mean_hellinger']:.4f}  "
          f"top1={'  n/a' if t1 is None else f'{t1:.3f}'}  "
          f"ppl {ppl_r:.2f}->{ppl_c:.2f}  "
          f"kl>={klb:.3f}  "
          f"-> {v.status.upper()}/{v.severity}  [{v.signature}]")
    ok = v.status == expect_status
    if expect_severity:
        ok &= v.severity == expect_severity
    if expect_sig:
        ok &= expect_sig in v.signature
    if not ok:
        FAILURES.append(f"{name}: expected {expect_status}/{expect_severity}"
                        f"/{expect_sig}, got {v.status}/{v.severity}/{v.signature}")
        print(f"    ^^ ASSERTION FAILED: expected {expect_status}"
              f"/{expect_severity}/{expect_sig}")
    return m, v


def main():
    os.makedirs(CACHE, exist_ok=True)
    texts = load_probes()
    print("=" * 100)
    print(f"E2E BATTERY: known perturbations, product-path verdicts "
          f"({len(texts)} probes, {POSITIONS} positions, D={D})")
    print("=" * 100)

    # ------------------------------------------------------------------- GPT-2
    tok, base_model = load("gpt2")
    P_base, ppl_base, dt = run(base_model, tok, texts)
    print(f"  gpt2 reference: {P_base.shape[0]} positions, vocab {P_base.shape[1]}, "
          f"perplexity {ppl_base:.2f}, forward {dt:.0f}s")
    ref = snap_from_P(P_base, texts, "gpt2@fp32", ppl_base)
    np.save(os.path.join(CACHE, "gpt2_P_base.npy"), P_base.astype(np.float16))
    ref.save(os.path.join(CACHE, "gpt2_ref.seal.npz"))

    print("\n  variant                    measurements"
          + " " * 34 + "verdict")
    # unchanged: a fresh forward pass of the same weights
    P_again, ppl_again, _ = run(base_model, tok, texts)
    show("unchanged (re-run)", ref,
         snap_from_P(P_again, texts, "gpt2@fp32/rerun", ppl_again),
         "sealed")

    # serving-layer changes: applied to the probabilities, as serving does
    P_topp = serve_top_p(P_base)
    show("serve: top-p 0.95", ref,
         snap_from_P(P_topp, texts, "gpt2@fp32+top-p0.95", ppl_base),
         "changed", "major", "serving-layer")
    show("serve: temperature 1.05", ref,
         snap_from_P(serve_temperature(P_base), texts, "gpt2@fp32+T1.05", ppl_base),
         "changed", "minor")

    # weight-level changes: fresh model each time, weights modified in place
    _, m2 = load("gpt2")
    P_bf16, ppl_bf16, _ = run(round_weights(m2, torch.bfloat16), tok, texts)
    show("weights -> bfloat16", ref,
         snap_from_P(P_bf16, texts, "gpt2@bf16", ppl_bf16),
         "changed", "minor", "precision")
    del m2
    _, m3 = load("gpt2")
    P_int8, ppl_int8, _ = run(quantise_weights(m3), tok, texts)
    show("weights -> int8/tensor", ref,
         snap_from_P(P_int8, texts, "gpt2@int8", ppl_int8),
         "changed", "major", "weight-level")
    del m3

    # the misdiagnosis classic: same weights, wrong prompt template
    P_tpl, ppl_tpl, _ = run(base_model, tok, texts,
                            template="[INST] {text} [/INST]")
    show("wrong chat template", ref,
         snap_from_P(P_tpl, texts, "gpt2@fp32+template", ppl_tpl,
                     template="[INST] {text} [/INST]"),
         "changed", None, "template")

    # substitution: a different model behind the same tokenizer
    _, dmodel = load("distilgpt2")
    P_dist, ppl_dist, _ = run(dmodel, tok, texts)
    show("distilgpt2 substituted", ref,
         snap_from_P(P_dist, texts, "distilgpt2@fp32", ppl_dist),
         "changed", "major")
    del dmodel

    # ------------------------------------------- second architecture spot check
    print("\n  second architecture (pythia-160m):")
    ptok, pmodel = load("EleutherAI/pythia-160m")
    Pp, pppl, dt = run(pmodel, ptok, texts)
    pref = snap_from_P(Pp, texts, "pythia-160m@fp32", pppl)
    np.save(os.path.join(CACHE, "pythia_P_base.npy"), Pp.astype(np.float16))
    pref.save(os.path.join(CACHE, "pythia_ref.seal.npz"))
    Pp2, pppl2, _ = run(pmodel, ptok, texts)
    show("unchanged (re-run)", pref,
         snap_from_P(Pp2, texts, "pythia-160m@fp32/rerun", pppl2), "sealed")
    show("serve: top-p 0.95", pref,
         snap_from_P(serve_top_p(Pp), texts, "pythia+top-p0.95", pppl),
         "changed", "major", "serving-layer")

    # gpt2 and pythia use different tokenizers: the product must refuse, not compare
    r = pref.compare(ref)
    v = classify(r)
    print(f"  {'pythia vs gpt2':<26} -> {v.status.upper()}  [{r.get('incomparable')}]")
    if v.status != "incomparable":
        FAILURES.append("cross-tokenizer comparison was not refused")

    print("\n" + "=" * 100)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} ASSERTION(S) FAILED")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("RESULT: every perturbation received the verdict its ground truth requires")
    return 0


if __name__ == "__main__":
    sys.exit(main())
