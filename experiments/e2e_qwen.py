"""The battery on a current architecture: Qwen2.5-0.5B, vocabulary 151,936.

GPT-2 is the right model to calibrate on -- small, fast, boring -- and the wrong model
to be judged on in 2026. This run re-asserts the product's verdicts on a modern model
with a 3x larger vocabulary and a much sharper output distribution (median effective
support ~7 against GPT-2's ~11, mass outside the top-64 only 0.087), where a tail-blind
fingerprint has the least to miss and this tool's margin should be at its thinnest.

Two cases differ from the GPT-2 battery by design, and both are the point:

  bfloat16 rounding ....... Qwen2.5's weights are natively bf16, so the "change" is an
                            exact no-op and the battery asserts SEALED at 0.0000 -- the
                            true-negative test a monitoring tool must pass to be
                            trusted with a pager.
  Qwen3-0.6B served ....... the next model generation behind the same tokenizer: the
                            silent-upgrade scenario providers actually do.

The template case uses Qwen's real ChatML wrapper -- serving a base model behind a
chat-templated endpoint is the deployment bug this measures.

    python e2e_qwen.py            # writes outputs/e2e_qwen_output.txt
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import e2e_real_models as B                                   # noqa: E402
from servseal.probes import load_probes                       # noqa: E402
from servseal.snapshot import Snapshot                        # noqa: E402
from servseal.verdict import classify                         # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B"
NEXT_GEN = "Qwen/Qwen3-0.6B"
CHATML = "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


def main():
    os.makedirs(B.CACHE, exist_ok=True)
    texts = load_probes()
    print("=" * 100)
    print(f"E2E BATTERY, CURRENT ARCHITECTURE: {MODEL} "
          f"({len(texts)} probes, {B.POSITIONS} positions, D={B.D})")
    print("=" * 100)

    tok, base = B.load(MODEL)
    P_base, ppl_base, dt = B.run(base, tok, texts)
    print(f"  reference: {P_base.shape[0]} positions, vocab {P_base.shape[1]:,}, "
          f"perplexity {ppl_base:.2f}, forward {dt:.0f}s")
    ref = B.snap_from_P(P_base, texts, "qwen2.5-0.5b@bf16", ppl_base)
    np.save(os.path.join(B.CACHE, "qwen_P_base.npy"), P_base.astype(np.float16))
    ref.save(os.path.join(B.CACHE, "qwen_ref.seal.npz"))

    print("\n  variant                    measurements"
          + " " * 34 + "verdict")
    P2, ppl2, _ = B.run(base, tok, texts)
    B.show("unchanged (re-run)", ref,
           B.snap_from_P(P2, texts, "qwen2.5-0.5b/rerun", ppl2), "sealed")

    # the true negative: rounding natively-bf16 weights to bf16 changes nothing, and a
    # tool that cries wolf here cannot be trusted with a pager
    _, m2 = B.load(MODEL)
    P_bf16, ppl_bf16, _ = B.run(B.round_weights(m2, torch.bfloat16), tok, texts)
    B.show("weights -> bfloat16 (no-op)", ref,
           B.snap_from_P(P_bf16, texts, "qwen@bf16-round", ppl_bf16), "sealed")
    del m2

    B.show("serve: top-p 0.95", ref,
           B.snap_from_P(B.serve_top_p(P_base), texts, "qwen+top-p0.95", ppl_base),
           "changed", "major", "serving-layer")
    B.show("serve: temperature 1.05", ref,
           B.snap_from_P(B.serve_temperature(P_base), texts, "qwen+T1.05", ppl_base),
           "changed", "minor")

    # int8 measured 0.1155 at 500 positions in the sqsketch benchmark -- close to the
    # moderate/major boundary, so the battery asserts the class, not the severity
    _, m3 = B.load(MODEL)
    P_int8, ppl_int8, _ = B.run(B.quantise_weights(m3), tok, texts)
    B.show("weights -> int8/tensor", ref,
           B.snap_from_P(P_int8, texts, "qwen@int8", ppl_int8),
           "changed", None, "weight-level")
    del m3

    # a base model served behind a chat template: the real ChatML wrapper
    P_tpl, ppl_tpl, _ = B.run(base, tok, texts, template=CHATML)
    B.show("ChatML template applied", ref,
           B.snap_from_P(P_tpl, texts, "qwen+chatml", ppl_tpl, template=CHATML),
           "changed", None, "template")

    # the silent upgrade: the next generation behind the same endpoint
    ntok, nmodel = B.load(NEXT_GEN)
    same_vocab = None
    try:
        Pn, ppln, _ = B.run(nmodel, ntok, texts)
        same_vocab = Pn.shape[1] == P_base.shape[1]
        B.show("Qwen3-0.6B served instead", ref,
               B.snap_from_P(Pn, texts, "qwen3-0.6b", ppln), "changed")
    finally:
        del nmodel

    # different architecture entirely: must refuse, not compare
    g = os.path.join(B.CACHE, "gpt2_ref.seal.npz")
    if os.path.exists(g):
        r = ref.compare(Snapshot.load(g))
        v = classify(r)
        print(f"  {'qwen vs gpt2':<26} -> {v.status.upper()}  [{r.get('incomparable')}]")
        if v.status != "incomparable":
            B.FAILURES.append("cross-architecture comparison was not refused")

    print("\n" + "=" * 100)
    if B.FAILURES:
        print(f"RESULT: {len(B.FAILURES)} ASSERTION(S) FAILED")
        for f in B.FAILURES:
            print(f"  - {f}")
        return 1
    print("RESULT: every perturbation received the verdict its ground truth requires")
    if same_vocab is not None:
        print(f"  (Qwen3 shares Qwen2.5's vocabulary size: {same_vocab})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
