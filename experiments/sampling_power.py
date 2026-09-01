"""How many sampled tokens buy how much detection power, on a real model.

The question a buyer of API mode asks first: my endpoint returns sampled tokens only --
how many do I need before modelseal can tell that the provider turned on top-p 0.95?
This run answers it on GPT-2 distributions computed from the real model (cached by the
e2e battery), for the two serving changes with known ground truth:

  top-p 0.95 ......... true mean per-position Hellinger 0.15, invisible to perplexity
  temperature 1.05 ... true mean 0.044, the hard case

Protocol as in modelseal.sampler: one sampled next token per probe position per call,
budget spread evenly. Bands calibrated on the unchanged endpoint (reps below), false
positives checked on held-out null draws, power on independent alternative draws.

The sweep uses a vectorised sampler for speed; its statistics are asserted equal to
modelseal.sampler.statistics on a shared stream before anything is measured, so the
numbers below are the product's numbers, not a stand-in's.

    python sampling_power.py           # needs experiments/cache from e2e_real_models
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelseal.sampler import _codes, sample_stream, statistics       # noqa: E402
from modelseal.snapshot import Snapshot                               # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
BUDGETS = [1500, 5000, 15000, 50000, 150000]
REPS_CAL, REPS_NULL, REPS_ALT = 200, 100, 100
ALPHA = 0.05


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


def stats_all_reps(P, snapshot, total, reps, seed):
    """(S1, S2) for `reps` independent streams at budget `total` -- vectorised over
    reps, position by position, computing exactly what sampler.statistics computes."""
    meta = snapshot.meta
    D, vocab = meta["D"], meta["vocab_size"]
    n_pos = P.shape[0]
    bucket, sign = _codes(vocab, D, meta["seed"])
    ref = snapshot.positions.astype(np.float64)
    ref_nrm = np.linalg.norm(ref, axis=1)
    rng = np.random.default_rng(seed)

    per = np.full(n_pos, total // n_pos, dtype=np.int64)
    per[: total - per.sum()] += 1
    s1 = np.zeros(reps)
    s2 = np.zeros(reps)
    seen = 0
    for i in range(n_pos):
        if per[i] == 0:
            continue
        seen += 1
        cum = np.cumsum(P[i], dtype=np.float64)
        cum /= cum[-1]
        toks = np.minimum(np.searchsorted(cum, rng.random((reps, per[i])),
                                          side="right"), vocab - 1)
        s2 += np.sum(toks == snapshot.argmax[i], axis=1)
        keys, counts = np.unique(
            (np.arange(reps)[:, None] * vocab + toks).ravel(), return_counts=True)
        kr, kt = keys // vocab, keys % vocab
        emp = np.zeros((reps, D))
        np.add.at(emp.ravel(), kr * D + bucket[kt],
                  sign[kt] * np.sqrt(counts / per[i]))
        den = np.linalg.norm(emp, axis=1) * ref_nrm[i]
        s1 += np.sum(emp * ref[i], axis=1) / np.where(den == 0, 1.0, den)
    return s1 / seen, s2 / total


def bands_from(s1, s2, alpha=ALPHA):
    q = alpha / 4.0
    return {"s1": (float(np.quantile(s1, q)), float(np.quantile(s1, 1 - q))),
            "s2": (float(np.quantile(s2, q)), float(np.quantile(s2, 1 - q)))}


def outside(s1, s2, b):
    return ((s1 < b["s1"][0]) | (s1 > b["s1"][1])
            | (s2 < b["s2"][0]) | (s2 > b["s2"][1]))


def main():
    base_f = os.path.join(CACHE, "gpt2_P_base.npy")
    ref_f = os.path.join(CACHE, "gpt2_ref.msl.npz")
    if not (os.path.exists(base_f) and os.path.exists(ref_f)):
        print("  run e2e_real_models.py first (it writes the cache)")
        return 1
    P = np.load(base_f).astype(np.float32)
    ref = Snapshot.load(ref_f)
    print("=" * 100)
    print(f"SAMPLING-MODE POWER on real GPT-2 distributions "
          f"({P.shape[0]} positions, vocab {P.shape[1]})")
    print("=" * 100)

    # The fast path must compute exactly the product statistic. Both paths consume
    # the generator in the same order (position by position, per[i] uniforms each),
    # so the same seed gives the same token stream and the values must agree to
    # floating-point precision -- an equality, not a resemblance.
    pos, tok = sample_stream(P, 5000, np.random.default_rng(99))
    s1p, s2p = statistics(pos, tok, ref)
    sf1, sf2 = stats_all_reps(P, ref, 5000, 1, seed=99)
    print(f"  product path vs sweep path, same stream: "
          f"S1 {s1p:.9f} vs {sf1[0]:.9f}   S2 {s2p:.9f} vs {sf2[0]:.9f}")
    if not (abs(s1p - sf1[0]) < 1e-9 and abs(s2p - sf2[0]) < 1e-12):
        print("  EQUIVALENCE FAILED: the sweep would not measure the product")
        return 1

    variants = {"top-p 0.95": serve_top_p(P), "temperature 1.05": serve_temperature(P)}
    print(f"\n  bands: {REPS_CAL} calibration reps; FPR on {REPS_NULL} held-out null "
          f"reps; power on {REPS_ALT} reps per change; family alpha {ALPHA}")
    print(f"\n{'sampled tokens':>15}{'per position':>14}{'FPR':>8}"
          + "".join(f"{('power: ' + k):>22}" for k in variants))
    for total in BUDGETS:
        t0 = time.perf_counter()
        n1, n2 = stats_all_reps(P, ref, total, REPS_CAL, seed=1000 + total)
        b = bands_from(n1, n2)
        h1, h2 = stats_all_reps(P, ref, total, REPS_NULL, seed=2000 + total)
        fpr = float(np.mean(outside(h1, h2, b)))
        row = f"{total:>15,}{total / P.shape[0]:>14.1f}{fpr:>8.2f}"
        for j, (name, Q) in enumerate(variants.items()):
            a1, a2 = stats_all_reps(Q, ref, total, REPS_ALT, seed=3000 + total + j)
            row += f"{float(np.mean(outside(a1, a2, b))):>22.2f}"
        print(row + f"    ({time.perf_counter() - t0:.0f}s)")

    print("\n  Reading: one sampled token = one max_tokens=1 completion on the probe")
    print("  context. The spec for API mode is the smallest budget whose power column")
    print("  reaches 0.95 at FPR <= alpha; below it, honesty requires saying the")
    print("  change may go unseen. Weights mode needs none of this: it is exact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
