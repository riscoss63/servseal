"""API mode: testing an endpoint you can only sample from.

Weights mode compares full softmax distributions and is exact. Against a black-box API
none of that is available -- the endpoint returns sampled tokens, already shaped by
whatever serving parameters the provider applies. That is not a limitation of the test;
it is the object under test: the sampled tokens ARE the served behaviour.

Protocol: ask the endpoint for one next token per probe position (temperature 1,
max_tokens 1), N times per position. Two statistics are computed from the stream:

  S1  the mean, over probe positions, of the sketch-estimated Bhattacharyya
      coefficient between that position's empirical sample and that position's
      *stored reference sketch* -- the distributional coordinate.
  S2  the fraction of sampled tokens equal to the reference argmax at their
      position -- the head coordinate.

Two designs died on the way to this one, both killed by measurement rather than
argument, and both worth recording. A one-sided "BC must drop" rule had power 0.10
against a tail cut: finite sampling already visits the tail rarely, so removing it
concentrates the sample on head tokens the reference knows well and can push the
statistic *up* -- hence every test here is two-sided. And a pooled corpus-aggregate
statistic had power ~0.03 against a cut with a true per-position Hellinger distance of
0.23: one position's tail is another position's head, so pooling redistributes the
removed mass inside largely the same support and erases the evidence. The signal lives
per position -- and the snapshot already stores a sketch per position, which is exactly
the reference S1 needs.

Decision thresholds are calibrated empirically by simulating the *unchanged* endpoint
at the same budget, which requires the reference distributions and therefore the
weights -- available exactly once, when the reference snapshot is made. How many
samples buy how much power is measured on real models in experiments/sampling_power.py;
the honest spec for this mode is that run's output.
"""
from __future__ import annotations

import numpy as np

from sqsketch.hashing import mix64

__all__ = ["sample_stream", "statistics", "calibrate", "detect"]


def sample_stream(P, total, rng):
    """Simulate the protocol against known distributions: `total` sampled tokens spread
    evenly over the positions (rows) of P. Returns flat (position, token) arrays."""
    P = np.asarray(P)
    n_pos, vocab = P.shape
    per = np.full(n_pos, total // n_pos, dtype=np.int64)
    per[: total - per.sum()] += 1
    pos_out, tok_out = [], []
    for i in range(n_pos):
        if per[i] == 0:
            continue
        cum = np.cumsum(P[i], dtype=np.float64)
        cum /= cum[-1]
        ids = np.minimum(np.searchsorted(cum, rng.random(per[i]), side="right"),
                         vocab - 1)
        pos_out.append(np.full(per[i], i, dtype=np.int64))
        tok_out.append(ids.astype(np.int64))
    return np.concatenate(pos_out), np.concatenate(tok_out)


def _codes(vocab, D, seed):
    """Bucket and sign for every token id -- the same hash fingerprint_batch uses, so
    an empirical sketch lands in the same space as the stored reference sketches."""
    h = mix64(np.arange(vocab, dtype=np.uint64), seed=seed + 1)
    return (h % np.uint64(D)).astype(np.int64), \
        np.where((h >> np.uint64(63)) & np.uint64(1), 1.0, -1.0)


def statistics(pos, tok, snapshot):
    """(S1, S2) for one observed stream, against a reference snapshot."""
    meta = snapshot.meta
    D, seed, vocab = meta["D"], meta["seed"], meta["vocab_size"]
    n_pos = snapshot.positions.shape[0]
    bucket, sign = _codes(vocab, D, seed)

    n_per = np.bincount(pos, minlength=n_pos).astype(np.float64)
    keys, counts = np.unique(pos * vocab + tok, return_counts=True)
    kp, kt = keys // vocab, keys % vocab
    emp = np.zeros((n_pos, D))
    np.add.at(emp.ravel(), kp * D + bucket[kt],
              sign[kt] * np.sqrt(counts / n_per[kp]))

    ref = snapshot.positions.astype(np.float64)
    seen = n_per > 0
    num = np.sum(emp[seen] * ref[seen], axis=1)
    den = np.linalg.norm(emp[seen], axis=1) * np.linalg.norm(ref[seen], axis=1)
    s1 = float(np.mean(num / np.where(den == 0, 1.0, den)))
    s2 = float(np.mean(tok == snapshot.argmax[pos]))
    return s1, s2


def calibrate(P_ref, snapshot, total, reps=200, alpha=0.05, seed=1234):
    """Two-sided acceptance bands for (S1, S2) under the unchanged endpoint at this
    budget, Bonferroni-split so the pair has family false-positive rate <= alpha."""
    rng = np.random.default_rng(seed)
    s1s, s2s = [], []
    for _ in range(reps):
        pos, tok = sample_stream(P_ref, total, rng)
        a, b = statistics(pos, tok, snapshot)
        s1s.append(a)
        s2s.append(b)
    q = alpha / 4.0                       # two statistics x two tails
    bands = {
        "s1": (float(np.quantile(s1s, q)), float(np.quantile(s1s, 1 - q))),
        "s2": (float(np.quantile(s2s, q)), float(np.quantile(s2s, 1 - q))),
        "total": int(total), "reps": int(reps), "alpha": float(alpha),
    }
    return bands, (np.array(s1s), np.array(s2s))


def detect(s1, s2, bands):
    """True when the observed pair falls outside the unchanged endpoint's bands."""
    lo1, hi1 = bands["s1"]
    lo2, hi2 = bands["s2"]
    return bool(s1 < lo1 or s1 > hi1 or s2 < lo2 or s2 > hi2)
