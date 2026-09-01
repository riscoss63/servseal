"""Snapshot mechanics on synthetic distributions: no torch, runs in seconds.

The distributions are small (vocab 400) but real probability vectors, pushed through
the same code path a model snapshot uses (Snapshot.from_distributions), so what is
tested is the product, not a stand-in.
"""
import numpy as np
import pytest

from servseal.snapshot import Snapshot
from servseal.verdict import classify


def dists(n_pos=200, vocab=400, seed=0, sharp=1.0):
    r = np.random.default_rng(seed)
    P = r.dirichlet(np.full(vocab, 0.05) * sharp, size=n_pos)
    return P.astype(np.float32)


def snap(P, model="ref", **kw):
    return Snapshot.from_distributions(P, probe="p1", model=model, **kw)


def test_roundtrip(tmp_path):
    s = snap(dists())
    f = tmp_path / "x.seal.npz"
    s.save(f)
    t = Snapshot.load(f)
    assert np.array_equal(s.positions, t.positions)
    assert np.array_equal(s.argmax, t.argmax)
    assert s.meta == t.meta


def test_identical_distributions_attest_at_zero():
    P = dists()
    r = snap(P).compare(snap(P, model="cand"))
    assert r["mean_hellinger"] < 1e-6
    assert r["top1_agreement"] == 1.0
    assert classify(r).status == "sealed"


def test_tail_only_change_measured_but_argmax_intact():
    P = dists()
    Q = P.copy()
    # cut the tail: zero everything below the per-row 90th percentile, renormalise --
    # a caricature of top-p that provably leaves the argmax unchanged
    thr = np.quantile(Q, 0.90, axis=1, keepdims=True)
    Q = np.where(Q >= thr, Q, 0.0)
    Q /= Q.sum(1, keepdims=True)
    r = snap(P).compare(snap(Q, model="cand"))
    assert r["top1_agreement"] == 1.0
    assert r["mean_hellinger"] > 0.05


def test_sketch_tracks_true_hellinger():
    # the sketch's mean Hellinger must approximate the exact one computed from the
    # full distributions, within the noise floor at D=256
    P, = [dists(seed=1)]
    r2 = np.random.default_rng(9)
    Q = P * np.exp(0.5 * r2.standard_normal(P.shape).astype(np.float32))
    Q /= Q.sum(1, keepdims=True)
    exact = np.sqrt(np.clip(1 - np.sum(np.sqrt(P.astype(np.float64)
                                               * Q.astype(np.float64)), 1), 0, None))
    r = snap(P).compare(snap(Q, model="cand"))
    assert abs(r["mean_hellinger"] - exact.mean()) < np.sqrt(2.0 / 256)


def test_different_probes_are_incomparable():
    P = dists()
    a = Snapshot.from_distributions(P, probe="p1", model="a")
    b = Snapshot.from_distributions(P, probe="p2", model="b")
    r = a.compare(b)
    assert "probe" in r["incomparable"]
    assert classify(r).exit_code == 2


def test_different_D_is_incomparable():
    P = dists()
    r = snap(P, D=256).compare(snap(P, D=512))
    assert "D" in r["incomparable"]


def test_position_count_mismatch_is_structure_not_refusal():
    P = dists()
    r = snap(P).compare(snap(P[:150], model="cand"))
    assert r["structure_mismatch"] is True
    v = classify(r)
    assert v.status in ("changed", "sealed")     # compared, not refused
    assert v.exit_code != 2


def test_histogram_shape():
    P = dists()
    r = snap(P).compare(snap(P, model="cand"))
    h = r["hellinger_histogram"]
    assert len(h["counts"]) == 24 and len(h["edges"]) == 25
    assert sum(h["counts"]) == r["n_positions"]
