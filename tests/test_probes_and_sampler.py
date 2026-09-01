"""Probe determinism, and the sampling-mode statistic's basic sanity."""
import numpy as np

from modelseal.probes import load_probes, probe_id
from modelseal.sampler import calibrate, detect, sample_stream, statistics
from modelseal.snapshot import Snapshot


def test_default_probes_load_and_are_stable():
    texts = load_probes()
    assert len(texts) >= 30
    assert probe_id(texts) == probe_id(load_probes())
    assert all(len(t) > 100 for t in texts)


def test_comments_and_blank_lines_ignored(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("# comment\n\nfirst text block\n\n\nsecond block\n", encoding="utf-8")
    assert load_probes(str(f)) == ["first text block", "second block"]


def test_sample_stream_budget_and_support():
    r = np.random.default_rng(0)
    P = r.dirichlet(np.full(50, 0.2), size=30).astype(np.float32)
    pos, tok = sample_stream(P, 999, np.random.default_rng(1))
    assert len(pos) == len(tok) == 999
    assert pos.min() >= 0 and pos.max() < 30 and tok.max() < 50


def _tail_cut(P, keep_mass=0.90):
    """Per-row top-p style cut: keep the smallest head holding `keep_mass`."""
    srt = -np.sort(-P, 1)
    thr = srt[np.arange(len(P)), np.argmax(np.cumsum(srt, 1) >= keep_mass, 1)][:, None]
    Q = np.where(P >= thr, P, 0.0)
    return Q / Q.sum(1, keepdims=True)


def test_sampling_mode_two_sided_per_position_detection():
    # Two designs measurably failed here before this one: a one-sided "BC must drop"
    # rule (power 0.10 -- a tail cut concentrates the sample on the head and pushes
    # the statistic UP), and a pooled corpus-aggregate statistic (power 0.03 -- one
    # position's tail is another's head, pooling erases the evidence). This test pins
    # the per-position, two-sided design that replaced them.
    r = np.random.default_rng(0)
    vocab = 2000
    P = r.dirichlet(np.full(vocab, 0.05), size=100).astype(np.float32)
    Q = _tail_cut(P, keep_mass=0.90)
    ref = Snapshot.from_distributions(P, probe="p", model="ref")
    bands, (n1, n2) = calibrate(P, ref, total=4000, reps=80, seed=7)
    fp = np.mean([detect(a, b, bands) for a, b in zip(n1, n2)])
    hits = 0
    for i in range(60):
        pos, tok = sample_stream(Q, 4000, np.random.default_rng(100 + i))
        hits += detect(*statistics(pos, tok, ref), bands)
    assert hits / 60 > 0.9, f"power {hits / 60:.2f}"
    # the bands are built from these very null draws, so their nominal level holds by
    # construction; assert the sanity of that, not a re-derivation
    assert fp <= 0.10
