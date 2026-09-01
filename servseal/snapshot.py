"""The snapshot: what a model does on the probe set, in about 1 KB per position.

A snapshot stores, per probe position, a D-coordinate square-root sketch of the full
next-token distribution (sqsketch's fingerprint encoding, D=256 by default), plus the
argmax token id -- two independent coordinates of behaviour. The sketch sees the whole
distribution including the tail, which is where serving filters and quantisation act
and where top-k logprob baselines are blind by construction; the argmax sees the head.
The pair is what lets a verdict distinguish *how much* moved from *which part* moved.

Everything here is numpy: verifying and reporting never load torch. Only creating a
snapshot from a live model does (runner.py).
"""
from __future__ import annotations

import datetime
import json

import numpy as np

from sqsketch.core import Sketch
from sqsketch.llm import compare as _row_bc
from sqsketch.llm import fingerprint_batch

from . import __version__ as _VERSION

__all__ = ["Snapshot"]

_COMPARABILITY_KEYS = ("probe", "D", "seed", "vocab_size")


class Snapshot:
    """Per-position sketches + argmax ids + aggregate sketch + metadata."""

    def __init__(self, positions, aggregate, argmax, meta):
        self.positions = np.asarray(positions, dtype=np.float32)
        self.aggregate = np.asarray(aggregate, dtype=np.float64)
        self.argmax = np.asarray(argmax, dtype=np.int32)
        self.meta = dict(meta)

    # ------------------------------------------------------------------ create

    @classmethod
    def from_distributions(cls, P, *, probe, model, D=256, seed=0,
                           template=None, dtype="float32", extra=None):
        """Snapshot a (positions x vocabulary) array of next-token distributions.

        This is the whole product path minus the forward pass, so the test battery and
        the CLI exercise the same code whether P came from torch or from a file.
        """
        P = np.asarray(P)
        positions = fingerprint_batch(P, D=D, seed=seed)
        aggregate = fingerprint_batch(P.mean(0, dtype=np.float64, keepdims=True),
                                      D=D, seed=seed)[0]
        meta = {
            "model": model, "D": int(D), "seed": int(seed),
            "vocab_size": int(P.shape[1]), "n_positions": int(P.shape[0]),
            "probe": probe, "template": template, "dtype": dtype,
            "created_utc": datetime.datetime.now(datetime.timezone.utc)
                                   .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "servseal": _VERSION,
        }
        meta.update(extra or {})
        return cls(positions, aggregate, np.argmax(P, axis=1), meta)

    # ---------------------------------------------------------------------- io

    def save(self, path):
        np.savez_compressed(path, positions=self.positions, aggregate=self.aggregate,
                            argmax=self.argmax, meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(z["positions"], z["aggregate"], z["argmax"],
                   json.loads(str(z["meta"])))

    def nbytes(self):
        return self.positions.nbytes + self.aggregate.nbytes + self.argmax.nbytes

    # ---------------------------------------------------------------- compare

    def compare(self, other: "Snapshot") -> dict:
        """Every measurement the verdict needs, and nothing that requires the models.

        Refuses (as `incomparable`) when the snapshots measured different things: other
        probes, another sketch width or seed, another vocabulary. A position-count
        mismatch on identical probes is different -- it means the text reaching the
        model changed (template, tokeniser) -- and is reported as `structure_mismatch`
        with aggregate-only distances rather than refused.
        """
        differing = [k for k in _COMPARABILITY_KEYS
                     if self.meta.get(k) != other.meta.get(k)]
        if differing:
            return {"incomparable": ", ".join(
                f"{k} ({self.meta.get(k)!r} vs {other.meta.get(k)!r})"
                for k in differing)}

        D, seed = self.meta["D"], self.meta["seed"]
        agg_a = Sketch(self.aggregate, D, seed, 1, 0, 0.0)
        agg_b = Sketch(other.aggregate, D, seed, 1, 0, 0.0)
        lo, hi = agg_a.confidence_interval(agg_b)
        out = {
            "aggregate_hellinger": agg_a.hellinger(agg_b),
            "aggregate_bc_interval": (float(lo), float(hi)),
            "aggregate_kl_lower_bound": agg_a.kl_lower_bound(agg_b),
            "noise_floor": float(np.sqrt(2.0 / D)),
            "templates_differ": self.meta.get("template") != other.meta.get("template"),
            "ref_meta": self.meta, "cand_meta": other.meta,
        }

        if self.positions.shape != other.positions.shape:
            out.update({
                "structure_mismatch": True,
                "mean_hellinger": out["aggregate_hellinger"],
                "median_hellinger": out["aggregate_hellinger"],
                "max_hellinger": out["aggregate_hellinger"],
                "top1_agreement": None, "positions_moved": None,
                "n_positions": (int(self.positions.shape[0]),
                                int(other.positions.shape[0])),
                "hellinger_histogram": None,
            })
            return out

        bc = np.clip(_row_bc(self.positions.astype(np.float64),
                             other.positions.astype(np.float64)), -1.0, 1.0)
        hel = np.sqrt(np.clip(1.0 - bc, 0.0, None))
        hist, edges = np.histogram(hel, bins=24, range=(0.0, 1.0))

        # Certified divergence, per position and then averaged: each position's bound
        # -2 ln(upper CI end of BC) is a valid lower bound on that position's KL, and a
        # mean of valid lower bounds is a valid lower bound on the mean KL -- sharing
        # one hash across positions correlates the errors but breaks no single bound.
        # The aggregate-profile bound is kept too, but pooling makes it weak: position
        # tails cancel in the average profile, the same effect that killed the pooled
        # sampling statistic. At D=256 the certificate only bites on gross changes;
        # a value of 0 certifies nothing and must never be read as "nothing changed".
        mean_kl = float(np.mean([
            max(0.0, Sketch(self.positions[i].astype(np.float64), D, seed, 1, 0, 0.0)
                .kl_lower_bound(Sketch(other.positions[i].astype(np.float64),
                                       D, seed, 1, 0, 0.0)))
            for i in range(self.positions.shape[0])]))
        out.update({
            "structure_mismatch": False,
            "mean_kl_lower_bound": mean_kl,
            "mean_hellinger": float(hel.mean()),
            "median_hellinger": float(np.median(hel)),
            "max_hellinger": float(hel.max()),
            "top1_agreement": float(np.mean(self.argmax == other.argmax)),
            "positions_moved": int(np.sum(hel > np.sqrt(2.0 / D))),
            "n_positions": int(hel.size),
            "hellinger_histogram": {"edges": [float(e) for e in edges],
                                    "counts": [int(c) for c in hist]},
        })
        return out

    def __repr__(self):
        m = self.meta
        return (f"Snapshot(model={m.get('model')!r}, positions={m.get('n_positions')}, "
                f"D={m.get('D')}, vocab={m.get('vocab_size')}, "
                f"{self.nbytes() / 1024:.0f} KB)")
