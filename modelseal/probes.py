"""Probe sets: the fixed texts a model is measured on.

A fingerprint is only meaningful relative to what the model was asked. Two snapshots are
comparable only if their probes are byte-identical, so every snapshot carries a hash of
the exact texts, and verification refuses to compare across different probe sets rather
than producing a number that quietly means nothing.

The bundled set is versioned and frozen: `default-v1` never changes. A user with a
domain of their own (legal drafting, code review, customer support) should snapshot on
their own probe file too -- drift is easiest to see on the distribution of the traffic
you actually serve.
"""
from __future__ import annotations

import hashlib
import os

__all__ = ["load_probes", "probe_id", "DEFAULT_SET"]

DEFAULT_SET = "default-v1"
_HERE = os.path.dirname(os.path.abspath(__file__))


def load_probes(spec: str | None = None) -> list:
    """Texts of a probe set.

    `spec` is either the name of a bundled set (``default-v1``) or a path to a text file.
    Format: paragraphs separated by blank lines; lines starting with ``#`` are comments.
    """
    if not spec:
        spec = DEFAULT_SET
    path = spec
    if not os.path.exists(path):
        bundled = os.path.join(_HERE, "probes", f"{spec}.txt")
        if not os.path.exists(bundled):
            raise FileNotFoundError(
                f"no probe file {spec!r} and no bundled set of that name")
        path = bundled
    blocks, cur = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.lstrip().startswith("#"):
                continue
            if line.strip():
                cur.append(line.rstrip("\n"))
            elif cur:
                blocks.append("\n".join(cur))
                cur = []
    if cur:
        blocks.append("\n".join(cur))
    if not blocks:
        raise ValueError(f"probe file {path!r} contains no texts")
    return blocks


def probe_id(texts) -> str:
    """Hash of the exact probe texts.

    Same construction as sqsketch.llm uses internally (blake2b-128 over the utf-8 texts,
    NUL-separated), kept identical so fingerprints made by either tool agree on identity.
    """
    h = hashlib.blake2b(digest_size=16)
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
        h.update(b"\0")
    return h.hexdigest()
