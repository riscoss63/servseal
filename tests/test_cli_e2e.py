"""The CLI through subprocess: real commands, real exit codes, a real model.

Slower than the unit tests and needs torch + the cached gpt2/distilgpt2 weights, so it
skips itself cleanly where they are absent. What it pins is the contract a pipeline
relies on: exit 0 for an unchanged model, exit 3 for a substituted one, exit 2 for
snapshots that measured different things, real files in, real files out.
"""
import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

POS = "400"


def run(args, **kw):
    return subprocess.run([sys.executable, "-m", "modelseal.cli", *args],
                          capture_output=True, text=True, timeout=600, **kw)


@pytest.fixture(scope="module")
def snaps(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    a = str(d / "ref.msl.npz")
    b = str(d / "same.msl.npz")
    c = str(d / "other.msl.npz")
    for model, out in (("gpt2", a), ("gpt2", b), ("distilgpt2", c)):
        r = run(["snapshot", model, "-o", out, "--positions", POS])
        assert r.returncode == 0, r.stderr
        assert "probe" in r.stdout
    return d, a, b, c


def test_unchanged_model_exits_0(snaps):
    d, a, b, _ = snaps
    r = run(["verify", a, b, "--json", str(d / "v.json")])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SEALED" in r.stdout
    rec = json.loads(open(d / "v.json", encoding="utf-8").read())
    assert rec["metrics"]["mean_hellinger"] < 1e-6


def test_substituted_model_exits_3_and_reports(snaps):
    d, a, _, c = snaps
    rep = str(d / "attestation.html")
    r = run(["verify", a, c, "--json", str(d / "s.json"), "--report", rep])
    assert r.returncode == 3, r.stdout + r.stderr
    assert "CHANGED" in r.stdout
    html = open(rep, encoding="utf-8").read()
    assert "CHANGED" in html and "svg" in html
    assert os.path.getsize(rep) > 5000


def test_different_probe_config_exits_2(snaps, tmp_path):
    d, a, _, _ = snaps
    other = str(tmp_path / "d512.msl.npz")
    r = run(["snapshot", "gpt2", "-o", other, "--positions", POS, "--D", "512"])
    assert r.returncode == 0, r.stderr
    r = run(["verify", a, other])
    assert r.returncode == 2, r.stdout + r.stderr
    assert "INCOMPARABLE" in r.stdout


def test_probes_command():
    r = run(["probes"])
    assert r.returncode == 0
    assert "default-v1" in r.stdout and "probe id" in r.stdout


def test_missing_file_exits_1():
    r = run(["verify", "no_such.msl.npz", "also_missing.msl.npz"])
    assert r.returncode == 1
