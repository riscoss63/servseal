"""The verdict logic against the measured behaviour of real perturbations.

The metric values in these tests are the ones measured on GPT-2 in the committed
calibration run (experiments/outputs/e2e_output.txt): each test pins the verdict the
product must give for a perturbation whose ground truth is known. If a threshold is
recalibrated, these tests break until the change is deliberate and documented.
"""
import pytest

from servseal.verdict import (EXIT_CHANGED, EXIT_INCOMPARABLE, EXIT_SEALED,
                               classify)


def m(**kw):
    base = dict(mean_hellinger=0.0, median_hellinger=0.0, max_hellinger=0.0,
                top1_agreement=1.0, positions_moved=0, n_positions=1500,
                noise_floor=0.088, aggregate_hellinger=0.0,
                aggregate_kl_lower_bound=0.0, structure_mismatch=False,
                templates_differ=False)
    base.update(kw)
    return base


def test_identical_model_is_sealed():
    v = classify(m(mean_hellinger=0.0000, top1_agreement=1.0))
    assert v.status == "sealed" and v.exit_code == EXIT_SEALED


def test_bfloat16_is_minor_precision():
    # measured on GPT-2: mean_h 0.036, top-1 agreement 0.951
    v = classify(m(mean_hellinger=0.036, top1_agreement=0.951))
    assert (v.status, v.severity) == ("changed", "minor")
    assert "precision" in v.signature


def test_temperature_nudge_is_minor():
    # measured: mean_h 0.044, top-1 agreement 1.000
    v = classify(m(mean_hellinger=0.044, top1_agreement=1.0))
    assert (v.status, v.severity) == ("changed", "minor")


def test_top_p_filter_is_major_tail_only():
    # measured: mean_h 0.149, top-1 agreement 1.000 -- the change nothing else sees
    v = classify(m(mean_hellinger=0.149, top1_agreement=1.0, positions_moved=1400))
    assert (v.status, v.severity) == ("changed", "major")
    assert "serving-layer" in v.signature


def test_int8_quantisation_is_major_weight_level():
    # measured: mean_h 0.184, top-1 agreement 0.756
    v = classify(m(mean_hellinger=0.184, top1_agreement=0.756))
    assert (v.status, v.severity) == ("changed", "major")
    assert "weight-level" in v.signature


def test_model_substitution():
    v = classify(m(mean_hellinger=0.55, top1_agreement=0.42))
    assert v.status == "changed" and "substitution" in v.signature
    assert v.exit_code == EXIT_CHANGED


def test_structure_mismatch_is_template_class():
    v = classify(m(mean_hellinger=0.3, top1_agreement=None, structure_mismatch=True,
                   n_positions=(1500, 1480)))
    assert "template" in v.signature or "tokenisation" in v.signature


def test_incomparable_refuses():
    v = classify({"incomparable": "probe ('a' vs 'b')"})
    assert v.status == "incomparable" and v.exit_code == EXIT_INCOMPARABLE


def test_no_false_positive_headroom():
    # weights-mode snapshots are deterministic: anything measurably above zero must
    # NOT be attested, so the sealed band must stay narrow
    v = classify(m(mean_hellinger=0.02, top1_agreement=1.0))
    assert v.status == "changed"


@pytest.mark.parametrize("mh,severity", [(0.03, "minor"), (0.08, "moderate"),
                                         (0.2, "major")])
def test_severity_bands(mh, severity):
    assert classify(m(mean_hellinger=mh, top1_agreement=0.9)).severity == severity
