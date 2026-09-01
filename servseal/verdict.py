"""Turning measurements into a verdict someone can act on.

The measurements are distances; the product is a decision. The thresholds here were not
chosen by taste: they are calibrated on real perturbations of real models, applied where
a deployment would apply them (experiments/e2e_real_models.py), and the calibration run
is committed next to this file. The bands they draw:

    mean Hellinger over the probe positions
      < UNCHANGED .......... behaviour identical to the reference (weights-mode
                             snapshots are deterministic, so identical means ~0)
      < MINOR .............. precision-level: bfloat16 rounding measures 0.036, a
                             temperature nudge to 1.05 measures 0.044
      < MODERATE ........... something structural moved
      otherwise ............ major: int8 weight quantisation measures 0.18, a top-p
                             serving filter 0.15

The *signature* uses a second, independent coordinate: how often the most likely token
changed. A serving-layer filter (top-p, min-p) reshapes the tail while leaving the
argmax untouched -- measured agreement 1.000 with mean Hellinger 0.149 -- which is
exactly the change that perplexity, output diffing and top-k logprobs cannot see.
Weight-level changes move the argmax too (int8: agreement 0.756). A substitution or a
prompt-template mismatch destroys it.

Every boundary is a named constant, and the e2e battery asserts the verdict of each
known perturbation, so a recalibration is a visible edit that breaks tests until the
documentation above is updated with it.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Verdict", "classify", "THRESHOLDS"]

THRESHOLDS = {
    "unchanged_mean_h": 0.010,   # below this, attest unchanged
    "minor_mean_h": 0.060,       # bf16 at 0.036 and temp 1.05 at 0.044 land here
    "moderate_mean_h": 0.120,    # top-p 0.95 (0.149) and int8 (0.18) land above
    "tail_only_top1": 0.990,     # argmax agreement above this = serving-layer signature
    "tail_only_min_h": 0.080,    # ...provided the distribution moved this much
    "substitution_top1": 0.600,  # argmax agreement below this = different model/prompt
}

# exit codes, CI-friendly: 0 attested, 3 changed, 2 not comparable, 1 tool error
EXIT_SEALED, EXIT_ERROR, EXIT_INCOMPARABLE, EXIT_CHANGED = 0, 1, 2, 3


@dataclass
class Verdict:
    status: str        # "sealed" | "changed" | "incomparable"
    severity: str      # "none" | "minor" | "moderate" | "major"
    signature: str     # short mechanical hypothesis
    explanation: str   # one paragraph a reader can act on
    exit_code: int

    def as_dict(self):
        return {"status": self.status, "severity": self.severity,
                "signature": self.signature, "explanation": self.explanation,
                "exit_code": self.exit_code}


def classify(metrics: dict, t: dict = THRESHOLDS) -> Verdict:
    """Metrics from Snapshot.compare -> a Verdict. Pure function, no I/O."""
    if metrics.get("incomparable"):
        return Verdict(
            "incomparable", "none", "not the same measurement",
            f"The snapshots differ in {metrics['incomparable']} and cannot be compared. "
            "Re-snapshot both sides with the same probe set, sketch width and seed.",
            EXIT_INCOMPARABLE)

    mh = metrics["mean_hellinger"]
    t1 = metrics.get("top1_agreement")           # None when positions are misaligned
    structure = bool(metrics.get("structure_mismatch"))
    templates_differ = bool(metrics.get("templates_differ"))

    if not structure and mh < t["unchanged_mean_h"] and (t1 is None or t1 > 0.999):
        return Verdict(
            "sealed", "none", "behaviour unchanged",
            f"Mean Hellinger distance {mh:.4f} over {metrics['n_positions']} positions "
            "is below the attestation threshold and the most likely token agrees "
            "everywhere. The deployed model behaves as the reference.",
            EXIT_SEALED)

    if mh < t["minor_mean_h"]:
        severity = "minor"
    elif mh < t["moderate_mean_h"]:
        severity = "moderate"
    else:
        severity = "major"

    if structure:
        signature = "tokenisation or template-level change"
        explanation = (
            "The two snapshots do not even cover the same token positions on identical "
            "probe texts, which means the text reaching the model changed: a chat "
            "template, a system prompt, or a different tokeniser. This is the class of "
            "bug that is routinely misdiagnosed as a bad quantisation.")
    elif t1 is not None and t1 < t["substitution_top1"]:
        signature = ("prompt/template mismatch" if templates_differ
                     else "model substitution or template mismatch")
        explanation = (
            f"The most likely token agrees at only {t1:.1%} of positions. No precision "
            "or sampling change does this; either a different model is being served, or "
            "the prompt reaching it is not the prompt you validated.")
    elif t1 is not None and t1 >= t["tail_only_top1"] and mh >= t["tail_only_min_h"]:
        signature = "serving-layer sampling filter (tail-only)"
        explanation = (
            f"The distribution moved substantially (mean Hellinger {mh:.3f}) while the "
            f"most likely token agrees at {t1:.1%} of positions: the head of the "
            "distribution is intact and the tail is reshaped. That is the signature of "
            "a sampling filter such as top-p or min-p applied at serving time -- a "
            "change invisible to greedy output diffs, to perplexity, and largely to "
            "top-k logprobs.")
    elif mh < t["minor_mean_h"]:
        signature = "precision-level (dtype, kernels) or mild sampling parameter"
        explanation = (
            f"A small, broad shift (mean Hellinger {mh:.3f}) with the most likely token "
            f"agreeing at {t1:.1%} of positions. Consistent with reduced-precision "
            "arithmetic (bfloat16 measures 0.036 on GPT-2) or a small temperature "
            "change (1.05 measures 0.044). Decide against your own tolerance; this is "
            "below the level at which weight quantisation typically lands.")
    else:
        signature = "weight-level change (quantisation, fine-tune, or related model)"
        explanation = (
            f"The distribution moved (mean Hellinger {mh:.3f}) and the most likely "
            f"token changed at {1 - t1:.1%} of positions. The weights producing the "
            "distribution are not the reference weights: quantisation, a fine-tune, a "
            "different checkpoint revision -- or a closely related model substituted "
            "for the reference, which behavioural evidence alone cannot always "
            "separate from a heavily modified one (a distilled sibling measures here).")

    return Verdict("changed", severity, signature, explanation, EXIT_CHANGED)
