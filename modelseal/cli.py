"""The command line: snapshot, verify, report, probes.

Exit codes are the contract, so a pipeline can gate on them:

    0  sealed        the served model behaves as the reference
    1  tool error    bad arguments, missing file, model failed to load
    2  incomparable  the snapshots measured different things; no attestation made
    3  changed       the served model does NOT behave as the reference

Output is ASCII, one fact per line, machine-greppable; --json gives the whole record.
"""
from __future__ import annotations

import argparse
import json
import sys

from .probes import load_probes, probe_id
from .snapshot import Snapshot
from .verdict import EXIT_ERROR, classify

__all__ = ["main"]


def _cmd_snapshot(a):
    from .runner import snapshot_model                     # torch only here
    snap = snapshot_model(a.model, probes=a.probes, D=a.D, seed=a.seed,
                          max_positions=a.positions, max_length=a.max_length,
                          template=a.template, dtype=a.dtype, label=a.label)
    snap.save(a.out)
    m = snap.meta
    print(f"snapshot   {a.out}")
    print(f"model      {m['model']}  dtype={m['dtype']}")
    print(f"positions  {m['n_positions']}  D={m['D']}  size={snap.nbytes() / 1024:.0f} KB")
    print(f"probe      {m['probe']}  ({m.get('probe_file')})")
    print(f"perplexity {m.get('perplexity')}")
    return 0


def _cmd_verify(a):
    ref = Snapshot.load(a.reference)
    cand = Snapshot.load(a.candidate)
    metrics = ref.compare(cand)
    verdict = classify(metrics)
    record = {"metrics": metrics, "verdict": verdict.as_dict()}

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
    print(f"verdict    {verdict.status.upper()}"
          + (f"  severity={verdict.severity}" if verdict.status == "changed" else ""))
    print(f"signature  {verdict.signature}")
    if not metrics.get("incomparable"):
        print(f"hellinger  mean={metrics['mean_hellinger']:.4f}"
              f"  max={metrics['max_hellinger']:.4f}"
              f"  floor={metrics['noise_floor']:.4f}")
        if metrics.get("top1_agreement") is not None:
            print(f"top1       agreement={metrics['top1_agreement']:.4f}")
        klb = metrics.get('mean_kl_lower_bound',
                          max(0.0, metrics['aggregate_kl_lower_bound']))
        print(f"kl_bound   mean per-position KL >= {klb:.4f}"
              f"  (certified; 0 certifies nothing, not absence of change)")
    else:
        print(f"reason     {metrics['incomparable']}")
    if a.report:
        from .report import render_report
        with open(a.report, "w", encoding="utf-8") as fh:
            fh.write(render_report(record))
        print(f"report     {a.report}")
    return verdict.exit_code


def _cmd_report(a):
    from .report import load_result, render_report
    record = load_result(a.result)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(render_report(record))
    print(f"report     {a.out}")
    return 0


def _cmd_probes(a):
    texts = load_probes(a.set)
    print(f"probe set  {a.set or 'default-v1'}")
    print(f"texts      {len(texts)}")
    print(f"probe id   {probe_id(texts)}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="modelseal",
        description="Behavioural attestation for deployed language models: "
                    "is the model you serve the model you validated?")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("snapshot", help="fingerprint a model over the probe set")
    s.add_argument("model", help="HF model id or local path")
    s.add_argument("-o", "--out", required=True, help="output .msl.npz file")
    s.add_argument("--probes", default=None, help="bundled set name or probe file")
    s.add_argument("--positions", type=int, default=1500)
    s.add_argument("--max-length", type=int, default=96)
    s.add_argument("--D", type=int, default=256)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16", "float16"])
    s.add_argument("--template", default=None,
                   help="prompt template with {text}, if the deployment applies one")
    s.add_argument("--label", default=None)
    s.set_defaults(fn=_cmd_snapshot)

    v = sub.add_parser("verify", help="compare a candidate snapshot to a reference")
    v.add_argument("reference")
    v.add_argument("candidate")
    v.add_argument("--json", default=None, help="write the full record here")
    v.add_argument("--report", default=None, help="write an HTML attestation here")
    v.set_defaults(fn=_cmd_verify)

    r = sub.add_parser("report", help="render an HTML attestation from a verify --json")
    r.add_argument("result")
    r.add_argument("-o", "--out", required=True)
    r.set_defaults(fn=_cmd_report)

    pr = sub.add_parser("probes", help="show a probe set and its id")
    pr.add_argument("--set", default=None)
    pr.set_defaults(fn=_cmd_probes)

    a = p.parse_args(argv)
    try:
        return a.fn(a)
    except FileNotFoundError as e:
        print(f"error      {e}", file=sys.stderr)
        return EXIT_ERROR
    except ValueError as e:
        print(f"error      {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
