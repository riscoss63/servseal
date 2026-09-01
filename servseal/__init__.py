"""servseal: behavioural attestation for deployed language models.

    from servseal import Snapshot, classify
    from servseal.runner import snapshot_model         # needs torch

    ref = snapshot_model("gpt2")
    ref.save("gpt2.seal.npz")
    ...
    verdict = classify(Snapshot.load("gpt2.seal.npz").compare(snapshot_model(served)))
    print(verdict.status, "-", verdict.signature)
"""
__version__ = "0.1.2"

from .probes import load_probes, probe_id          # noqa: E402
from .snapshot import Snapshot                     # noqa: E402
from .verdict import Verdict, classify             # noqa: E402

__all__ = ["Snapshot", "Verdict", "classify", "load_probes", "probe_id",
           "__version__"]
