"""The attestation report: one page a reviewer can read without the tool.

Everything a decision needs on one screen -- verdict, severity, signature, the two
numbers that justify them, and the certified bound -- followed by everything an audit
needs below the fold: per-position distribution of movement, snapshot identities, probe
hash, versions. The report is a static, self-contained HTML file with no external
scripts, so it can be attached to a ticket or archived with a release.
"""
from __future__ import annotations

import html
import json

__all__ = ["render_report", "render_body"]

_CSS = """
:root {
  --paper:#fbfaf7; --card:#ffffff; --ink:#22262d; --muted:#5c6472;
  --line:#e5e2da; --accent:#3d566f; --mono-bg:#f2f0ea;
  --sealed:#1b6f5f; --changed:#b03230; --incomparable:#9a6b1a;
}
:root:not([data-theme="light"]) { }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper:#191b1f; --card:#212429; --ink:#e8e6e1; --muted:#9aa1ac;
    --line:#33373e; --accent:#8fb0cf; --mono-bg:#26292f;
    --sealed:#4dbfa4; --changed:#e0706c; --incomparable:#d8a94e;
  }
}
:root[data-theme="dark"] {
  --paper:#191b1f; --card:#212429; --ink:#e8e6e1; --muted:#9aa1ac;
  --line:#33373e; --accent:#8fb0cf; --mono-bg:#26292f;
  --sealed:#4dbfa4; --changed:#e0706c; --incomparable:#d8a94e;
}
body { background:var(--paper); color:var(--ink);
  font:16px/1.55 "IBM Plex Sans", "Segoe UI", system-ui, sans-serif; margin:0; }
.wrap { max-width:860px; margin:0 auto; padding:2.2rem 1.4rem 3rem; }
.tool { font-size:.78rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--muted); }
h1 { font-size:1.65rem; margin:.25rem 0 1.2rem; text-wrap:balance; font-weight:600; }
.banner { border-radius:8px; padding:1rem 1.25rem; color:#fff; margin:0 0 1.5rem;
  display:flex; align-items:baseline; gap:.8rem; flex-wrap:wrap; }
.banner .status { font-size:1.25rem; font-weight:700; letter-spacing:.02em; }
.banner .sub { opacity:.92; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
  gap:10px; margin:0 0 1.5rem; }
.cell { background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:.7rem .9rem; }
.cell .k { font-size:.74rem; text-transform:uppercase; letter-spacing:.1em;
  color:var(--muted); }
.cell .v { font-size:1.3rem; font-variant-numeric:tabular-nums; margin-top:.15rem; }
.cell .n { font-size:.8rem; color:var(--muted); }
h2 { font-size:1.02rem; margin:1.8rem 0 .6rem; }
p { margin:.5rem 0; max-width:68ch; }
.sig { border-left:3px solid var(--accent); padding:.2rem 0 .2rem 1rem; }
table { border-collapse:collapse; width:100%; font-size:.9rem; }
td, th { text-align:left; padding:.35rem .6rem .35rem 0; vertical-align:top;
  border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:500; }
code, .mono { font-family:"IBM Plex Mono", ui-monospace, Consolas, monospace;
  font-size:.85em; background:var(--mono-bg); border-radius:4px; padding:.08em .35em; }
.hist { background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding: .9rem; }
.hist svg { display:block; width:100%; height:auto; }
.foot { margin-top:2.2rem; padding-top:1rem; border-top:1px solid var(--line);
  color:var(--muted); font-size:.82rem; }
.scroll { overflow-x:auto; }
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior:smooth; } }
"""

_FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
          'family=IBM+Plex+Sans:wght@400;600;700&family=IBM+Plex+Mono&display=swap">')

_COLOR = {"sealed": "var(--sealed)", "changed": "var(--changed)",
          "incomparable": "var(--incomparable)"}
_TITLE = {"sealed": "SEALED — behaviour matches the reference",
          "changed": "CHANGED — the served behaviour is not the reference",
          "incomparable": "INCOMPARABLE — not the same measurement"}


def _esc(x):
    return html.escape(str(x))


def _histogram_svg(hist, floor):
    edges, counts = hist["edges"], hist["counts"]
    top = max(max(counts), 1)
    W, H, pad = 720, 150, 24
    bw = (W - 2 * pad) / len(counts)
    bars = []
    for i, c in enumerate(counts):
        h = 0 if c == 0 else max(2.0, (H - 2 * pad) * c / top)
        x, y = pad + i * bw, H - pad - h
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - 1.5:.1f}" '
                    f'height="{h:.1f}" fill="var(--accent)" opacity="0.85">'
                    f'<title>{edges[i]:.2f}-{edges[i + 1]:.2f}: {c} positions</title></rect>')
    fx = pad + (W - 2 * pad) * min(floor, 1.0)
    bars.append(f'<line x1="{fx:.1f}" y1="{pad / 2}" x2="{fx:.1f}" y2="{H - pad}" '
                f'stroke="var(--changed)" stroke-dasharray="4 3"/>'
                f'<text x="{fx + 5:.1f}" y="{pad}" fill="var(--muted)" '
                f'font-size="11">noise floor {floor:.3f}</text>')
    axis = (f'<line x1="{pad}" y1="{H - pad}" x2="{W - pad}" y2="{H - pad}" '
            f'stroke="var(--line)"/>'
            f'<text x="{pad}" y="{H - 6}" fill="var(--muted)" font-size="11">0.0</text>'
            f'<text x="{W - pad - 18}" y="{H - 6}" fill="var(--muted)" '
            f'font-size="11">1.0</text>')
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Distribution of '
            f'per-position Hellinger distances">{"".join(bars)}{axis}</svg>')


def render_body(result: dict) -> str:
    """The report content, headless (no doctype/html wrapper): metrics + verdict from a
    `modelseal verify --json` result."""
    m, v = result["metrics"], result["verdict"]
    ref = m.get("ref_meta", {})
    cand = m.get("cand_meta", {})
    color, title = _COLOR[v["status"]], _TITLE[v["status"]]

    cells = []

    def cell(k, val, note=""):
        cells.append(f'<div class="cell"><div class="k">{_esc(k)}</div>'
                     f'<div class="v">{_esc(val)}</div>'
                     + (f'<div class="n">{_esc(note)}</div>' if note else "")
                     + "</div>")

    if not m.get("incomparable"):
        npos = m["n_positions"]
        npos = f"{npos[0]} vs {npos[1]}" if isinstance(npos, (list, tuple)) else npos
        cell("mean Hellinger", f"{m['mean_hellinger']:.4f}",
             f"over {npos} probe positions")
        if m.get("top1_agreement") is not None:
            cell("top-1 agreement", f"{m['top1_agreement']:.1%}",
                 "most likely token unchanged")
        if m.get("positions_moved") is not None:
            cell("positions moved", f"{m['positions_moved']}",
                 f"above the noise floor {m['noise_floor']:.3f}")
        klb = m.get('mean_kl_lower_bound',
                    max(0.0, m['aggregate_kl_lower_bound']))
        cell("certified KL", f">= {klb:.3f}",
             "mean per-position lower bound; 0 certifies nothing")
        pa, pb = ref.get("perplexity"), cand.get("perplexity")
        if pa and pb:
            cell("perplexity", f"{pa:.2f} -> {pb:.2f}",
                 "what a log-loss check would see")

    hist_html = ""
    if m.get("hellinger_histogram"):
        hist_html = ('<h2>How far each probe position moved</h2><div class="hist">'
                     + _histogram_svg(m["hellinger_histogram"], m["noise_floor"])
                     + "</div>")

    rows = "".join(
        f"<tr><th>{_esc(k)}</th><td>{_esc(ref.get(k, chr(0x2014)))}</td>"
        f"<td>{_esc(cand.get(k, chr(0x2014)))}</td></tr>"
        for k in ("model", "dtype", "template", "n_positions", "created_utc",
                  "modelseal"))
    probe = ref.get("probe", "?")

    return f"""<style>{_CSS}</style>{_FONTS}
<div class="wrap">
  <div class="tool">modelseal &middot; behavioural attestation</div>
  <h1>{_esc(ref.get('model', '?'))} vs {_esc(cand.get('model', '?'))}</h1>
  <div class="banner" style="background:{color}">
    <span class="status">{_esc(title.split(' — ')[0])}</span>
    <span class="sub">{_esc(title.split(' — ')[1])}
      {' &middot; severity: ' + _esc(v['severity']) if v['status'] == 'changed' else ''}</span>
  </div>
  <div class="grid">{''.join(cells)}</div>
  <h2>Reading</h2>
  <div class="sig"><p><strong>{_esc(v['signature'])}.</strong> {_esc(v['explanation'])}</p></div>
  {hist_html}
  <h2>Snapshot identities</h2>
  <div class="scroll"><table>
    <tr><th></th><th>reference</th><th>candidate</th></tr>{rows}
    <tr><th>probe set</th><td colspan="2"><code>{_esc(probe)}</code>
        ({_esc(ref.get('probe_file', '?'))}) &mdash; snapshots refuse comparison unless
        this hash matches</td></tr>
  </table></div>
  <div class="foot">
    <p>Method: each snapshot stores a {_esc(ref.get('D', '?'))}-coordinate square-root
    sketch of the full next-token distribution at every probe position
    (&asymp;1&nbsp;KB/position), plus the most likely token. The sketch estimates the
    Bhattacharyya coefficient of the complete distributions &mdash; tail included,
    which is where serving filters and quantisation act and where top-k logprobs are
    blind by construction. The KL line is a one-sided certificate: it proves the
    behaviour moved at least that much, never that it did not.</p>
    <p>Exit code {v['exit_code']} &middot; generated by modelseal
    {_esc(ref.get('modelseal', ''))} &middot; verdict thresholds are calibrated on
    measured perturbations of real models; see the project's committed calibration run.</p>
  </div>
</div>"""


def render_report(result: dict, title: str | None = None) -> str:
    """Standalone HTML document for `modelseal report -o report.html`."""
    body = render_body(result)
    t = title or "modelseal attestation"
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{html.escape(t)}</title></head><body>{body}</body></html>')


def load_result(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
