"""Aggregate H1a/b/c JSONs into one markdown report."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "outputs" / "v01_peak_from_latent"


def fmt(v, prec=3):
    return f"{v:.{prec}f}" if isinstance(v, (int, float)) else str(v)


def main():
    h1a = json.load(open(OUT / "probe_h1a.json"))
    h1b = json.load(open(OUT / "quantize_h1b.json")) if (OUT / "quantize_h1b.json").exists() else None
    h1c = json.load(open(OUT / "coldstart_h1c.json")) if (OUT / "coldstart_h1c.json").exists() else None

    lines = ["# v01-01 Peak from Latent — Result Report", ""]
    lines.append(f"- H1a pass arms: `{h1a['pass_arms']}`  (R² ≥ {h1a['pass_threshold']})")
    if h1b:
        lines.append(f"- H1b pass arms: `{h1b.get('pass_arms_h1b', [])}`  (ratio ≥ {h1b.get('pass_threshold_ratio', 'n/a')})")
    if h1c:
        lines.append(f"- H1c pass arms: `{h1c.get('pass_arms_h1c', [])}`  (PAPE ratio ≤ {h1c.get('pass_threshold_ratio', 'n/a')})")
    lines.append("")

    lines.append("## H1a — Probe (peak_amp_fc, across-household)")
    lines.append("")
    lines.append("| Arm | dim | Ridge R² | Ridge PAPE % | MLP R² | hr top-1 | hr top-3 | gate |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for arm, d in h1a["results"].items():
        a = d["peak_amp_fc"]; h = d["peak_hr_fc"]
        gate = "PASS" if a["ridge_R2"] >= h1a["pass_threshold"] else "FAIL"
        lines.append(f"| {arm} | {d['feat_dim']} | {fmt(a['ridge_R2'])} | {fmt(a['ridge_PAPE'],1)} | {fmt(a['mlp_R2'])} | {fmt(h['top1'])} | {fmt(h['top3'])} | {gate} |")
    lines.append("")

    if h1b and h1b.get("results"):
        lines.append("## H1b — Quantization (M=32, KMeans++)")
        lines.append("")
        lines.append("| Arm | R²(raw) | R²(q) | ratio | util | k_min | gate |")
        lines.append("|---|---|---|---|---|---|---|")
        for arm, d in h1b["results"].items():
            v = d["vq_diagnostics"]
            lines.append(f"| {arm} | {fmt(d['r2_raw'])} | {fmt(d['r2_quantized'])} | {fmt(d['ratio'])} | {fmt(v['utilization'],2)} | {v['k_min']} | {d['gate_h1b']} |")
        lines.append("")

    if h1c and h1c.get("results"):
        lines.append("## H1c — Cold-start KV-VQ (50 cold apts)")
        lines.append("")
        lines.append("| Arm | base PAPE | KV PAPE | ratio | base HR@1 | KV HR@1 | gate |")
        lines.append("|---|---|---|---|---|---|---|")
        for arm, d in h1c["results"].items():
            b = d["baseline"]; k = d["kv_vq"]
            lines.append(f"| {arm} | {fmt(b['pape'],2)} | {fmt(k['pape'],2)} | {fmt(d['pape_ratio'])} | {fmt(b['hr@1'],1)} | {fmt(k['hr@1'],1)} | {d['gate_h1c']} |")
        lines.append("")

    text = "\n".join(lines)
    out_path = OUT / "report.md"
    out_path.write_text(text, encoding="utf-8")
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
