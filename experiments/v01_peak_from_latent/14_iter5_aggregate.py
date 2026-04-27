"""Aggregate iter5 A/B/C/D results into a single comparison report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "outputs" / "v01_peak_from_latent"


def load(path):
    return json.load(open(path)) if path.exists() else None


def fmt(v, prec=2):
    return f"{v:.{prec}f}" if isinstance(v, (int, float)) else str(v)


def main():
    A = load(OUT / "iter5_A" / "iter5A_results.json")
    B = load(OUT / "iter5_B" / "iter5B_results.json")
    C = load(OUT / "iter5_C" / "iter5C_results.json")
    D = load(OUT / "iter5_D" / "iter5D_results.json")

    lines = ["# iter5 Aggregate — 4 directions for hour ceiling breakthrough", ""]

    # baseline anchor
    base_pape, base_hr1, base_hr2 = 55.17, 27.0, 38.5
    if D and "baseline" in D:
        base_pape = D["baseline"]["pape"]; base_hr1 = D["baseline"]["hr1"]; base_hr2 = D["baseline"]["hr2"]
    lines.append(f"NBEATSx baseline (no KV-VQ): PAPE={base_pape:.2f}  HR@1={base_hr1:.1f}  HR@2={base_hr2:.1f}")
    lines.append("")
    lines.append("Best from iter4: W5 (σ=1.5, α_v0=2.0, α_w1=0.5) → PAPE 37.45  HR@1 26.3  HR@2 38.1")
    lines.append("")

    lines.append("## Direction A — hr_weight × λ_peak retrain (T2 with stronger hour supervision)")
    lines.append("")
    if A:
        lines.append("| tag | aux_w1h% | base_PAPE | base_HR@1 | W5_PAPE | W5_HR@1 | W5_HR@2 |")
        lines.append("|---|---|---|---|---|---|---|")
        for tag, r in A.items():
            lines.append(f"| {tag} | {r['aux_within1']*100:.1f} | {fmt(r['base_pape'])} | "
                         f"{fmt(r['base_hr@1'],1)} | {fmt(r['corr_pape'])} | "
                         f"{fmt(r['corr_hr@1'],1)} | {fmt(r['corr_hr@2'],1)} |")
    else:
        lines.append("(not yet complete)")
    lines.append("")

    lines.append("## Direction B — Calendar features (hour/dow)")
    lines.append("")
    if B:
        lines.append("| tag | use_dow | hr_w | aux_w1h% | base_PAPE | base_HR@1 | W5_PAPE | W5_HR@1 | W5_HR@2 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for tag, r in B.items():
            lines.append(f"| {tag} | {r['use_dow']} | {r['hr_weight']} | "
                         f"{r['aux_within1']*100:.1f} | {fmt(r['base_pape'])} | "
                         f"{fmt(r['base_hr@1'],1)} | {fmt(r['corr_pape'])} | "
                         f"{fmt(r['corr_hr@1'],1)} | {fmt(r['corr_hr@2'],1)} |")
    else:
        lines.append("(not yet complete)")
    lines.append("")

    lines.append("## Direction C — NHITS backbone")
    lines.append("")
    if C:
        lines.append("| tag | latent | aux_w1h% | base_PAPE | base_HR@1 | W5_PAPE | W5_HR@1 | W5_HR@2 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for tag, r in C.items():
            lines.append(f"| {tag} | {r['latent_source']} | "
                         f"{r['aux_within1']*100:.1f} | {fmt(r['base_pape'])} | "
                         f"{fmt(r['base_hr@1'],1)} | {fmt(r['corr_pape'])} | "
                         f"{fmt(r['corr_hr@1'],1)} | {fmt(r['corr_hr@2'],1)} |")
    else:
        lines.append("(not yet complete)")
    lines.append("")

    lines.append("## Direction D — W5 grid (σ × α_v0 × α_w1)")
    lines.append("")
    if D:
        lines.append("Top-10 by PAPE (corr_hr@1 ≥ 26.5 filter):")
        lines.append("")
        lines.append("| σ | α_v0 | α_w1 | M | PAPE | HR@1 | HR@2 |")
        lines.append("|---|---|---|---|---|---|---|")
        rows = [r for r in D["rows"] if r["corr_hr@1"] >= 26.5]
        rows.sort(key=lambda r: r["corr_pape"])
        for r in rows[:10]:
            lines.append(f"| {r.get('sigma','?')} | {r.get('alpha_v0','?')} | "
                         f"{r.get('alpha_w1','?')} | {r.get('M', 32)} | "
                         f"{fmt(r['corr_pape'])} | {fmt(r['corr_hr@1'],1)} | {fmt(r['corr_hr@2'],1)} |")
        lines.append("")
        lines.append("Best HR@1 (any PAPE):")
        lines.append("")
        lines.append("| σ | α_v0 | α_w1 | PAPE | HR@1 | HR@2 |")
        lines.append("|---|---|---|---|---|---|")
        rows2 = sorted(D["rows"], key=lambda r: -r["corr_hr@1"])[:5]
        for r in rows2:
            lines.append(f"| {r.get('sigma','?')} | {r.get('alpha_v0','?')} | "
                         f"{r.get('alpha_w1','?')} | {fmt(r['corr_pape'])} | "
                         f"{fmt(r['corr_hr@1'],1)} | {fmt(r['corr_hr@2'],1)} |")
    else:
        lines.append("(not complete)")
    lines.append("")

    # cross-direction Pareto: best variant from each
    lines.append("## Cross-direction best (Pareto: PAPE × HR@1)")
    lines.append("")
    cross = []
    if A:
        best_a = min(A.items(), key=lambda kv: kv[1]["corr_pape"])
        cross.append(("A: " + best_a[0], best_a[1]["corr_pape"], best_a[1]["corr_hr@1"], best_a[1]["corr_hr@2"]))
    if B:
        best_b = min(B.items(), key=lambda kv: kv[1]["corr_pape"])
        cross.append(("B: " + best_b[0], best_b[1]["corr_pape"], best_b[1]["corr_hr@1"], best_b[1]["corr_hr@2"]))
    if C:
        best_c = min(C.items(), key=lambda kv: kv[1]["corr_pape"])
        cross.append(("C: " + best_c[0], best_c[1]["corr_pape"], best_c[1]["corr_hr@1"], best_c[1]["corr_hr@2"]))
    if D:
        best_d = min(D["rows"], key=lambda r: r["corr_pape"])
        cross.append(("D: " + str({k: best_d[k] for k in ("sigma","alpha_v0","alpha_w1") if k in best_d}),
                      best_d["corr_pape"], best_d["corr_hr@1"], best_d["corr_hr@2"]))
    cross.append(("baseline NBEATSx", base_pape, base_hr1, base_hr2))
    lines.append("| variant | PAPE | HR@1 | HR@2 |")
    lines.append("|---|---|---|---|")
    for n, p, h1, h2 in sorted(cross, key=lambda r: r[1]):
        lines.append(f"| {n} | {fmt(p)} | {fmt(h1,1)} | {fmt(h2,1)} |")

    text = "\n".join(lines)
    out_path = OUT / "iter5_aggregate.md"
    out_path.write_text(text, encoding="utf-8")
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
