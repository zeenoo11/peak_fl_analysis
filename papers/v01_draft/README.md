# Paper artifacts — `papers/`

## Files

| File | Purpose |
|------|---------|
| `v01_peak_vq.md` | Markdown master draft (human-readable, fully self-contained) |
| `v01_peak_vq.tex` | IEEE conference LaTeX (compile via Overleaf or local TeXLive) |
| `figures/F0_architecture.png` | System architecture diagram (training / codebook / cold inference) |
| `figures/F1_ablation_e1.png` | E1: peak_aux ON/OFF bar chart (V0 + W5) |
| `figures/F2_multiseed_e3.png` | E3: 3-seed PAPE/HR with error bars |
| `figures/F3_arms_pareto.png` | iter4: PAPE × HR@1 Pareto across mechanisms |
| `figures/F4_cluster_benefit.png` | E4: per-cluster cold benefit scatter |
| `figures/F5_training_curves.png` | T2 training history (loss + val curves) |
| `figures/F6_peak_hour_dist.png` | True peak-hour histogram (train vs cold) |

## Compile LaTeX

Local (TeXLive/MikTeX):
```bash
pdflatex v01_peak_vq.tex
pdflatex v01_peak_vq.tex   # second pass for cross-refs
```

Overleaf: upload `v01_peak_vq.tex` + `figures/` folder, compile with
pdfLaTeX. The .tex includes an inline `thebibliography` block, so a
separate `.bib` is not strictly required.

## Reproducing the experiments cited in the paper

| Section | Script |
|---------|--------|
| Section 4.1 (training arms T0/T2) | `experiments/v01_peak_from_latent/01_train_arms.py` |
| Section 4.2 (main result) | `04_quantize_h1b.py` + `05_coldstart_h1c.py` |
| Section 4.3 (E1 ablation) | `15_E1_peak_aux_ablation.py` |
| Section 4.4 (E3 multi-seed) | `16_E3_multiseed.py` |
| Section 4.5 (E4 per-cluster) | `17_E4_cluster_benefit.py` |
| Section 5.1 (external info) | `12_iter5B_calendar.py`, `19_iv_weather_calendar.py` |
| Final aggregate report | `18_E_aggregate.py` |

All scripts assume the active working directory is the project root and the
project is uv-installed (`uv sync`). Outputs land in
`outputs/v01_peak_from_latent/`.

## Status

- v01 draft: complete (markdown + LaTeX + figures)
- LaTeX not yet compiled locally (no pdflatex on dev machine)
- All ablations and stability evidence are in `outputs/`
- **v02 (FL-aligned 80:20 split + pretrain-size sweep)**: scaffolding only
  (`papers/v02_draft/`, `experiments/v02_fl_8020_ratio/`,
  `outputs/v02_fl_8020_ratio/`). Plan: `plans/v02-01_fl_8020_ratio.md`.
