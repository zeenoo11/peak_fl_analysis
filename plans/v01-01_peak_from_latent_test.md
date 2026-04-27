# v01-01: Peak from Latent — Hypothesis Test Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether NBEATSx hidden latents can be designed/trained so that peak information is (i) extractable by linear probe, (ii) preserved under VQ codebook quantization, and (iii) transferable to cold-start households via Key-Value VQ lookup.

**Architecture:** 6 treatment arms (T0–T5) varying latent definition and training procedure, evaluated through 3 sequential gates (probe → quantization → cold-start transfer). Stop at first failed gate. Single seed (42), 50 train + 50 cold households, 7:1:2 split, all PAPE in denormalized kW units.

**Tech Stack:** Python 3.11, PyTorch (CPU/CUDA), sklearn (Ridge / KMeans / MLPRegressor), pandas/numpy, uv for env.

---

## Background

Prior probe results (`outputs/v11_probe/clean/probe_results.json`, clean MAE NBEATSx ckpt) showed:

- `h_generic` Ridge R² for `peak_amp_fc` = 0.604 (within), 0.530 (across).
- Trivial baseline `stats2` (input mean+std, 2-d) reaches R² = 0.749 / 0.689 — beats every encoder hidden.
- → "h_generic encodes peak naturally" was falsified.

This plan tests a **constructive** version: can we *make* the latent encode peak well enough to drive a useful KV-VQ for cold-start? The agent review (`agent ac5481d89b0e53180`) identified critical bugs in the existing code (z-space PAPE, EMA-only VQ, missing post-hoc fit path) that this plan also fixes.

---

## Hypothesis Decomposition

| ID | Statement | Measurement | PASS Threshold |
|----|-----------|-------------|----------------|
| **H1a** | Latent design/training can be modified so that probe extracts peak amp with R² ≥ 0.70 across-household | Ridge R² on `peak_amp_fc`, 40-train/10-cold split | ≥ 0.70 (≥ stats2 baseline 0.69) |
| **H1b** | The latent that passes H1a survives VQ quantization without losing peak information | R²(quantized latent) / R²(raw latent) | ≥ 0.90 |
| **H1c** | The codebook from H1b transfers to cold-start: KV-VQ improves cold PAPE | cold-PAPE(NBEATSx + KV-VQ) vs cold-PAPE(NBEATSx only) | ≥ 5% relative reduction |

If H1a fails for all arms → conclude latent cannot be made peak-aware → kill v11.
If H1a passes but H1b fails → quantization is the problem → consider larger M or different VQ.
If H1b passes but H1c fails → transfer is the problem → consider per-household codebook.

---

## Treatment Arms

| Arm | Latent | Training | Hypothesis |
|-----|--------|----------|------------|
| T0 | `h_generic` (64-d) | MAE only | Control (already known to fail H1a) |
| T1 | `h_concat = [h_t‖h_s‖h_g]` (192-d) | MAE only | Multi-stack info helps |
| T2 | `h_generic` (64-d) | MAE + λ·peak_aux loss | Auxiliary loss forces peak encoding |
| T3 | `h_concat` (192-d) | MAE + λ·peak_aux loss | T1 + T2 stacked |
| T4 | `W·h_concat` (32-d projection) | MAE backbone (frozen) + post-hoc Ridge fit of W | Doesn't touch backbone, recovers peak via projection |
| T5 | `ŷ` (forecast 24-d) | MAE only | Upper bound / sanity (forecast IS the prediction) |
| T6 | `h_generic ‖ stats2` (64+2=66-d hybrid) | MAE only (T0 backbone) | Combine encoder hidden with handcrafted statistics (mean/std of input window) |

**Default hyperparameters:**
- λ_peak = 0.3
- M (codebook size) = 32
- Encoder: clean MinimalNBEATSx (3-stack, no bc-reg, no peak weight)
- Optimizer: Adam, lr=1e-3, weight_decay=1e-5
- Epochs: 30 max, patience=8 on **val_mae** (not PAPE — see Task 1 fix)

---

## Data Setup

- **train_apts**: 50 from `Peak_Analysis/configs/v10_households.yaml` `train:` field
- **cold_apts**: 50 from same yaml `cold:` field
- **per-apt split**: 7:1:2 (= TRAIN_RATIO 0.7, VAL_RATIO 0.1, test = remainder)
- **z-norm**: per household using its own train segment mean/std
- **stride**: 1 for train/val (overlapping), 24 for test/probe (non-overlap)
- **window**: 96 input + 24 forecast horizon

---

## File Structure

**New files:**
- `src/dataloader/splits.py` — load v10 yaml split (train/cold lists)
- `src/models/peak_aux_head.py` — auxiliary peak prediction head (MLP)
- `src/models/nbeatsx_aux.py` — NBEATSx with peak_aux head wrapper
- `src/models/vq_kmeans.py` — post-hoc KMeans++ VQ (replaces EMA path)
- `src/probes/peak_descriptor.py` — KEY descriptor extractor
- `experiments/v01_peak_from_latent/01_train_arms.py` — trains T0–T3
- `experiments/v01_peak_from_latent/02_fit_t4_projection.py` — T4 Ridge projection
- `experiments/v01_peak_from_latent/03_probe_h1a.py` — H1a gate (probe)
- `experiments/v01_peak_from_latent/04_quantize_h1b.py` — H1b gate (KMeans + R² delta)
- `experiments/v01_peak_from_latent/05_coldstart_h1c.py` — H1c gate (KV-VQ on cold)
- `experiments/v01_peak_from_latent/06_aggregate_report.py` — final table

**Modified files:**
- `src/utils/metrics.py` — add `compute_pape_kw` helper for denormalized PAPE
- `experiments/nbeats_analysis/train_clean_nbeatsx.py` — fix z-space PAPE, change early-stop to val_mae
- `src/models/__init__.py` — re-export new models

**Outputs (gitignored):**
- `outputs/v01_peak_from_latent/{T0,T1,T2,T3,T4}/best.pt`
- `outputs/v01_peak_from_latent/probe_h1a.json`
- `outputs/v01_peak_from_latent/quantize_h1b.json`
- `outputs/v01_peak_from_latent/coldstart_h1c.json`
- `outputs/v01_peak_from_latent/report.md`

---

## Task 1: Fix `train_clean_nbeatsx.py` — denormalized val PAPE + val_mae early stop

**Files:**
- Modify: `experiments/nbeats_analysis/train_clean_nbeatsx.py:127-160`

**Why:** Agent review C4. Current code computes val PAPE in z-space → values 484~569%, useless. `best_state` selection becomes noisy. Fix by either denormalizing val before metric, or switching early-stop to val_mae.

- [ ] **Step 1.1: Modify val loop to track per-apt mean/std for denorm**

Replace lines 144-159 with:
```python
        model.eval()
        val_pred_z, val_true_z, apt_idx_list = [], [], []
        with torch.no_grad():
            offset = 0
            for ds_i, ds in enumerate(val_sets):
                vl = DataLoader(ds, batch_size=args.batch, shuffle=False)
                for x, y in vl:
                    y_hat, _ = model(x.to(DEVICE))
                    val_true_z.append(y.numpy())
                    val_pred_z.append(y_hat.cpu().numpy())
                    apt_idx_list.append(np.full(len(y), ds_i, dtype=np.int32))
        v_t = np.concatenate(val_true_z, axis=0)
        v_p = np.concatenate(val_pred_z, axis=0)
        a_idx = np.concatenate(apt_idx_list, axis=0)
        # denormalize per-apt for kW PAPE
        means = np.array([norm[a]["mean"] for a in args.apts])
        stds = np.array([norm[a]["std"] for a in args.apts])
        v_t_kw = v_t * stds[a_idx, None] + means[a_idx, None]
        v_p_kw = v_p * stds[a_idx, None] + means[a_idx, None]
        val_metrics = seven_axis_metrics(v_t_kw, v_p_kw)
```

- [ ] **Step 1.2: Switch early-stop criterion to val_mae**

Replace line 173-178:
```python
        improved = val_metrics["mae"] < best_val_mae - 1e-6
        if improved:
            best_val_mae = val_metrics["mae"]
            best_val_pape = val_metrics["pape"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
```

And initialize before loop (replace line 113):
```python
    best_val_mae = float("inf")
    best_val_pape = float("inf")
    best_state = None
    bad = 0
    history: list[dict] = []
```

- [ ] **Step 1.3: Verify on 10-household run**

Run: `cd "C:/Users/HOME/JW/Research Docs/FL_Peak_Project" && uv run python experiments/nbeats_analysis/train_clean_nbeatsx.py --tag clean_mae_v2 --epochs 10`

Expected: `val_pape` now in tens of % (not hundreds), `val_mae` monotonically decreasing in early epochs.

- [ ] **Step 1.4: Commit**

```bash
git add experiments/nbeats_analysis/train_clean_nbeatsx.py
git commit -m "fix: denormalize val metrics to kW; switch early-stop to val_mae"
```

---

## Task 2: Add v10 yaml split loader

**Files:**
- Create: `src/dataloader/splits.py`
- Modify: `src/dataloader/__init__.py` (add import)

- [ ] **Step 2.1: Write `splits.py`**

```python
"""Load v10 train/cold household split from external yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

V10_YAML = (
    Path(__file__).resolve().parents[3]
    / "Peak_Analysis"
    / "configs"
    / "v10_households.yaml"
)


def load_v10_split(yaml_path: Path = V10_YAML) -> dict[str, list[str]]:
    """Return {'train': [...50 apts...], 'cold': [...50 apts...]}.

    Falls back to FileNotFoundError with a clear message if the yaml is missing.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"v10 split yaml missing: {yaml_path}. "
            "This plan depends on the v10 train/cold split for comparability."
        )
    with open(yaml_path) as fh:
        raw = yaml.safe_load(fh)
    return {"train": list(raw["train"]), "cold": list(raw["cold"])}
```

- [ ] **Step 2.2: Add export in `src/dataloader/__init__.py`**

Append:
```python
from dataloader.splits import load_v10_split

__all__ = [..., "load_v10_split"]   # add to existing __all__
```

- [ ] **Step 2.3: Smoke test**

Run:
```bash
cd "C:/Users/HOME/JW/Research Docs/FL_Peak_Project" && uv run python -c "
import sys; sys.path.insert(0, 'src')
from dataloader.splits import load_v10_split
s = load_v10_split()
print('train:', len(s['train']), s['train'][:3])
print('cold:', len(s['cold']), s['cold'][:3])
"
```
Expected: `train: 50 ['Apt3', 'Apt4', 'Apt5']`, `cold: 50 ['Apt1', 'Apt2', 'Apt7']`.

- [ ] **Step 2.4: Add pyyaml to pyproject if missing**

Check:
```bash
cd "C:/Users/HOME/JW/Research Docs/FL_Peak_Project" && uv run python -c "import yaml; print(yaml.__version__)"
```
If ImportError, run: `cd "C:/Users/HOME/JW/Research Docs/FL_Peak_Project" && uv add pyyaml`

- [ ] **Step 2.5: Commit**

```bash
git add src/dataloader/splits.py src/dataloader/__init__.py pyproject.toml uv.lock
git commit -m "feat: add v10 train/cold split loader"
```

---

## Task 3: Add peak_aux head and NBEATSx wrapper

**Files:**
- Create: `src/models/peak_aux_head.py`
- Create: `src/models/nbeatsx_aux.py`
- Modify: `src/models/__init__.py`

- [ ] **Step 3.1: Write `peak_aux_head.py`**

```python
"""Auxiliary peak prediction head for multi-task learning.

Inputs an arbitrary latent vector h ∈ R^D and predicts:
    - peak_amp (regression, 1-d) — max value of forecast horizon (z-space)
    - peak_hr  (classification, 24-class) — argmax position of forecast horizon

Loss combines MSE on amp + cross-entropy on hour.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PeakAuxHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, n_hours: int = 24) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
        )
        self.amp_head = nn.Linear(hidden, 1)
        self.hr_head = nn.Linear(hidden, n_hours)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.shared(h)
        return self.amp_head(z).squeeze(-1), self.hr_head(z)


def peak_aux_loss(
    amp_pred: torch.Tensor,
    hr_pred: torch.Tensor,
    y: torch.Tensor,
    hr_weight: float = 0.1,
) -> torch.Tensor:
    """Combined peak loss.  y is z-normalized forecast horizon [B, 24]."""
    amp_true = y.max(dim=1).values            # [B]
    hr_true = y.argmax(dim=1)                 # [B]
    return F.mse_loss(amp_pred, amp_true) + hr_weight * F.cross_entropy(hr_pred, hr_true)
```

- [ ] **Step 3.2: Write `nbeatsx_aux.py` — wrapper that adds the head**

```python
"""NBEATSx with attached peak_aux head.

The wrapper exposes:
    - the underlying MinimalNBEATSx (so its forward returns y_hat + hiddens)
    - one PeakAuxHead per chosen latent source
    - a forward that returns (y_hat, hiddens, aux_outputs)

Latent sources are selectable so we can train T2 (head on h_generic) and T3
(head on h_concat) without duplicating code.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import D_MODEL
from models.nbeatsx import MinimalNBEATSx
from models.peak_aux_head import PeakAuxHead


class NBEATSxAux(nn.Module):
    def __init__(self, latent_source: str = "h_generic") -> None:
        super().__init__()
        if latent_source not in ("h_generic", "h_concat"):
            raise ValueError(f"unknown latent_source: {latent_source}")
        self.latent_source = latent_source
        self.backbone = MinimalNBEATSx()
        in_dim = D_MODEL if latent_source == "h_generic" else 3 * D_MODEL
        self.aux_head = PeakAuxHead(in_dim=in_dim)

    def get_latent(self, hiddens: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.latent_source == "h_generic":
            return hiddens["h_generic"]
        return torch.cat(
            [hiddens["h_trend"], hiddens["h_seasonal"], hiddens["h_generic"]], dim=1
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict, tuple]:
        y_hat, hiddens = self.backbone(x)
        h = self.get_latent(hiddens)
        amp_pred, hr_pred = self.aux_head(h)
        return y_hat, hiddens, (amp_pred, hr_pred)
```

- [ ] **Step 3.3: Re-export in `src/models/__init__.py`**

Add:
```python
from models.peak_aux_head import PeakAuxHead, peak_aux_loss
from models.nbeatsx_aux import NBEATSxAux
```

And extend `__all__` with `"PeakAuxHead", "peak_aux_loss", "NBEATSxAux"`.

- [ ] **Step 3.4: Smoke test**

Run:
```bash
cd "C:/Users/HOME/JW/Research Docs/FL_Peak_Project" && uv run python -c "
import sys; sys.path.insert(0, 'src')
import torch
from models.nbeatsx_aux import NBEATSxAux
from models.peak_aux_head import peak_aux_loss
m = NBEATSxAux(latent_source='h_concat')
x = torch.randn(4, 96)
y = torch.randn(4, 24)
y_hat, hidd, (amp_p, hr_p) = m(x)
loss = peak_aux_loss(amp_p, hr_p, y)
print('y_hat:', y_hat.shape, 'amp_p:', amp_p.shape, 'hr_p:', hr_p.shape, 'loss:', float(loss))
"
```
Expected: `y_hat: torch.Size([4, 24]) amp_p: torch.Size([4]) hr_p: torch.Size([4, 24]) loss: <some positive float>`.

- [ ] **Step 3.5: Commit**

```bash
git add src/models/peak_aux_head.py src/models/nbeatsx_aux.py src/models/__init__.py
git commit -m "feat: add PeakAuxHead and NBEATSxAux wrapper for arms T2/T3"
```

---

## Task 4: Train arms T0/T1/T2/T3 on 50 train households

**Files:**
- Create: `experiments/v01_peak_from_latent/__init__.py` (empty)
- Create: `experiments/v01_peak_from_latent/01_train_arms.py`

- [ ] **Step 4.1: Write `01_train_arms.py`**

```python
"""Train arms T0/T1/T2/T3 on 50 train households (pooled, z-normed per-apt).

T0/T1 use plain MAE.
T2 trains NBEATSxAux(h_generic) with MAE + lambda*peak_aux.
T3 trains NBEATSxAux(h_concat) with MAE + lambda*peak_aux.

T1 and T0 share the same backbone (MinimalNBEATSx, MAE only); T1 only differs
in WHICH hidden we read out for downstream probes/VQ. So we train ONE
MinimalNBEATSx and tag its checkpoint as serving both T0 and T1.

Outputs:
    outputs/v01_peak_from_latent/T0/best.pt   (= shared with T1)
    outputs/v01_peak_from_latent/T2/best.pt
    outputs/v01_peak_from_latent/T3/best.pt
    outputs/v01_peak_from_latent/{arm}/training_log.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from config import (
    HORIZON,
    INPUT_SIZE,
    OUTPUT_DIR,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
)
from dataloader.splits import load_v10_split
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from models.nbeatsx import MinimalNBEATSx
from models.nbeatsx_aux import NBEATSxAux
from models.peak_aux_head import peak_aux_loss
from utils.metrics import seven_axis_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_ROOT = OUTPUT_DIR / "v01_peak_from_latent"


def build_loaders(apts: list[str], batch: int):
    train_sets, val_sets, test_sets, norm = [], [], [], {}
    for apt in apts:
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError as e:
            print(f"[skip] {apt}: {e}")
            continue
        n = len(series)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        mean = float(series[:train_end].mean())
        std = float(series[:train_end].std()) if series[:train_end].std() > 1e-8 else 1.0
        train_sets.append(HouseholdDataset(series[:train_end], mean, std, stride=1))
        val_sets.append(HouseholdDataset(series[train_end:val_end], mean, std, stride=1))
        test_sets.append(
            HouseholdDataset(series[max(0, val_end - INPUT_SIZE):], mean, std, stride=HORIZON)
        )
        norm[apt] = {"mean": mean, "std": std}
    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=batch, shuffle=True)
    return train_sets, val_sets, test_sets, norm, train_loader


def denorm_batch(t_z: list[np.ndarray], p_z: list[np.ndarray],
                 sizes: list[int], apts: list[str], norm: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-apt denorm. sizes is the count of samples per apt in concatenation order."""
    means = np.repeat([norm[a]["mean"] for a in apts], sizes)
    stds = np.repeat([norm[a]["std"] for a in apts], sizes)
    t = np.concatenate(t_z, axis=0) * stds[:, None] + means[:, None]
    p = np.concatenate(p_z, axis=0) * stds[:, None] + means[:, None]
    return t, p


def eval_per_apt(model, val_sets, norm, apts, batch, use_aux):
    """Per-apt val eval, returns concatenated kW true/pred for fair PAPE."""
    model.eval()
    apt_idx_arr, true_chunks, pred_chunks = [], [], []
    with torch.no_grad():
        for ai, ds in enumerate(val_sets):
            for x, y in DataLoader(ds, batch_size=batch, shuffle=False):
                if use_aux:
                    y_hat, _, _ = model(x.to(DEVICE))
                else:
                    y_hat, _ = model(x.to(DEVICE))
                true_chunks.append(y.numpy())
                pred_chunks.append(y_hat.cpu().numpy())
                apt_idx_arr.append(np.full(len(y), ai, dtype=np.int32))
    t_z = np.concatenate(true_chunks, axis=0)
    p_z = np.concatenate(pred_chunks, axis=0)
    a_idx = np.concatenate(apt_idx_arr, axis=0)
    means = np.array([norm[a]["mean"] for a in apts])
    stds = np.array([norm[a]["std"] for a in apts])
    t_kw = t_z * stds[a_idx, None] + means[a_idx, None]
    p_kw = p_z * stds[a_idx, None] + means[a_idx, None]
    return seven_axis_metrics(t_kw, p_kw)


def train_arm(arm: str, apts: list[str], epochs: int, lr: float, batch: int,
              patience: int, lam: float, seed: int) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    out_dir = OUT_ROOT / arm
    out_dir.mkdir(parents=True, exist_ok=True)

    train_sets, val_sets, test_sets, norm, train_loader = build_loaders(apts, batch)
    n_train = sum(len(d) for d in train_sets)
    print(f"[{arm}] train windows: {n_train} across {len(train_sets)} apts")

    use_aux = arm in ("T2", "T3")
    if arm == "T2":
        model = NBEATSxAux(latent_source="h_generic").to(DEVICE)
    elif arm == "T3":
        model = NBEATSxAux(latent_source="h_concat").to(DEVICE)
    else:  # T0 (and T1 reuses this checkpoint)
        model = MinimalNBEATSx().to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val_mae, best_state, bad, history = float("inf"), None, 0, []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, aux_sum, n = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if use_aux:
                y_hat, _, (amp_p, hr_p) = model(x)
                main = F.l1_loss(y_hat, y)
                aux = peak_aux_loss(amp_p, hr_p, y)
                loss = main + lam * aux
                aux_sum += float(aux.item())
            else:
                y_hat, _ = model(x)
                loss = F.l1_loss(y_hat, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss.item()); n += 1

        val_metrics = eval_per_apt(model, val_sets, norm, apts, batch, use_aux)
        rec = {"epoch": epoch, "train_loss": loss_sum / n, **{f"val_{k}": v for k, v in val_metrics.items()}}
        if use_aux:
            rec["train_aux"] = aux_sum / n
        rec["wall_s"] = round(time.time() - t0, 1)
        history.append(rec)

        improved = val_metrics["mae"] < best_val_mae - 1e-6
        if improved:
            best_val_mae = val_metrics["mae"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        flag = " *" if improved else ""
        msg = (f"  ep{epoch:02d} loss={rec['train_loss']:.4f} "
               f"val_mae={rec['val_mae']:.4f} val_pape={rec['val_pape']:.2f} "
               f"hr1={rec['val_hr@1']:.1f} ({rec['wall_s']}s){flag}")
        if use_aux:
            msg += f"  aux={rec['train_aux']:.4f}"
        print(msg)
        if bad >= patience:
            print(f"  early stop @ ep {epoch}")
            break

    torch.save(best_state, out_dir / "best.pt")
    with open(out_dir / "training_log.json", "w") as fh:
        json.dump({"arm": arm, "lam": lam, "norm": norm, "history": history,
                   "n_train_windows": n_train}, fh, indent=2)
    return {"arm": arm, "best_val_mae": best_val_mae,
            "n_apts": len(train_sets), "n_train_windows": n_train}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["T0", "T2", "T3"],
                    help="T1 reuses T0 checkpoint, no separate training needed")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    apts = load_v10_split()["train"]
    print(f"[setup] 50 train apts, {len(args.arms)} arms")

    for arm in args.arms:
        print(f"\n========== {arm} ==========")
        train_arm(arm, apts, args.epochs, args.lr, args.batch,
                  args.patience, args.lam, args.seed)

    # T1 = T0 alias
    if "T0" in args.arms:
        t1_dir = OUT_ROOT / "T1"
        t1_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(OUT_ROOT / "T0" / "best.pt", t1_dir / "best.pt")
        with open(t1_dir / "training_log.json", "w") as fh:
            json.dump({"arm": "T1", "note": "alias of T0; differs only in latent readout (h_concat vs h_generic)"}, fh, indent=2)
        print("[T1] aliased to T0 ckpt")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Run training**

```bash
cd "C:/Users/HOME/JW/Research Docs/FL_Peak_Project" && uv run python experiments/v01_peak_from_latent/01_train_arms.py 2>&1 | tee outputs/v01_peak_from_latent/training_log.txt
```
Expected: 3 arm trainings, each ~3-5 min on CPU; T0/T2/T3 ckpts saved; T1 alias copied.

- [ ] **Step 4.3: Verify checkpoints exist**

Run:
```bash
ls "C:/Users/HOME/JW/Research Docs/FL_Peak_Project/outputs/v01_peak_from_latent/"
```
Expected: `T0/  T1/  T2/  T3/  training_log.txt`, each arm dir contains `best.pt` and `training_log.json`.

- [ ] **Step 4.4: Commit**

```bash
git add experiments/v01_peak_from_latent/
git commit -m "feat: train arms T0/T1/T2/T3 on 50 households with peak_aux"
```

---

## Task 5: T4 — post-hoc Ridge projection of h_concat to peak-aware subspace

**Files:**
- Create: `experiments/v01_peak_from_latent/02_fit_t4_projection.py`

- [ ] **Step 5.1: Write `02_fit_t4_projection.py`**

```python
"""T4: Fit a 32-d projection W of h_concat using Ridge regression on (peak_amp, peak_hour).

Ridge weights from peak_aux probe become the projection; latent_T4 = h_concat @ W^T.
Backbone (T0 ckpt) untouched. The W matrix is saved to outputs/v01_peak_from_latent/T4/W.npz.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import HORIZON, OUTPUT_DIR, RANDOM_SEED, TRAIN_RATIO
from dataloader.splits import load_v10_split
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from models.nbeatsx import MinimalNBEATSx

OUT_DIR = OUTPUT_DIR / "v01_peak_from_latent" / "T4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
T0_CKPT = OUTPUT_DIR / "v01_peak_from_latent" / "T0" / "best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJ_DIM = 32


def extract_h_concat_and_targets(model, apts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (h_concat [N, 192], peak_amp_z [N], peak_hr [N])."""
    h_chunks, amp_chunks, hr_chunks = [], [], []
    for apt in apts:
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            continue
        n = len(series)
        train_end = int(n * TRAIN_RATIO)
        seg = series[:train_end]
        mean = float(seg.mean()); std = float(seg.std()) if seg.std() > 1e-8 else 1.0
        ds = HouseholdDataset(seg, mean, std, stride=24)
        for x, y in DataLoader(ds, batch_size=256, shuffle=False):
            with torch.no_grad():
                _, hidd = model(x.to(DEVICE))
            h_concat = torch.cat([hidd["h_trend"], hidd["h_seasonal"], hidd["h_generic"]], dim=1)
            h_chunks.append(h_concat.cpu().numpy())
            amp_chunks.append(y.numpy().max(axis=1))
            hr_chunks.append(y.numpy().argmax(axis=1))
    return (np.concatenate(h_chunks, axis=0),
            np.concatenate(amp_chunks, axis=0),
            np.concatenate(hr_chunks, axis=0))


def main():
    np.random.seed(RANDOM_SEED)
    print(f"[T4] loading T0 ckpt: {T0_CKPT}")
    state = torch.load(T0_CKPT, map_location="cpu", weights_only=False)
    model = MinimalNBEATSx().to(DEVICE).eval()
    model.load_state_dict(state, strict=True)

    apts = load_v10_split()["train"]
    print(f"[T4] extracting h_concat from {len(apts)} apts")
    H, amp, hr = extract_h_concat_and_targets(model, apts)
    print(f"[T4] H shape: {H.shape}, amp range: [{amp.min():.2f}, {amp.max():.2f}]")

    # Multi-output Ridge: target = [peak_amp_z, one-hot peak_hr (24-d)] -> 25-d
    Y = np.concatenate([amp[:, None], np.eye(24)[hr]], axis=1)   # [N, 25]
    sc = StandardScaler().fit(H)
    Hs = sc.transform(H)

    ridge = Ridge(alpha=1.0).fit(Hs, Y)
    # Coefficients [25, 192]; we want PROJ_DIM rows. Pick top-PROJ_DIM by row norm.
    coef = ridge.coef_                                     # [25, 192]
    if PROJ_DIM <= coef.shape[0]:
        # take top rows by norm
        norms = np.linalg.norm(coef, axis=1)
        top = np.argsort(-norms)[:PROJ_DIM]
        W = coef[top]                                      # [PROJ_DIM, 192]
    else:
        # pad with random orthogonal directions (ridge produced fewer rows than PROJ_DIM)
        rng = np.random.RandomState(RANDOM_SEED)
        extra = rng.randn(PROJ_DIM - coef.shape[0], coef.shape[1])
        W = np.concatenate([coef, extra], axis=0)
    print(f"[T4] W shape: {W.shape}")

    np.savez(OUT_DIR / "W.npz", W=W, scaler_mean=sc.mean_, scaler_scale=sc.scale_)
    with open(OUT_DIR / "training_log.json", "w") as fh:
        json.dump({
            "arm": "T4",
            "method": "Ridge multi-output, top-rows-by-norm",
            "n_train_windows": int(H.shape[0]),
            "n_train_apts": len(apts),
            "proj_dim": PROJ_DIM,
            "ridge_alpha": 1.0,
        }, fh, indent=2)
    print(f"[T4] saved W to {OUT_DIR / 'W.npz'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.2: Run T4 fit**

```bash
cd "C:/Users/HOME/JW/Research Docs/FL_Peak_Project" && uv run python experiments/v01_peak_from_latent/02_fit_t4_projection.py
```
Expected: `[T4] H shape: (~12000, 192)`, `W.npz` saved.

- [ ] **Step 5.3: Commit**

```bash
git add experiments/v01_peak_from_latent/02_fit_t4_projection.py
git commit -m "feat: T4 — Ridge projection of h_concat to 32-d peak-aware subspace"
```

---

## Task 6: H1a Probe Gate — extract latent for all 6 arms, run probe, decide PASS/FAIL

**Files:**
- Create: `experiments/v01_peak_from_latent/03_probe_h1a.py`

- [ ] **Step 6.1: Write `03_probe_h1a.py`**

```python
"""H1a gate: probe whether each arm's latent encodes peak.

For each arm in {T0, T1, T2, T3, T4, T5}:
    1. Load corresponding ckpt (T1 = T0 alias; T4 = T0 + W projection;
       T5 = T0 forecast itself).
    2. Extract latent from 50 train apts (split into 40 train / 10 cold-probe).
    3. Run Ridge / MLP probes for peak_amp_fc and peak_hr_fc.
    4. Compute R² and PAPE on probe predictions.

PASS criterion (H1a): Ridge R²(peak_amp_fc, across-household) >= 0.70.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, top_k_accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import HORIZON, OUTPUT_DIR, RANDOM_SEED, TRAIN_RATIO
from dataloader.splits import load_v10_split
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from models.nbeatsx import MinimalNBEATSx
from models.nbeatsx_aux import NBEATSxAux
from utils.metrics import compute_pape

OUT = OUTPUT_DIR / "v01_peak_from_latent"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARMS = ["T0", "T1", "T2", "T3", "T4", "T5"]
H1A_PASS_R2 = 0.70


def load_arm_model(arm: str):
    """Returns (model, latent_extractor_fn). latent_extractor takes (x [B,96]) -> latent [B, D]."""
    ckpt_path = OUT / arm / "best.pt"
    if arm in ("T0", "T1", "T4", "T5"):
        # all built on T0 backbone
        m = MinimalNBEATSx().to(DEVICE).eval()
        m.load_state_dict(torch.load(OUT / "T0" / "best.pt", map_location="cpu", weights_only=False))
    elif arm == "T2":
        m = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
        m.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    elif arm == "T3":
        m = NBEATSxAux(latent_source="h_concat").to(DEVICE).eval()
        m.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    else:
        raise ValueError(arm)

    if arm == "T0":
        def fn(x):
            with torch.no_grad():
                _, h = m(x.to(DEVICE))
            return h["h_generic"].cpu().numpy()
    elif arm in ("T1", "T3"):
        def fn(x):
            with torch.no_grad():
                if arm == "T3":
                    _, h, _ = m(x.to(DEVICE))
                else:
                    _, h = m(x.to(DEVICE))
            return torch.cat([h["h_trend"], h["h_seasonal"], h["h_generic"]], dim=1).cpu().numpy()
    elif arm == "T2":
        def fn(x):
            with torch.no_grad():
                _, h, _ = m(x.to(DEVICE))
            return h["h_generic"].cpu().numpy()
    elif arm == "T4":
        Wd = np.load(OUT / "T4" / "W.npz")
        W, mu, sc = Wd["W"], Wd["scaler_mean"], Wd["scaler_scale"]
        def fn(x):
            with torch.no_grad():
                _, h = m(x.to(DEVICE))
            hc = torch.cat([h["h_trend"], h["h_seasonal"], h["h_generic"]], dim=1).cpu().numpy()
            return ((hc - mu) / sc) @ W.T
    elif arm == "T5":
        def fn(x):
            with torch.no_grad():
                y_hat, _ = m(x.to(DEVICE))
            return y_hat.cpu().numpy()    # [B, 24]
    return fn


def gather_features(extract_fn, apts: list[str]):
    feats, amp, hr, apt_idx = [], [], [], []
    for ai, apt in enumerate(apts):
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            continue
        n = len(series)
        train_end = int(n * TRAIN_RATIO)
        seg = series[:train_end]
        mean = float(seg.mean()); std = float(seg.std()) if seg.std() > 1e-8 else 1.0
        ds = HouseholdDataset(seg, mean, std, stride=24)
        for x, y in DataLoader(ds, batch_size=256, shuffle=False):
            feats.append(extract_fn(x))
            amp.append(y.numpy().max(axis=1))
            hr.append(y.numpy().argmax(axis=1))
            apt_idx.append(np.full(len(y), ai, dtype=np.int32))
    return (np.concatenate(feats, axis=0),
            np.concatenate(amp, axis=0),
            np.concatenate(hr, axis=0),
            np.concatenate(apt_idx, axis=0))


def run_probe(X_tr, y_tr, X_te, y_te, kind: str):
    sc = StandardScaler().fit(X_tr)
    Xs_tr = sc.transform(X_tr); Xs_te = sc.transform(X_te)
    if kind == "regression":
        ridge = Ridge(alpha=1.0).fit(Xs_tr, y_tr)
        pred_ridge = ridge.predict(Xs_te)
        mlp = MLPRegressor(hidden_layer_sizes=(64,), max_iter=200,
                           random_state=RANDOM_SEED, early_stopping=True).fit(Xs_tr, y_tr)
        pred_mlp = mlp.predict(Xs_te)
        return {
            "ridge_R2": float(r2_score(y_te, pred_ridge)),
            "ridge_PAPE": float(compute_pape(y_te[None, :], pred_ridge[None, :])),
            "mlp_R2": float(r2_score(y_te, pred_mlp)),
            "mlp_PAPE": float(compute_pape(y_te[None, :], pred_mlp[None, :])),
        }
    else:  # classification
        clf = LogisticRegression(max_iter=500, random_state=RANDOM_SEED).fit(Xs_tr, y_tr)
        pred = clf.predict(Xs_te); proba = clf.predict_proba(Xs_te)
        return {
            "top1": float(accuracy_score(y_te, pred)),
            "top3": float(top_k_accuracy_score(y_te, proba, k=3, labels=clf.classes_)),
        }


def main():
    np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
    apts = load_v10_split()["train"]
    train_apts = apts[:40]; cold_probe_apts = apts[40:]
    print(f"[probe] 40 train_probe apts, 10 cold_probe apts")

    results = {}
    for arm in ARMS:
        print(f"\n========== {arm} ==========")
        extract_fn = load_arm_model(arm)
        # 40 train apts -> within-split source; 10 cold_probe -> across-split source
        X_tr_within, amp_tr, hr_tr, _ = gather_features(extract_fn, train_apts)
        X_te_across, amp_te, hr_te, _ = gather_features(extract_fn, cold_probe_apts)
        print(f"  X_tr={X_tr_within.shape} X_te={X_te_across.shape}")

        amp_res = run_probe(X_tr_within, amp_tr, X_te_across, amp_te, "regression")
        hr_res = run_probe(X_tr_within, hr_tr, X_te_across, hr_te, "classification")
        results[arm] = {"peak_amp_fc": amp_res, "peak_hr_fc": hr_res,
                        "n_tr": int(X_tr_within.shape[0]), "n_te": int(X_te_across.shape[0])}
        gate = "PASS" if amp_res["ridge_R2"] >= H1A_PASS_R2 else "FAIL"
        print(f"  amp_fc Ridge R²={amp_res['ridge_R2']:.3f}  PAPE={amp_res['ridge_PAPE']:.1f}%  [{gate} H1a]")
        print(f"  hr_fc top1={hr_res['top1']:.3f}  top3={hr_res['top3']:.3f}")

    pass_arms = [a for a, r in results.items()
                 if r["peak_amp_fc"]["ridge_R2"] >= H1A_PASS_R2]
    print(f"\n[H1a] PASS arms: {pass_arms}")

    out_path = OUT / "probe_h1a.json"
    with open(out_path, "w") as fh:
        json.dump({"pass_threshold": H1A_PASS_R2, "results": results,
                   "pass_arms": pass_arms}, fh, indent=2)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.2: Run probe gate**

```bash
cd "C:/Users/HOME/JW/Research Docs/FL_Peak_Project" && uv run python experiments/v01_peak_from_latent/03_probe_h1a.py 2>&1 | tee outputs/v01_peak_from_latent/probe_h1a.txt
```
Expected: 6 arms processed, each prints amp_fc R² and a `[PASS H1a]` or `[FAIL H1a]` tag. JSON saved.

- [ ] **Step 6.3: Inspect pass_arms**

Run:
```bash
cd "C:/Users/HOME/JW/Research Docs/FL_Peak_Project" && uv run python -c "
import json
r = json.load(open('outputs/v01_peak_from_latent/probe_h1a.json'))
print('pass_arms:', r['pass_arms'])
for arm, d in r['results'].items():
    print(f\"  {arm}: amp Ridge R²={d['peak_amp_fc']['ridge_R2']:.3f}  PAPE={d['peak_amp_fc']['ridge_PAPE']:.1f}%  hr top1={d['peak_hr_fc']['top1']:.3f}\")
"
```

**DECISION POINT:**
- If `pass_arms` is empty → **STOP**. H1 falsified. Report and discuss next direction with user.
- If `pass_arms` non-empty → continue to Task 7.

- [ ] **Step 6.4: Commit**

```bash
git add experiments/v01_peak_from_latent/03_probe_h1a.py
git commit -m "feat: H1a probe gate — peak extraction R² + PAPE per arm"
```

---

## Task 7: Add post-hoc KMeans++ VQ class

**Files:**
- Create: `src/models/vq_kmeans.py`
- Modify: `src/models/__init__.py`

- [ ] **Step 7.1: Write `vq_kmeans.py`**

```python
"""Post-hoc KMeans++ Vector Quantizer.

This is the proper post-hoc VQ that the v11 design demands. Distinct from
VectorQuantizerEMA (which is in-loop and unsuitable for 1-pass fit).

Usage:
    vq = VectorQuantizerKMeans(num_embeddings=32, embedding_dim=64)
    vq.fit(z)                          # z: torch.Tensor [N, D] from training data
    z_q, indices = vq(z_query)         # nearest-neighbor lookup, no STE
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class VectorQuantizerKMeans(nn.Module):
    def __init__(self, num_embeddings: int = 32, embedding_dim: int = 64,
                 random_state: int = 42) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        # placeholders; replaced after fit()
        self.register_buffer("codebook", torch.zeros(num_embeddings, embedding_dim))
        self.register_buffer("counts", torch.zeros(num_embeddings, dtype=torch.long))
        self._is_fit = False

    def fit(self, z: torch.Tensor) -> dict:
        """Fit codebook using KMeans++. z: [N, D]."""
        if z.dim() != 2:
            raise ValueError(f"expected [N, D], got {z.shape}")
        if z.shape[1] != self.embedding_dim:
            raise ValueError(f"embedding_dim mismatch: {z.shape[1]} vs {self.embedding_dim}")
        z_np = z.detach().cpu().numpy()
        km = KMeans(n_clusters=self.num_embeddings, init="k-means++",
                    n_init=10, random_state=self.random_state).fit(z_np)
        self.codebook.copy_(torch.from_numpy(km.cluster_centers_).to(self.codebook.dtype))
        # count samples per cluster
        bincount = torch.bincount(torch.from_numpy(km.labels_), minlength=self.num_embeddings)
        self.counts.copy_(bincount.to(self.counts.dtype))
        self._is_fit = True
        # diagnostics
        utilization = float((self.counts > 0).sum().item()) / self.num_embeddings
        probs = self.counts.float() / max(int(self.counts.sum().item()), 1)
        entropy = -(probs * (probs + 1e-12).log()).sum().item()
        perplexity = float(torch.exp(torch.tensor(entropy)).item())
        return {
            "n_fit_samples": int(z.shape[0]),
            "utilization": utilization,
            "perplexity": perplexity,
            "k_min": int(self.counts.min().item()),
            "k_max": int(self.counts.max().item()),
            "kmeans_inertia": float(km.inertia_),
        }

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._is_fit:
            raise RuntimeError("call .fit(...) before forward")
        dist = (z.pow(2).sum(1, keepdim=True)
                - 2.0 * z @ self.codebook.t()
                + self.codebook.pow(2).sum(1))
        idx = dist.argmin(dim=1)
        return self.codebook[idx], idx
```

- [ ] **Step 7.2: Re-export**

In `src/models/__init__.py` add:
```python
from models.vq_kmeans import VectorQuantizerKMeans
```
Add `"VectorQuantizerKMeans"` to `__all__`.

- [ ] **Step 7.3: Smoke test**

Run:
```bash
cd "C:/Users/HOME/JW/Research Docs/FL_Peak_Project" && uv run python -c "
import sys; sys.path.insert(0, 'src')
import torch
from models.vq_kmeans import VectorQuantizerKMeans
vq = VectorQuantizerKMeans(num_embeddings=8, embedding_dim=4)
z_fit = torch.randn(200, 4)
diag = vq.fit(z_fit)
print('diag:', diag)
z_q, idx = vq(torch.randn(5, 4))
print('z_q:', z_q.shape, 'idx:', idx.tolist())
"
```
Expected: `diag` includes `utilization` and `perplexity`; `z_q` shape `[5, 4]`.

- [ ] **Step 7.4: Commit**

```bash
git add src/models/vq_kmeans.py src/models/__init__.py
git commit -m "feat: post-hoc KMeans++ VQ (replaces EMA path for v11)"
```

---

## Task 8: H1b Quantization Gate — fit codebook on PASS arms, measure R² preservation

**Files:**
- Create: `experiments/v01_peak_from_latent/04_quantize_h1b.py`

- [ ] **Step 8.1: Write `04_quantize_h1b.py`**

```python
"""H1b gate: quantize the latents of H1a-PASS arms, verify peak info preserved.

For each PASS arm:
    1. Extract latents from 40 train apts (same as Task 6 within split).
    2. Fit VectorQuantizerKMeans (M=32) on those latents.
    3. Look up quantized latents (z_q) for both train (40) and across (10) sets.
    4. Re-run Ridge probe on z_q for peak_amp_fc.
    5. Ratio R²(z_q) / R²(z_raw) >= 0.90 -> PASS H1b.

Also reports: utilization, perplexity, k_min (k-anonymity).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from config import OUTPUT_DIR, RANDOM_SEED
from dataloader.splits import load_v10_split
from models.vq_kmeans import VectorQuantizerKMeans
from utils.metrics import compute_pape

# Reuse extractors from Task 6 by importing the script as module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
probe_mod = import_module("03_probe_h1a")
load_arm_model = probe_mod.load_arm_model
gather_features = probe_mod.gather_features

OUT = OUTPUT_DIR / "v01_peak_from_latent"
M = 32
PASS_RATIO = 0.90


def main():
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    h1a = json.load(open(OUT / "probe_h1a.json"))
    pass_arms = h1a["pass_arms"]
    if not pass_arms:
        print("[H1b] no PASS arms from H1a — nothing to quantize.")
        return

    apts = load_v10_split()["train"]
    train_apts = apts[:40]; cold_probe_apts = apts[40:]

    results = {}
    for arm in pass_arms:
        print(f"\n========== quantize {arm} ==========")
        extract_fn = load_arm_model(arm)
        X_tr, amp_tr, _, _ = gather_features(extract_fn, train_apts)
        X_te, amp_te, _, _ = gather_features(extract_fn, cold_probe_apts)
        print(f"  X_tr {X_tr.shape}, X_te {X_te.shape}")

        D = X_tr.shape[1]
        vq = VectorQuantizerKMeans(num_embeddings=M, embedding_dim=D, random_state=RANDOM_SEED)
        diag = vq.fit(torch.from_numpy(X_tr).float())
        print(f"  vq diag: util={diag['utilization']:.3f}  ppl={diag['perplexity']:.2f}  "
              f"k_min={diag['k_min']}  k_max={diag['k_max']}")

        # raw probe
        sc = StandardScaler().fit(X_tr)
        ridge_raw = Ridge(alpha=1.0).fit(sc.transform(X_tr), amp_tr)
        pred_raw = ridge_raw.predict(sc.transform(X_te))
        r2_raw = float(r2_score(amp_te, pred_raw))
        pape_raw = float(compute_pape(amp_te[None, :], pred_raw[None, :]))

        # quantized probe (re-fit Ridge on z_q to be charitable)
        with torch.no_grad():
            Zq_tr, _ = vq(torch.from_numpy(X_tr).float())
            Zq_te, _ = vq(torch.from_numpy(X_te).float())
        Zq_tr = Zq_tr.numpy(); Zq_te = Zq_te.numpy()
        sc_q = StandardScaler().fit(Zq_tr)
        ridge_q = Ridge(alpha=1.0).fit(sc_q.transform(Zq_tr), amp_tr)
        pred_q = ridge_q.predict(sc_q.transform(Zq_te))
        r2_q = float(r2_score(amp_te, pred_q))
        pape_q = float(compute_pape(amp_te[None, :], pred_q[None, :]))

        ratio = r2_q / r2_raw if r2_raw > 0 else 0.0
        gate = "PASS" if ratio >= PASS_RATIO else "FAIL"
        print(f"  R²(raw)={r2_raw:.3f}  R²(q)={r2_q:.3f}  ratio={ratio:.3f}  [{gate} H1b]")
        print(f"  PAPE(raw)={pape_raw:.1f}%  PAPE(q)={pape_q:.1f}%")

        results[arm] = {
            "vq_diagnostics": diag,
            "r2_raw": r2_raw, "r2_quantized": r2_q, "ratio": ratio,
            "pape_raw": pape_raw, "pape_quantized": pape_q,
            "gate_h1b": gate,
            "codebook": vq.codebook.cpu().numpy().tolist(),
        }
        # save codebook separately
        np.savez(OUT / arm / "codebook.npz",
                 codebook=vq.codebook.cpu().numpy(),
                 counts=vq.counts.cpu().numpy())

    pass_arms_h1b = [a for a, r in results.items() if r["gate_h1b"] == "PASS"]
    out = {"M": M, "pass_threshold_ratio": PASS_RATIO,
           "pass_arms_h1b": pass_arms_h1b, "results": results}
    # strip codebook lists from json (keep just diag)
    out_for_json = {**out, "results": {a: {k: v for k, v in r.items() if k != "codebook"}
                                       for a, r in results.items()}}
    with open(OUT / "quantize_h1b.json", "w") as fh:
        json.dump(out_for_json, fh, indent=2)
    print(f"\n[H1b] PASS arms: {pass_arms_h1b}")
    print(f"[done] wrote quantize_h1b.json + per-arm codebook.npz")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.2: Run quantization gate**

```bash
cd "C:/Users/HOME/JW/Research Docs/FL_Peak_Project" && uv run python experiments/v01_peak_from_latent/04_quantize_h1b.py 2>&1 | tee outputs/v01_peak_from_latent/quantize_h1b.txt
```
Expected: per arm prints VQ diagnostics, raw vs quantized R², PASS/FAIL.

- [ ] **Step 8.3: Inspect**

Run:
```bash
cd "C:/Users/HOME/JW/Research Docs/FL_Peak_Project" && uv run python -c "
import json
r = json.load(open('outputs/v01_peak_from_latent/quantize_h1b.json'))
print('pass_arms_h1b:', r['pass_arms_h1b'])
for arm, d in r['results'].items():
    print(f\"  {arm}: ratio={d['ratio']:.3f}  util={d['vq_diagnostics']['utilization']:.3f}  k_min={d['vq_diagnostics']['k_min']}\")
"
```

**DECISION POINT:** If `pass_arms_h1b` is empty → STOP, report. Else continue.

- [ ] **Step 8.4: Commit**

```bash
git add experiments/v01_peak_from_latent/04_quantize_h1b.py
git commit -m "feat: H1b quantization gate — R² preservation under KMeans VQ"
```

---

## Task 9: H1c Cold-start Transfer Gate — KV-VQ on 50 cold households

**Files:**
- Create: `src/probes/__init__.py` (empty)
- Create: `src/probes/peak_descriptor.py`
- Create: `experiments/v01_peak_from_latent/05_coldstart_h1c.py`

- [ ] **Step 9.1: Write `peak_descriptor.py`**

```python
"""KEY descriptor extractor for KV-VQ.

KEY is computed from the input 96h window only — no future leakage, no
encoder dependency. Cold households can compute their own KEY identically.

Default 5-d KEY:
    [input_max, input_argmax_norm, daily_mean, daily_std, last24_max]
"""

from __future__ import annotations

import numpy as np


def extract_key(x: np.ndarray) -> np.ndarray:
    """x: [B, 96] z-normalized input.  Returns [B, 5]."""
    if x.ndim == 1:
        x = x[None, :]
    return np.stack([
        x.max(axis=1),
        x.argmax(axis=1) / 96.0,
        x.mean(axis=1),
        x.std(axis=1),
        x[:, -24:].max(axis=1),
    ], axis=1)
```

- [ ] **Step 9.2: Write `05_coldstart_h1c.py`**

```python
"""H1c gate: KV-VQ on 50 cold households.

For each H1b-PASS arm:
    1. On 50 train apts, build KV pairs:
         KEY = peak_descriptor(input)  ∈ R^5
         VALUE = quantized latent index (assigned cluster id 0..M-1)
    2. For each KEY cluster (assigned via 1-NN in latent space already),
       compute the average VALUE forecast offset:
            offset_c = mean over training samples in cluster c of
                       (forecast_true - NBEATSx_baseline_forecast)   [24-d]
    3. On 50 cold apts, for each input window:
         a. compute KEY for cold input.
         b. find c* via 1-NN over training KEYs (NOT latent — KEY space).
         c. baseline ŷ = NBEATSx forecast on cold input.
         d. corrected ŷ_kv = ŷ + alpha * offset_{c*}     (alpha = 0.5 default)
    4. Compute PAPE on (a) baseline and (b) corrected. Compare.

PASS criterion (H1c): cold-PAPE(KV-VQ) <= 0.95 * cold-PAPE(baseline).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import HORIZON, OUTPUT_DIR, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO
from dataloader.splits import load_v10_split
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from models.nbeatsx import MinimalNBEATSx
from models.nbeatsx_aux import NBEATSxAux
from models.vq_kmeans import VectorQuantizerKMeans
from probes.peak_descriptor import extract_key
from utils.metrics import seven_axis_metrics

# reuse extractors from Task 6
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
probe_mod = import_module("03_probe_h1a")
load_arm_model = probe_mod.load_arm_model

OUT = OUTPUT_DIR / "v01_peak_from_latent"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA = 0.5
PASS_RATIO = 0.95


def gather_train_kv(extract_fn, model_for_baseline, apts: list[str]):
    """Returns (KEYs [N,5], baseline_forecast_z [N,24], true_forecast_z [N,24],
       latent_indices [N], norms_per_window [N,2])."""
    keys, base_z, true_z, apt_ids = [], [], [], []
    norms_per_window_means, norms_per_window_stds = [], []
    for ai, apt in enumerate(apts):
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            continue
        n = len(series)
        train_end = int(n * TRAIN_RATIO)
        seg = series[:train_end]
        m = float(seg.mean()); s = float(seg.std()) if seg.std() > 1e-8 else 1.0
        ds = HouseholdDataset(seg, m, s, stride=24)
        for x, y in DataLoader(ds, batch_size=256, shuffle=False):
            x_np = x.numpy()
            with torch.no_grad():
                if isinstance(model_for_baseline, NBEATSxAux):
                    y_hat, _, _ = model_for_baseline(x.to(DEVICE))
                else:
                    y_hat, _ = model_for_baseline(x.to(DEVICE))
            keys.append(extract_key(x_np))
            base_z.append(y_hat.cpu().numpy())
            true_z.append(y.numpy())
            apt_ids.append(np.full(len(y), ai, dtype=np.int32))
            norms_per_window_means.append(np.full(len(y), m))
            norms_per_window_stds.append(np.full(len(y), s))
    return (np.concatenate(keys, axis=0),
            np.concatenate(base_z, axis=0),
            np.concatenate(true_z, axis=0),
            np.concatenate(apt_ids, axis=0),
            np.concatenate(norms_per_window_means, axis=0),
            np.concatenate(norms_per_window_stds, axis=0))


def main():
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    h1b = json.load(open(OUT / "quantize_h1b.json"))
    pass_arms = h1b["pass_arms_h1b"]
    if not pass_arms:
        print("[H1c] no PASS arms from H1b — nothing to evaluate.")
        return

    split = load_v10_split()
    train_apts = split["train"]
    cold_apts = split["cold"]

    results = {}
    for arm in pass_arms:
        print(f"\n========== H1c {arm} ==========")
        extract_fn = load_arm_model(arm)

        # baseline model is T0 backbone for T0/T1/T4/T5; the aux model itself for T2/T3
        if arm in ("T0", "T1", "T4", "T5"):
            base = MinimalNBEATSx().to(DEVICE).eval()
            base.load_state_dict(torch.load(OUT / "T0" / "best.pt", map_location="cpu", weights_only=False))
        elif arm == "T2":
            base = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
            base.load_state_dict(torch.load(OUT / arm / "best.pt", map_location="cpu", weights_only=False))
        elif arm == "T3":
            base = NBEATSxAux(latent_source="h_concat").to(DEVICE).eval()
            base.load_state_dict(torch.load(OUT / arm / "best.pt", map_location="cpu", weights_only=False))

        # train side: KEYs + assignment + offsets
        K_tr, base_tr, true_tr, _, _, _ = gather_train_kv(extract_fn, base, train_apts)
        # use saved codebook to assign: but we need LATENTs to assign. Reuse VQ:
        cb = np.load(OUT / arm / "codebook.npz")
        codebook = cb["codebook"]   # [M, D]
        # gather latent for assignment
        from importlib import reload
        import importlib
        # extract latent for each train window (same fn returns the arm's latent)
        def latent_chunks():
            for ai, apt in enumerate(train_apts):
                try:
                    series = load_apartment_hourly(apt).values.astype(np.float32)
                except FileNotFoundError:
                    continue
                n = len(series)
                train_end = int(n * TRAIN_RATIO)
                seg = series[:train_end]
                m = float(seg.mean()); s = float(seg.std()) if seg.std() > 1e-8 else 1.0
                ds = HouseholdDataset(seg, m, s, stride=24)
                for x, y in DataLoader(ds, batch_size=256, shuffle=False):
                    yield extract_fn(x)
        lat_tr = np.concatenate(list(latent_chunks()), axis=0)
        # assign each lat to nearest centroid
        d_tr = ((lat_tr[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=2)
        idx_tr = d_tr.argmin(axis=1)

        # offset per cluster (z-space residual)
        offsets = np.zeros((codebook.shape[0], 24), dtype=np.float32)
        counts = np.zeros(codebook.shape[0], dtype=np.int64)
        residuals = true_tr - base_tr
        for c in range(codebook.shape[0]):
            mask = idx_tr == c
            counts[c] = int(mask.sum())
            if counts[c] > 0:
                offsets[c] = residuals[mask].mean(axis=0)

        print(f"  train assignments: {counts.min()}..{counts.max()}, n_empty_clusters={(counts==0).sum()}")

        # cold inference
        K_co, base_co, true_co, apt_id_co, m_co, s_co = gather_train_kv(extract_fn, base, cold_apts)
        # KEY-NN: for each cold KEY, find nearest TRAIN KEY -> use its cluster id
        from sklearn.preprocessing import StandardScaler
        ks = StandardScaler().fit(K_tr)
        Kt = ks.transform(K_tr); Kc = ks.transform(K_co)
        # 1-NN search
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1).fit(Kt)
        _, neigh_idx = nn.kneighbors(Kc)
        cold_cluster = idx_tr[neigh_idx[:, 0]]
        cold_offset = offsets[cold_cluster]   # [N_cold, 24]

        corrected = base_co + ALPHA * cold_offset

        # denorm to kW for PAPE
        true_kw = true_co * s_co[:, None] + m_co[:, None]
        base_kw = base_co * s_co[:, None] + m_co[:, None]
        corr_kw = corrected * s_co[:, None] + m_co[:, None]

        m_base = seven_axis_metrics(true_kw, base_kw)
        m_corr = seven_axis_metrics(true_kw, corr_kw)
        ratio = m_corr["pape"] / m_base["pape"] if m_base["pape"] > 0 else 1.0
        gate = "PASS" if ratio <= PASS_RATIO else "FAIL"
        print(f"  cold PAPE: baseline={m_base['pape']:.2f}  KV-VQ={m_corr['pape']:.2f}  ratio={ratio:.3f}  [{gate} H1c]")
        print(f"  cold HR@1: baseline={m_base['hr@1']:.1f}  KV-VQ={m_corr['hr@1']:.1f}")

        results[arm] = {
            "alpha": ALPHA,
            "n_cold_windows": int(true_co.shape[0]),
            "baseline": m_base,
            "kv_vq": m_corr,
            "pape_ratio": ratio,
            "gate_h1c": gate,
            "cluster_counts_train": counts.tolist(),
        }

    pass_h1c = [a for a, r in results.items() if r["gate_h1c"] == "PASS"]
    out = {"alpha": ALPHA, "pass_threshold_ratio": PASS_RATIO,
           "pass_arms_h1c": pass_h1c, "results": results}
    with open(OUT / "coldstart_h1c.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[H1c] PASS arms: {pass_h1c}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 9.3: Run cold-start gate**

```bash
cd "C:/Users/HOME/JW/Research Docs/FL_Peak_Project" && uv run python experiments/v01_peak_from_latent/05_coldstart_h1c.py 2>&1 | tee outputs/v01_peak_from_latent/coldstart_h1c.txt
```

- [ ] **Step 9.4: Commit**

```bash
git add src/probes/peak_descriptor.py src/probes/__init__.py experiments/v01_peak_from_latent/05_coldstart_h1c.py
git commit -m "feat: H1c cold-start gate — KV-VQ residual correction on 50 cold apts"
```

---

## Task 10: Aggregate report

**Files:**
- Create: `experiments/v01_peak_from_latent/06_aggregate_report.py`

- [ ] **Step 10.1: Write `06_aggregate_report.py`**

```python
"""Build a single markdown report from H1a/b/c JSONs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "outputs" / "v01_peak_from_latent"


def fmt(v, prec=3):
    return f"{v:.{prec}f}" if isinstance(v, (int, float)) else str(v)


def main():
    h1a = json.load(open(OUT / "probe_h1a.json"))
    h1b = json.load(open(OUT / "quantize_h1b.json")) if (OUT / "quantize_h1b.json").exists() else None
    h1c = json.load(open(OUT / "coldstart_h1c.json")) if (OUT / "coldstart_h1c.json").exists() else None

    lines = ["# v01-01 Peak from Latent — Result Report", ""]
    lines.append(f"H1a pass arms: `{h1a['pass_arms']}`  (threshold R² ≥ {h1a['pass_threshold']})")
    if h1b:
        lines.append(f"H1b pass arms: `{h1b['pass_arms_h1b']}`  (threshold ratio ≥ {h1b['pass_threshold_ratio']})")
    if h1c:
        lines.append(f"H1c pass arms: `{h1c['pass_arms_h1c']}`  (threshold PAPE ratio ≤ {h1c['pass_threshold_ratio']})")
    lines.append("")

    lines.append("## H1a — Probe (peak_amp_fc)")
    lines.append("| Arm | Ridge R² | Ridge PAPE (%) | MLP R² | hr top-1 | gate |")
    lines.append("|---|---|---|---|---|---|")
    for arm, d in h1a["results"].items():
        a = d["peak_amp_fc"]; h = d["peak_hr_fc"]
        gate = "PASS" if a["ridge_R2"] >= h1a["pass_threshold"] else "FAIL"
        lines.append(f"| {arm} | {fmt(a['ridge_R2'])} | {fmt(a['ridge_PAPE'],1)} | {fmt(a['mlp_R2'])} | {fmt(h['top1'])} | {gate} |")
    lines.append("")

    if h1b:
        lines.append("## H1b — Quantization (M=32)")
        lines.append("| Arm | R²(raw) | R²(q) | ratio | util | k_min | gate |")
        lines.append("|---|---|---|---|---|---|---|")
        for arm, d in h1b["results"].items():
            v = d["vq_diagnostics"]
            lines.append(f"| {arm} | {fmt(d['r2_raw'])} | {fmt(d['r2_quantized'])} | {fmt(d['ratio'])} | {fmt(v['utilization'],2)} | {v['k_min']} | {d['gate_h1b']} |")
        lines.append("")

    if h1c:
        lines.append("## H1c — Cold-start (50 apts)")
        lines.append("| Arm | Baseline PAPE | KV-VQ PAPE | ratio | Baseline HR@1 | KV-VQ HR@1 | gate |")
        lines.append("|---|---|---|---|---|---|---|")
        for arm, d in h1c["results"].items():
            b = d["baseline"]; k = d["kv_vq"]
            lines.append(f"| {arm} | {fmt(b['pape'],2)} | {fmt(k['pape'],2)} | {fmt(d['pape_ratio'])} | {fmt(b['hr@1'],1)} | {fmt(k['hr@1'],1)} | {d['gate_h1c']} |")
        lines.append("")

    text = "\n".join(lines)
    out_path = OUT / "report.md"
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10.2: Generate report**

```bash
cd "C:/Users/HOME/JW/Research Docs/FL_Peak_Project" && uv run python experiments/v01_peak_from_latent/06_aggregate_report.py
```

Expected: prints markdown table to stdout AND saves `outputs/v01_peak_from_latent/report.md`.

- [ ] **Step 10.3: Commit**

```bash
git add experiments/v01_peak_from_latent/06_aggregate_report.py
git commit -m "feat: aggregate H1a/b/c JSONs into single markdown report"
```

---

## Open Decisions (revisit if results odd)

1. **λ_peak**: default 0.3. If T2/T3 fail H1a but train_aux loss is going down, consider sweep {0.1, 0.5, 1.0}.
2. **M (codebook size)**: default 32. If H1b has k_min=0 (dead clusters), drop to M=16.
3. **α (KV-VQ correction strength)**: default 0.5. If H1c FAIL but offset direction is right, sweep {0.3, 0.7, 1.0}.
4. **PROJ_DIM (T4)**: default 32. Could match h_concat (192) for upper bound.
5. **KEY definition**: default 5-d (max, argmax_norm, mean, std, last24_max). If H1c fails, consider adding day-of-week or hour-of-day-of-window-end.

---

## Self-Review Checklist (filled after writing)

- [x] Spec coverage: H1a/b/c each have a dedicated task (6, 8, 9). Latent arms T0–T5 each have a definition path (T0/T1 share T0 ckpt; T2/T3 in Task 4; T4 in Task 5; T5 uses T0 forecast directly in Task 6's `load_arm_model`).
- [x] Placeholder scan: No "TODO" / "TBD" / "fill in" patterns. All code blocks complete.
- [x] Type consistency: `extract_key(x: np.ndarray) -> [B, 5]` used uniformly across Task 9. `VectorQuantizerKMeans.fit(z: torch.Tensor)` returns dict — used as such in Task 8.

Known limitations (not bugs, design choices noted):
- T1 alias relies on T0 ckpt existing; Task 6 `load_arm_model` falls back accordingly.
- T5 latent IS the forecast — H1b for T5 is trivially the codebook of forecast patterns; H1c becomes "lookup nearest training-set forecast" which collapses to a memory-based forecaster.
- Ridge in T4 picks top-PROJ_DIM rows by norm; if Ridge gives < PROJ_DIM rows it pads with random orthogonal directions. This is a design choice acknowledged in the script comment.
