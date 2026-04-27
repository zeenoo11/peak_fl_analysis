"""E3: T2 multi-seed stability check.

Trains T2 (NBEATSxAux on h_generic, MAE + 0.3 * peak_aux) under 3 seeds:
    seed=42 (existing)
    seed=123, seed=7 (new)

Each variant is then evaluated with W5 best hyperparams (σ=3.0, α_v0=1.5, α_w1=0.5).
Reports mean ± std for cold PAPE, HR@1, HR@2.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader

from config import HORIZON, INPUT_SIZE, OUTPUT_DIR, TRAIN_RATIO, VAL_RATIO
from dataloader.splits import load_v10_split
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from models.nbeatsx_aux import NBEATSxAux
from models.peak_aux_head import peak_aux_loss
from models.vq_kmeans import VectorQuantizerKMeans
from probes.peak_descriptor import extract_key
from utils.metrics import compute_hr, compute_pape, seven_axis_metrics

OUT = OUTPUT_DIR / "v01_peak_from_latent"
E3 = OUT / "E3"
E3.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 30; BATCH = 256; LR = 1e-3; PATIENCE = 8; LAM = 0.3   # match 01_train_arms.py schedule (the seed=42/T2 reuse path)
W5 = {"sigma": 3.0, "alpha_v0": 1.5, "alpha_w1": 0.5}
SEEDS = [42, 123, 7]


def build_loaders(apts, batch):
    train_sets, val_sets, norm, present = [], [], {}, []
    for apt in apts:
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            continue
        n = len(series); train_end = int(n * TRAIN_RATIO); val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        mean = float(series[:train_end].mean())
        std = float(series[:train_end].std()) if series[:train_end].std() > 1e-8 else 1.0
        train_sets.append(HouseholdDataset(series[:train_end], mean, std, stride=1))
        val_sets.append(HouseholdDataset(series[train_end:val_end], mean, std, stride=1))
        norm[apt] = {"mean": mean, "std": std}; present.append(apt)
    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=batch, shuffle=True)
    return train_sets, val_sets, norm, train_loader, present


def train_t2_seed(seed: int, present, train_loader, val_sets, norm) -> Path:
    out_dir = E3 / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "best.pt"
    if ckpt.exists():
        print(f"  [skip] seed={seed} ckpt exists")
        return ckpt
    if seed == 42 and (OUT / "T2" / "best.pt").exists():
        # reuse existing T2 ckpt
        shutil.copy(OUT / "T2" / "best.pt", ckpt)
        print(f"  [reuse] seed=42 from T2 ckpt")
        return ckpt

    torch.manual_seed(seed); np.random.seed(seed)
    model = NBEATSxAux(latent_source="h_generic").to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    best_val_mae, best_state, bad = float("inf"), None, 0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        loss_sum, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_hat, _, (amp_p, hr_p) = model(x)
            main = F.l1_loss(y_hat, y)
            aux = peak_aux_loss(amp_p, hr_p, y)
            loss = main + LAM * aux
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss.item()); n += 1

        model.eval()
        a_idx, t_chunks, p_chunks = [], [], []
        with torch.no_grad():
            for ai, ds in enumerate(val_sets):
                for x, y in DataLoader(ds, batch_size=BATCH, shuffle=False):
                    y_hat, _, _ = model(x.to(DEVICE))
                    t_chunks.append(y.numpy()); p_chunks.append(y_hat.cpu().numpy())
                    a_idx.append(np.full(len(y), ai, dtype=np.int32))
        t_z = np.concatenate(t_chunks, 0); p_z = np.concatenate(p_chunks, 0)
        ai_arr = np.concatenate(a_idx, 0)
        means = np.array([norm[a]["mean"] for a in present])
        stds = np.array([norm[a]["std"] for a in present])
        t_kw = t_z * stds[ai_arr, None] + means[ai_arr, None]
        p_kw = p_z * stds[ai_arr, None] + means[ai_arr, None]
        val_mae = seven_axis_metrics(t_kw, p_kw)["mae"]
        improved = val_mae < best_val_mae - 1e-6
        if improved:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        flag = " *" if improved else ""
        print(f"    ep{epoch:02d} loss={loss_sum/n:.4f} val_mae={val_mae:.4f} ({time.time()-t0:.1f}s){flag}")
        if bad >= PATIENCE:
            print(f"    early stop @ ep {epoch}"); break

    torch.save(best_state, ckpt)
    return ckpt


def gather_for_eval(ckpt: Path, apts):
    m = NBEATSxAux(latent_source="h_generic").to(DEVICE).eval()
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    keys, lats, base_z, true_z, p_amp, p_hr, m_arr, s_arr = [], [], [], [], [], [], [], []
    for apt in apts:
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            continue
        n = len(series); train_end = int(n * TRAIN_RATIO)
        seg = series[:train_end]
        mean = float(seg.mean()); std = float(seg.std()) if seg.std() > 1e-8 else 1.0
        ds = HouseholdDataset(seg, mean, std, stride=24)
        for x, y in DataLoader(ds, batch_size=256, shuffle=False):
            keys.append(extract_key(x.numpy()))
            with torch.no_grad():
                y_hat, hidd, (amp_p, hr_p) = m(x.to(DEVICE))
            lats.append(hidd["h_generic"].cpu().numpy())
            base_z.append(y_hat.cpu().numpy()); true_z.append(y.numpy())
            p_amp.append(amp_p.cpu().numpy()); p_hr.append(hr_p.argmax(dim=1).cpu().numpy())
            m_arr.append(np.full(len(y), mean)); s_arr.append(np.full(len(y), std))
    return {
        "key": np.concatenate(keys, 0), "lat": np.concatenate(lats, 0),
        "base_z": np.concatenate(base_z, 0), "true_z": np.concatenate(true_z, 0),
        "pred_amp": np.concatenate(p_amp, 0), "pred_hr": np.concatenate(p_hr, 0),
        "mean": np.concatenate(m_arr, 0), "std": np.concatenate(s_arr, 0),
    }


def evaluate_W5(ckpt, train_apts, cold_apts, seed):
    tr = gather_for_eval(ckpt, train_apts)
    co = gather_for_eval(ckpt, cold_apts)

    cold_true_hr = co["true_z"].argmax(axis=1)
    aux_within1 = float((np.abs(co["pred_hr"] - cold_true_hr) <= 1).mean())

    vq = VectorQuantizerKMeans(num_embeddings=32, embedding_dim=tr["lat"].shape[1], random_state=seed)
    vq.fit(torch.from_numpy(tr["lat"]).float())
    cb = vq.codebook.cpu().numpy()
    d = ((tr["lat"][:, None, :] - cb[None, :, :]) ** 2).sum(axis=2)
    idx_tr = d.argmin(axis=1); M = cb.shape[0]
    offsets = np.zeros((M, 24), dtype=np.float32)
    for c in range(M):
        mask = idx_tr == c
        if mask.sum() > 0:
            offsets[c] = (tr["true_z"][mask] - tr["base_z"][mask]).mean(axis=0)

    ks = StandardScaler().fit(tr["key"])
    nn = NearestNeighbors(n_neighbors=1).fit(ks.transform(tr["key"]))
    _, ni = nn.kneighbors(ks.transform(co["key"]))
    cold_cluster = idx_tr[ni[:, 0]]

    sigma, av, aw = W5["sigma"], W5["alpha_v0"], W5["alpha_w1"]
    t = np.arange(24)[None, :]
    g = np.exp(-0.5 * ((t - co["pred_hr"][:, None]) / sigma) ** 2)
    g = g / g.max(axis=1, keepdims=True) * co["pred_amp"][:, None]
    corrected = co["base_z"] + av * offsets[cold_cluster] + aw * g

    true_kw = co["true_z"] * co["std"][:, None] + co["mean"][:, None]
    base_kw = co["base_z"] * co["std"][:, None] + co["mean"][:, None]
    corr_kw = corrected * co["std"][:, None] + co["mean"][:, None]
    return {
        "aux_within1": aux_within1,
        "base_pape": compute_pape(true_kw, base_kw),
        "corr_pape": compute_pape(true_kw, corr_kw),
        "base_hr@1": compute_hr(true_kw, base_kw, tol=1),
        "corr_hr@1": compute_hr(true_kw, corr_kw, tol=1),
        "base_hr@2": compute_hr(true_kw, base_kw, tol=2),
        "corr_hr@2": compute_hr(true_kw, corr_kw, tol=2),
    }


def main():
    split = load_v10_split()
    print(f"[setup] T2 multi-seed; train={len(split['train'])}, cold={len(split['cold'])}")
    train_sets, val_sets, norm, train_loader, present = build_loaders(split["train"], BATCH)
    print(f"[data] {sum(len(d) for d in train_sets)} train windows")

    results = {}
    for seed in SEEDS:
        print(f"\n========== seed={seed} ==========")
        ckpt = train_t2_seed(seed, present, train_loader, val_sets, norm)
        ev = evaluate_W5(ckpt, split["train"], split["cold"], seed)
        results[f"seed_{seed}"] = ev
        print(f"  base PAPE={ev['base_pape']:.2f}  W5 PAPE={ev['corr_pape']:.2f}  "
              f"HR@1={ev['corr_hr@1']:.1f}  HR@2={ev['corr_hr@2']:.1f}  aux_w1h={ev['aux_within1']*100:.1f}%")

    # Aggregate
    print("\n========== E3 MULTI-SEED SUMMARY ==========")
    keys = ["base_pape", "corr_pape", "base_hr@1", "corr_hr@1", "base_hr@2", "corr_hr@2", "aux_within1"]
    print(f"{'metric':18s}  seed=42  seed=123  seed=7  mean ± std")
    print("-" * 70)
    summary = {}
    for k in keys:
        vals = np.array([results[f"seed_{s}"][k] for s in SEEDS])
        mean, std = float(vals.mean()), float(vals.std())
        summary[k] = {"values": vals.tolist(), "mean": mean, "std": std}
        if k == "aux_within1":
            print(f"  {k:18s}  {vals[0]*100:6.1f}%  {vals[1]*100:7.1f}%  {vals[2]*100:6.1f}%  {mean*100:.1f}% ± {std*100:.2f}%")
        else:
            print(f"  {k:18s}  {vals[0]:6.2f}   {vals[1]:7.2f}   {vals[2]:6.2f}   {mean:.2f} ± {std:.2f}")

    out = {"per_seed": results, "summary": summary, "seeds": SEEDS, "W5_best": W5}
    with open(E3 / "E3_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[done] wrote {E3 / 'E3_results.json'}")


if __name__ == "__main__":
    main()
