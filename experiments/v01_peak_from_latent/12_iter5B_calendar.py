"""iter5-B: train NBEATSx with calendar features, then evaluate W5 hybrid.

3 variants:
    B1  hour-of-day only (n_cal=2)
    B2  hour + day-of-week (n_cal=4)
    B3  same as B2 but stronger hr_weight (5.0) — combines with iter5-A finding
"""

from __future__ import annotations

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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader

from config import HORIZON, INPUT_SIZE, OUTPUT_DIR, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO
from dataloader.splits import load_v10_split
from dataloader.umass import load_apartment_hourly
from dataloader.umass_calendar import HouseholdDatasetCal
from models.nbeatsx_calendar import NBEATSxAuxCal
from models.peak_aux_head import peak_aux_loss
from models.vq_kmeans import VectorQuantizerKMeans
from probes.peak_descriptor import extract_key
from utils.metrics import compute_hr, compute_pape, seven_axis_metrics

OUT = OUTPUT_DIR / "v01_peak_from_latent"
ITER5B = OUT / "iter5_B"
ITER5B.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 15; BATCH = 256; LR = 1e-3; PATIENCE = 5
W5_BEST = {"sigma": 3.0, "alpha_v0": 1.5, "alpha_w1": 0.5}


def custom_aux_loss(amp_pred, hr_pred, y, hr_weight: float):
    amp_true = y.max(dim=1).values
    hr_true = y.argmax(dim=1)
    return F.mse_loss(amp_pred, amp_true) + hr_weight * F.cross_entropy(hr_pred, hr_true)


def build_loaders(apts, batch, use_dow):
    train_sets, val_sets, norm, present = [], [], {}, []
    for apt in apts:
        try:
            series_pd = load_apartment_hourly(apt)
        except FileNotFoundError:
            continue
        values = series_pd.values.astype(np.float32)
        n = len(values)
        train_end = int(n * TRAIN_RATIO); val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        mean = float(values[:train_end].mean())
        std = float(values[:train_end].std()) if values[:train_end].std() > 1e-8 else 1.0
        train_sets.append(HouseholdDatasetCal(series_pd.iloc[:train_end], mean, std,
                                               stride=1, use_dow=use_dow))
        val_sets.append(HouseholdDatasetCal(series_pd.iloc[train_end:val_end], mean, std,
                                             stride=1, use_dow=use_dow))
        norm[apt] = {"mean": mean, "std": std}
        present.append(apt)
    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=batch, shuffle=True)
    return train_sets, val_sets, norm, train_loader, present


def train_one(use_dow, hr_w, lam, tag, present, train_loader, val_sets, norm):
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    n_cal = 4 if use_dow else 2
    model = NBEATSxAuxCal(n_cal=n_cal).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    best_val_mae, best_state, bad = float("inf"), None, 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        loss_sum, aux_sum, n = 0.0, 0.0, 0
        for x, y, cal in train_loader:
            x, y, cal = x.to(DEVICE), y.to(DEVICE), cal.to(DEVICE)
            y_hat, _, (amp_p, hr_p) = model(x, cal)
            main = F.l1_loss(y_hat, y)
            aux = custom_aux_loss(amp_p, hr_p, y, hr_weight=hr_w)
            loss = main + lam * aux
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss.item()); aux_sum += float(aux.item()); n += 1

        # val
        model.eval()
        apt_idx_arr, t_chunks, p_chunks = [], [], []
        with torch.no_grad():
            for ai, ds in enumerate(val_sets):
                for x, y, cal in DataLoader(ds, batch_size=BATCH, shuffle=False):
                    y_hat, _, _ = model(x.to(DEVICE), cal.to(DEVICE))
                    t_chunks.append(y.numpy()); p_chunks.append(y_hat.cpu().numpy())
                    apt_idx_arr.append(np.full(len(y), ai, dtype=np.int32))
        t_z = np.concatenate(t_chunks, axis=0); p_z = np.concatenate(p_chunks, axis=0)
        a_idx = np.concatenate(apt_idx_arr, axis=0)
        means = np.array([norm[a]["mean"] for a in present])
        stds = np.array([norm[a]["std"] for a in present])
        t_kw = t_z * stds[a_idx, None] + means[a_idx, None]
        p_kw = p_z * stds[a_idx, None] + means[a_idx, None]
        m = seven_axis_metrics(t_kw, p_kw)
        rec = {"epoch": epoch, "train_loss": loss_sum/n, "train_aux": aux_sum/n,
               "val_mae": m["mae"], "val_pape": m["pape"], "val_hr@1": m["hr@1"],
               "wall_s": round(time.time()-t0, 1)}
        history.append(rec)
        improved = m["mae"] < best_val_mae - 1e-6
        if improved:
            best_val_mae = m["mae"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        flag = " *" if improved else ""
        print(f"  ep{epoch:02d} loss={rec['train_loss']:.4f} aux={rec['train_aux']:.4f} "
              f"val_mae={rec['val_mae']:.4f} val_pape={rec['val_pape']:.2f} "
              f"hr1={rec['val_hr@1']:.1f} ({rec['wall_s']}s){flag}")
        if bad >= PATIENCE:
            print(f"  early stop @ ep {epoch}"); break

    out_dir = ITER5B / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "best.pt")
    with open(out_dir / "training_log.json", "w") as fh:
        json.dump({"use_dow": use_dow, "hr_weight": hr_w, "lam": lam,
                   "history": history, "best_val_mae": best_val_mae}, fh, indent=2)
    return out_dir / "best.pt"


def gather_for_eval(ckpt_path, apts, use_dow):
    n_cal = 4 if use_dow else 2
    m = NBEATSxAuxCal(n_cal=n_cal).to(DEVICE).eval()
    m.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    keys, lats, base_z, true_z, p_amp, p_hr, m_arr, s_arr = [], [], [], [], [], [], [], []
    for apt in apts:
        try:
            series_pd = load_apartment_hourly(apt)
        except FileNotFoundError:
            continue
        n_ts = len(series_pd); train_end = int(n_ts * TRAIN_RATIO)
        seg_pd = series_pd.iloc[:train_end]
        seg_vals = seg_pd.values.astype(np.float32)
        mean = float(seg_vals.mean())
        std = float(seg_vals.std()) if seg_vals.std() > 1e-8 else 1.0
        ds = HouseholdDatasetCal(seg_pd, mean, std, stride=24, use_dow=use_dow)
        for x, y, cal in DataLoader(ds, batch_size=256, shuffle=False):
            keys.append(extract_key(x.numpy()))
            with torch.no_grad():
                y_hat, hidd, (amp_pp, hr_pp) = m(x.to(DEVICE), cal.to(DEVICE))
            lats.append(hidd["h_generic"].cpu().numpy())
            base_z.append(y_hat.cpu().numpy()); true_z.append(y.numpy())
            p_amp.append(amp_pp.cpu().numpy()); p_hr.append(hr_pp.argmax(dim=1).cpu().numpy())
            m_arr.append(np.full(len(y), mean)); s_arr.append(np.full(len(y), std))
    return {
        "key": np.concatenate(keys, axis=0), "lat": np.concatenate(lats, axis=0),
        "base_z": np.concatenate(base_z, axis=0), "true_z": np.concatenate(true_z, axis=0),
        "pred_amp": np.concatenate(p_amp, axis=0), "pred_hr": np.concatenate(p_hr, axis=0),
        "mean": np.concatenate(m_arr, axis=0), "std": np.concatenate(s_arr, axis=0),
    }


def evaluate_W5(ckpt_path, train_apts, cold_apts, use_dow):
    tr = gather_for_eval(ckpt_path, train_apts, use_dow)
    co = gather_for_eval(ckpt_path, cold_apts, use_dow)

    cold_true_hr = co["true_z"].argmax(axis=1)
    aux_top1 = float((co["pred_hr"] == cold_true_hr).mean())
    aux_within1 = float((np.abs(co["pred_hr"] - cold_true_hr) <= 1).mean())
    aux_within2 = float((np.abs(co["pred_hr"] - cold_true_hr) <= 2).mean())

    vq = VectorQuantizerKMeans(num_embeddings=32, embedding_dim=tr["lat"].shape[1], random_state=RANDOM_SEED)
    vq.fit(torch.from_numpy(tr["lat"]).float())
    cb = vq.codebook.cpu().numpy()
    d = ((tr["lat"][:, None, :] - cb[None, :, :]) ** 2).sum(axis=2)
    idx_tr = d.argmin(axis=1); M = cb.shape[0]
    offsets_v0 = np.zeros((M, 24), dtype=np.float32)
    for c in range(M):
        mask = idx_tr == c
        if mask.sum() > 0:
            offsets_v0[c] = (tr["true_z"][mask] - tr["base_z"][mask]).mean(axis=0)

    ks = StandardScaler().fit(tr["key"])
    nn = NearestNeighbors(n_neighbors=1).fit(ks.transform(tr["key"]))
    _, ni = nn.kneighbors(ks.transform(co["key"]))
    cold_cluster = idx_tr[ni[:, 0]]

    sigma = W5_BEST["sigma"]; a_v = W5_BEST["alpha_v0"]; a_w = W5_BEST["alpha_w1"]
    t = np.arange(24)[None, :]
    g = np.exp(-0.5 * ((t - co["pred_hr"][:, None]) / sigma) ** 2)
    g = g / g.max(axis=1, keepdims=True) * co["pred_amp"][:, None]
    corrected = co["base_z"] + a_v * offsets_v0[cold_cluster] + a_w * g

    true_kw = co["true_z"] * co["std"][:, None] + co["mean"][:, None]
    base_kw = co["base_z"] * co["std"][:, None] + co["mean"][:, None]
    corr_kw = corrected * co["std"][:, None] + co["mean"][:, None]
    return {
        "aux_top1": aux_top1, "aux_within1": aux_within1, "aux_within2": aux_within2,
        "base_pape": compute_pape(true_kw, base_kw),
        "corr_pape": compute_pape(true_kw, corr_kw),
        "base_hr@1": compute_hr(true_kw, base_kw, tol=1),
        "corr_hr@1": compute_hr(true_kw, corr_kw, tol=1),
        "base_hr@2": compute_hr(true_kw, base_kw, tol=2),
        "corr_hr@2": compute_hr(true_kw, corr_kw, tol=2),
    }


def main():
    split = load_v10_split()
    print(f"[setup] B variants on {len(split['train'])} train apts")

    settings = [
        # (use_dow, hr_w, lam, tag)
        (False, 0.1, 0.3, "B1_hour_only"),
        (True, 0.1, 0.3, "B2_hour_dow"),
        (True, 5.0, 0.3, "B3_hour_dow_hrw5"),
    ]

    results = {}
    for use_dow, hr_w, lam, tag in settings:
        print(f"\n========== {tag} (use_dow={use_dow}, hr_w={hr_w}, lam={lam}) ==========")
        train_sets, val_sets, norm, train_loader, present = build_loaders(
            split["train"], BATCH, use_dow)
        ckpt = train_one(use_dow, hr_w, lam, tag, present, train_loader, val_sets, norm)
        ev = evaluate_W5(ckpt, split["train"], split["cold"], use_dow)
        results[tag] = {"use_dow": use_dow, "hr_weight": hr_w, "lam": lam, **ev}
        print(f"  [eval] aux_within1={ev['aux_within1']*100:.1f}%  "
              f"base PAPE={ev['base_pape']:.2f} HR@1={ev['base_hr@1']:.1f}  "
              f"W5 PAPE={ev['corr_pape']:.2f} HR@1={ev['corr_hr@1']:.1f} HR@2={ev['corr_hr@2']:.1f}")

    print("\n========== SUMMARY ==========")
    print(f"{'tag':25s}  base_PAPE base_HR1  aux_w1h%  W5_PAPE  W5_HR1  W5_HR2")
    print("-" * 90)
    for tag, r in results.items():
        print(f"{tag:25s}  {r['base_pape']:7.2f}  {r['base_hr@1']:6.1f}  "
              f"{r['aux_within1']*100:7.1f}  {r['corr_pape']:7.2f}  "
              f"{r['corr_hr@1']:5.1f}  {r['corr_hr@2']:5.1f}")

    with open(ITER5B / "iter5B_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[done] wrote {ITER5B / 'iter5B_results.json'}")


if __name__ == "__main__":
    main()
