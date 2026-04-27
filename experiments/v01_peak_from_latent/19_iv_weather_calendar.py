"""(iv) External information experiment: calendar + weather features.

Trains NBEATSxAuxCal with n_cal=8 (4 calendar + 4 weather) and peak_aux head.
Then evaluates W5 hybrid on cold households. Goal: test whether external
information (temperature, humidity, etc.) breaks the HR@1 ceiling that
peak_aux + calendar alone could not.
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
from dataloader.umass_weather import (
    HouseholdDatasetCalWeather,
    load_weather_with_stats,
    N_WEATHER,
)
from models.nbeatsx_calendar import NBEATSxAuxCal
from models.peak_aux_head import peak_aux_loss
from models.vq_kmeans import VectorQuantizerKMeans
from probes.peak_descriptor import extract_key
from utils.metrics import compute_hr, compute_pape, seven_axis_metrics

OUT = OUTPUT_DIR / "v01_peak_from_latent"
IV = OUT / "iv_weather"
IV.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 15; BATCH = 256; LR = 1e-3; PATIENCE = 5; LAM = 0.3
W5 = {"sigma": 3.0, "alpha_v0": 1.5, "alpha_w1": 0.5}
N_CAL = 4 + N_WEATHER  # 4 calendar + 4 weather


def build_loaders(apts, batch, weather_df, w_mean, w_std):
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
        train_sets.append(HouseholdDatasetCalWeather(
            series_pd.iloc[:train_end], mean, std, weather_df, w_mean, w_std, stride=1))
        val_sets.append(HouseholdDatasetCalWeather(
            series_pd.iloc[train_end:val_end], mean, std, weather_df, w_mean, w_std, stride=1))
        norm[apt] = {"mean": mean, "std": std}
        present.append(apt)
    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=batch, shuffle=True)
    return train_sets, val_sets, norm, train_loader, present


def train(present, train_loader, val_sets, norm):
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    model = NBEATSxAuxCal(n_cal=N_CAL).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    best_val_mae, best_state, bad = float("inf"), None, 0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        loss_sum, n = 0.0, 0
        for x, y, cal in train_loader:
            x, y, cal = x.to(DEVICE), y.to(DEVICE), cal.to(DEVICE)
            y_hat, _, (amp_p, hr_p) = model(x, cal)
            main = F.l1_loss(y_hat, y)
            aux = peak_aux_loss(amp_p, hr_p, y)
            loss = main + LAM * aux
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss.item()); n += 1

        model.eval()
        a_idx, t_chunks, p_chunks = [], [], []
        with torch.no_grad():
            for ai, ds in enumerate(val_sets):
                for x, y, cal in DataLoader(ds, batch_size=BATCH, shuffle=False):
                    y_hat, _, _ = model(x.to(DEVICE), cal.to(DEVICE))
                    t_chunks.append(y.numpy()); p_chunks.append(y_hat.cpu().numpy())
                    a_idx.append(np.full(len(y), ai, dtype=np.int32))
        t_z = np.concatenate(t_chunks, 0); p_z = np.concatenate(p_chunks, 0)
        ai_arr = np.concatenate(a_idx, 0)
        means = np.array([norm[a]["mean"] for a in present])
        stds = np.array([norm[a]["std"] for a in present])
        t_kw = t_z * stds[ai_arr, None] + means[ai_arr, None]
        p_kw = p_z * stds[ai_arr, None] + means[ai_arr, None]
        m = seven_axis_metrics(t_kw, p_kw)
        improved = m["mae"] < best_val_mae - 1e-6
        if improved:
            best_val_mae = m["mae"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        flag = " *" if improved else ""
        print(f"  ep{epoch:02d} loss={loss_sum/n:.4f} val_mae={m['mae']:.4f} val_pape={m['pape']:.2f} "
              f"hr1={m['hr@1']:.1f} ({time.time()-t0:.1f}s){flag}")
        if bad >= PATIENCE:
            print(f"  early stop @ ep {epoch}"); break

    ckpt = IV / "best.pt"
    torch.save(best_state, ckpt)
    return ckpt


def gather(ckpt, apts, weather_df, w_mean, w_std):
    m = NBEATSxAuxCal(n_cal=N_CAL).to(DEVICE).eval()
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    keys, lats, base_z, true_z, p_amp, p_hr, m_arr, s_arr = [], [], [], [], [], [], [], []
    for apt in apts:
        try:
            series_pd = load_apartment_hourly(apt)
        except FileNotFoundError:
            continue
        n = len(series_pd); train_end = int(n * TRAIN_RATIO)
        seg = series_pd.iloc[:train_end]
        seg_vals = seg.values.astype(np.float32)
        mean = float(seg_vals.mean()); std = float(seg_vals.std()) if seg_vals.std() > 1e-8 else 1.0
        ds = HouseholdDatasetCalWeather(seg, mean, std, weather_df, w_mean, w_std, stride=24)
        for x, y, cal in DataLoader(ds, batch_size=256, shuffle=False):
            keys.append(extract_key(x.numpy()))
            with torch.no_grad():
                y_hat, hidd, (amp_p, hr_p) = m(x.to(DEVICE), cal.to(DEVICE))
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


def evaluate_W5(ckpt, train_apts, cold_apts, weather_df, w_mean, w_std):
    tr = gather(ckpt, train_apts, weather_df, w_mean, w_std)
    co = gather(ckpt, cold_apts, weather_df, w_mean, w_std)

    cold_true_hr = co["true_z"].argmax(axis=1)
    aux_within1 = float((np.abs(co["pred_hr"] - cold_true_hr) <= 1).mean())

    vq = VectorQuantizerKMeans(num_embeddings=32, embedding_dim=tr["lat"].shape[1], random_state=RANDOM_SEED)
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
    print(f"[setup] calendar+weather (n_cal={N_CAL}); train={len(split['train'])}, cold={len(split['cold'])}")
    print(f"[weather] loading 2016 hourly...")
    weather_df, w_mean, w_std = load_weather_with_stats("2016")
    print(f"  weather columns: {weather_df.columns.tolist()}")
    print(f"  weather mean: {w_mean}, std: {w_std}")

    print("\n========== build loaders ==========")
    train_sets, val_sets, norm, train_loader, present = build_loaders(
        split["train"], BATCH, weather_df, w_mean, w_std)
    print(f"[data] {sum(len(d) for d in train_sets)} train windows")

    print("\n========== train W (calendar+weather+peak_aux) ==========")
    ckpt = train(present, train_loader, val_sets, norm)

    print("\n========== W5 cold eval ==========")
    ev = evaluate_W5(ckpt, split["train"], split["cold"], weather_df, w_mean, w_std)
    print(f"  base PAPE={ev['base_pape']:.2f}  W5 PAPE={ev['corr_pape']:.2f}  "
          f"HR@1={ev['corr_hr@1']:.1f}  HR@2={ev['corr_hr@2']:.1f}  "
          f"aux_w1h={ev['aux_within1']*100:.1f}%")

    # comparison anchor
    print("\n========== COMPARISON ==========")
    print(f"  baseline NBEATSx no codebook:        PAPE 55.17  HR@1 27.0  HR@2 38.5")
    print(f"  T2 (peak_aux) + W5 cold:             PAPE 37.05  HR@1 26.5  HR@2 38.2")
    print(f"  B1 (cal hour) + W5 cold:             PAPE 36.87  HR@1 26.0  HR@2 37.9")
    print(f"  W (cal+weather+peak_aux) + W5 cold:  PAPE {ev['corr_pape']:.2f}  "
          f"HR@1 {ev['corr_hr@1']:.1f}  HR@2 {ev['corr_hr@2']:.1f}")

    with open(IV / "iv_results.json", "w") as fh:
        json.dump({"n_cal": N_CAL, "weather_cols": list(weather_df.columns), "result": ev}, fh, indent=2)
    print(f"\n[done] wrote {IV / 'iv_results.json'}")


if __name__ == "__main__":
    main()
