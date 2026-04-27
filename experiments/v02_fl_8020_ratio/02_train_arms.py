"""Train arms T0/T2 on the v02 80-train-apt split (per-seed).

Adapter over experiments/v01_peak_from_latent/01_train_arms.py:
    - reads the v02 stratified split via load_v02_split(seed) instead of v10's;
    - writes to outputs/v02_fl_8020_ratio/seed{seed}/{T0,T2}/best.pt;
    - drops T3 (h_concat) — v02 scope is T0 and T2 only (plans/v02-01_fl_8020_ratio.md).

T0: MinimalNBEATSx with pure MAE.        — peak_aux OFF (E1 baseline)
T2: NBEATSxAux(latent_source='h_generic') with MAE + lambda * peak_aux.

Per-seed invocation:
    uv run python experiments/v02_fl_8020_ratio/02_train_arms.py --seed 42
    uv run python experiments/v02_fl_8020_ratio/02_train_arms.py --seed 123
    uv run python experiments/v02_fl_8020_ratio/02_train_arms.py --seed 7

[한글 요약]
v02 (FL-aligned 80:20 zero-shot, FedHiP framing) 의 백본 학습 스크립트.
v01 의 동급 스크립트(`01_train_arms.py`)를 80:20 split 입력으로 어댑트한 것이며
모델/손실/하이퍼파라미터는 v01 과 동일하게 유지한다.

- 한 번의 호출은 **단일 seed** 만 학습한다. {42, 123, 7} 멀티시드 스윕은
  스크립트 내부 루프가 아니라 외부 launcher (or 사용자) 가 `--seed S` 를
  바꿔가며 3번 호출하는 방식이다 (프로젝트 컨벤션).
- T0 = peak_aux OFF, 순수 MAE 학습. (E1 ablation 의 baseline 역할)
- T2 = peak_aux ON, 손실 = MAE(y) + λ · peak_aux(y), λ=0.3, hr_weight=0.1.
  latent source 는 `h_generic` (64-d). v01 의 T3 (h_concat, 192-d) 는 v02
  스코프에서 제외.
- v02 framing 상 백본은 "centrally pretrained, frozen shared encoder"
  로 간주된다. 이 02 단계는 *그 frozen encoder 를 만들어내는 1회성
  중앙 사전학습* 에 해당하며, 03/04 단계 이후로 가중치는 재학습되지 않는다.
- 출력: `outputs/v02_fl_8020_ratio/seed{S}/{T0,T2}/best.pt` + `training_log.json`.
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

from config import OUTPUT_DIR, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO
from dataloader.splits import load_v02_split
from dataloader.umass import HouseholdDataset, load_apartment_hourly
from models.nbeatsx import MinimalNBEATSx
from models.nbeatsx_aux import NBEATSxAux
from models.peak_aux_head import peak_aux_loss
from utils.metrics import seven_axis_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V02_OUT_ROOT = OUTPUT_DIR / "v02_fl_8020_ratio"


def build_loaders(apts: list[str], batch: int):
    """주어진 가구(아파트) 리스트로 train/val 윈도우 데이터셋과 loader 를 만든다.

    각 가구별로:
      1. `load_apartment_hourly(apt)` 로 2016년 시간단위 kW 시계열 로드.
         파일이 없으면 해당 가구는 조용히 스킵한다 (실패 시 학습은 계속 진행).
      2. 시계열 길이를 (TRAIN_RATIO=0.7) / (TRAIN+VAL=0.8) 비율로 잘라
         train / val / test 구간을 정한다 (test 구간은 본 스크립트에서는 미사용).
      3. **z-norm 통계는 train 구간에서만 계산**한다. std<1e-8 인 경우에는
         1.0 으로 폴백 (정보가 없는 시계열에서 0-나누기 방지).
         이 통계는 추후 cold inference / kW 단위 환산에서 동일하게 재사용되도록
         `norm` 딕셔너리에 보관된다.
      4. `HouseholdDataset(series, mean, std, stride=1)` — 96-step input,
         24-step horizon 슬라이딩 윈도우 (stride=1, 즉 매 시각마다 하나의 윈도우).

    여러 가구의 train 셋을 `ConcatDataset` 으로 합쳐 단일 배치 분포에서
    섞어 학습한다 (pooled training; per-apt 정규화 후 합치므로 스케일 차이는
    이미 제거된 상태).

    Returns:
        train_sets:    가구별 train 데이터셋 리스트 (윈도우 수 집계용으로 보존).
        val_sets:      가구별 val 데이터셋 리스트 (per-apt 평가에서 사용).
        norm:          {apt: {'mean','std'}} — z-norm 환산용.
        train_loader:  pooled train DataLoader (shuffle=True).
        present_apts:  실제로 로드 성공한 가구 이름 리스트 (스킵된 것 제외).
    """
    train_sets, val_sets, norm, present_apts = [], [], {}, []
    for apt in apts:
        try:
            series = load_apartment_hourly(apt).values.astype(np.float32)
        except FileNotFoundError:
            # 데이터 파일이 없으면 학습을 멈추지 않고 그 가구만 건너뛴다.
            print(f"  [skip] {apt}: missing")
            continue
        n = len(series)
        # train / val / test 구간 경계 계산 (config.TRAIN_RATIO=0.7, VAL_RATIO=0.1).
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        # per-apt z-norm 통계는 **train 구간** 에서만 계산 (data leakage 방지).
        mean = float(series[:train_end].mean())
        std = float(series[:train_end].std()) if series[:train_end].std() > 1e-8 else 1.0
        train_sets.append(HouseholdDataset(series[:train_end], mean, std, stride=1))
        val_sets.append(HouseholdDataset(series[train_end:val_end], mean, std, stride=1))
        norm[apt] = {"mean": mean, "std": std}
        present_apts.append(apt)
    # 가구별 train 셋을 모두 합쳐 한 배치 안에서 섞는 pooled training.
    train_loader = DataLoader(
        ConcatDataset(train_sets), batch_size=batch, shuffle=True, drop_last=False
    )
    return train_sets, val_sets, norm, train_loader, present_apts


def eval_per_apt(model, val_sets, present_apts, norm, batch, use_aux):
    """Per-apt val eval. Returns 7-axis metrics in kW units.

    [한글 설명]
    검증(validation) 구간에 대해 가구별 윈도우를 모두 forward 시키고,
    z-norm 된 예측을 **kW 단위로 역정규화** 한 뒤 7-axis 메트릭
    (MAE, MSE, PAPE, HR@1/2/3 등) 을 계산한다.

    핵심 포인트:
      - `use_aux=True` (T2) 인 경우 모델 forward 는 (y_hat, hiddens, (amp,hr)) 세
        값을 리턴하므로 분기해서 y_hat 만 취한다. T0 는 (y_hat, hiddens) 둘만 리턴.
      - 각 윈도우마다 어느 가구에서 왔는지 `apt_idx` 로 기록 → 가구별
        (mean, std) 로 다시 곱·합해 kW 단위로 환산한다 (pooled forward, per-apt
        denormalization). 이 환산이 정확해야 PAPE/HR@k 가 v01 과 비교 가능하다.
      - 메트릭 정의는 `src/utils/metrics.py:seven_axis_metrics` (v01 과 bit-exact
        동일; 절대 손대지 말 것).
    """
    model.eval()
    apt_idx_arr, true_chunks, pred_chunks = [], [], []
    with torch.no_grad():
        for ai, ds in enumerate(val_sets):
            for x, y in DataLoader(ds, batch_size=batch, shuffle=False):
                if use_aux:
                    # T2: NBEATSxAux 는 (y_hat, hiddens, (amp_pred, hr_pred)) 반환.
                    y_hat, _, _ = model(x.to(DEVICE))
                else:
                    # T0: MinimalNBEATSx 는 (y_hat, hiddens) 반환.
                    y_hat, _ = model(x.to(DEVICE))
                true_chunks.append(y.numpy())
                pred_chunks.append(y_hat.cpu().numpy())
                apt_idx_arr.append(np.full(len(y), ai, dtype=np.int32))
    t_z = np.concatenate(true_chunks, axis=0)
    p_z = np.concatenate(pred_chunks, axis=0)
    a_idx = np.concatenate(apt_idx_arr, axis=0)
    # 가구별 (mean, std) 를 windowed batch 에 broadcasting 해서 kW 단위로 환산.
    means = np.array([norm[a]["mean"] for a in present_apts])
    stds = np.array([norm[a]["std"] for a in present_apts])
    t_kw = t_z * stds[a_idx, None] + means[a_idx, None]
    p_kw = p_z * stds[a_idx, None] + means[a_idx, None]
    return seven_axis_metrics(t_kw, p_kw)


def train_arm(
    arm: str,
    apts: list[str],
    epochs: int,
    lr: float,
    batch: int,
    patience: int,
    lam: float,
    seed: int,
    out_root: Path,
) -> None:
    """단일 arm (T0 또는 T2) 을 한 시드로 학습하고 best checkpoint 를 저장한다.

    [동작 개요]
      - T0: `MinimalNBEATSx`. 손실 = MAE(y_hat, y) 만 사용 (peak_aux 없음).
            E1 ablation 의 baseline 역할.
      - T2: `NBEATSxAux(latent_source='h_generic')`. backbone 은 동일하지만
            `h_generic` (64-d) 위에 32-d hidden 을 거쳐 amp(scalar) +
            hr(24-class) 를 예측하는 peak-aux head 가 붙는다.
            손실 = MAE(y_hat, y) + λ · peak_aux_loss(amp, hr, y)
                 = MAE + λ · ( MSE(amp_pred, y.max) + hr_weight · CE(hr_pred, y.argmax) )
            기본 λ=0.3, hr_weight=0.1 (peak_aux_head 모듈 내 default).

    [학습 루프]
      1. seed 설정 (torch + numpy) → 동일 시드에서 reproducible.
      2. `build_loaders` 로 80개 train apt 의 pooled train DataLoader 와
         per-apt val 셋, z-norm 통계를 준비.
      3. Optimizer = Adam(lr=1e-3 default, weight_decay=1e-5). v01 과 동일.
      4. 매 epoch:
           - train: pooled batch 단위 forward → loss → backward → step.
           - eval : `eval_per_apt` 로 kW 단위 7-axis 메트릭 계산.
           - log  : (epoch, train_loss, val_mae, val_pape, val_hr@1, wall_s)
                    및 T2 의 경우 train_aux 를 history 에 append.
      5. **early-stopping & best 추적**: val_mae 가 1e-6 이상 줄어들면
         improvement → bad counter 리셋, best_state 갱신.
         그렇지 않으면 bad += 1; bad >= patience(=8 default) 시 조기 종료.
      6. 학습 종료 후:
           - `best.pt`            : 가장 낮은 val_mae 시점의 state_dict (CPU 사본).
           - `training_log.json`  : arm/seed/lam/norm/history/best_val_* 기록.
         두 파일 모두 `out_root / arm / ...` 아래에 저장된다 (예:
         `outputs/v02_fl_8020_ratio/seed42/T2/best.pt`).

    [v02 주의사항]
      - `arm` 은 'T0' 또는 'T2' 만 허용. v01 의 T3 (h_concat, 192-d) 는
        v02 스코프에서 제외 — `ValueError` 로 거부한다.
      - v02 framing 상 backbone 은 "frozen shared encoder" 이지만, 본
        02 단계는 그 frozen 가중치를 *생성* 하는 1회성 중앙 사전학습이므로
        여기서 학습이 일어나는 것은 정상이다. 이후 03/04 단계에서는
        가중치가 절대 갱신되지 않는다 (모순 아님).
    """
    # 시드는 모델 init/Adam 첫 sample/DataLoader shuffle 에 모두 영향을 준다.
    torch.manual_seed(seed); np.random.seed(seed)
    out_dir = out_root / arm
    out_dir.mkdir(parents=True, exist_ok=True)

    train_sets, val_sets, norm, train_loader, present = build_loaders(apts, batch)
    n_train = sum(len(d) for d in train_sets)
    print(f"[{arm}] {len(present)} apts, {n_train} train windows")

    # T2 만 peak_aux head 를 가지므로 forward/backward 분기에서 사용.
    use_aux = arm == "T2"
    if arm == "T2":
        # NBEATSxAux: backbone(MinimalNBEATSx) + PeakAuxHead(in_dim=64, hidden=32, n_hours=24).
        model = NBEATSxAux(latent_source="h_generic").to(DEVICE)
    elif arm == "T0":
        # 순수 NBEATSx (peak_aux 없음). 동일 backbone 키 → state_dict 호환.
        model = MinimalNBEATSx().to(DEVICE)
    else:
        # v02 는 T0/T2 만 지원 (T3 제외). 잘못된 arm 은 즉시 에러.
        raise ValueError(f"unsupported arm for v02: {arm} (expected T0 or T2)")
    # Adam + 약한 L2 (weight_decay=1e-5). v01 원본과 동일 설정.
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # best 추적용 상태. bad = "improvement 없는 epoch 의 누적 카운트" → patience 와 비교.
    best_val_mae, best_val_pape, best_state, bad, history = float("inf"), float("inf"), None, 0, []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, aux_sum, n = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if use_aux:
                # T2 forward: (y_hat[B,24], hiddens dict, (amp_pred[B], hr_pred[B,24])).
                y_hat, _, (amp_p, hr_p) = model(x)
                main = F.l1_loss(y_hat, y)                 # MAE on z-normalized horizon
                aux = peak_aux_loss(amp_p, hr_p, y)        # MSE(amp) + 0.1·CE(hr)
                loss = main + lam * aux                    # 총 손실 = MAE + λ · peak_aux
                aux_sum += float(aux.item())
            else:
                # T0 forward: (y_hat, hiddens). peak_aux 없음 → 순수 MAE.
                y_hat, _ = model(x)
                loss = F.l1_loss(y_hat, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss.item()); n += 1

        # Validation: per-apt forward + kW 역정규화 + 7-axis 메트릭.
        m = eval_per_apt(model, val_sets, present, norm, batch, use_aux)
        rec = {
            "epoch": epoch,
            "train_loss": loss_sum / n,
            "val_mae": m["mae"],
            "val_pape": m["pape"],
            "val_hr@1": m["hr@1"],
            "wall_s": round(time.time() - t0, 1),
        }
        if use_aux:
            rec["train_aux"] = aux_sum / n
        history.append(rec)

        # val_mae 가 1e-6 이상 개선되었을 때만 best 갱신 (수치 노이즈 무시).
        improved = m["mae"] < best_val_mae - 1e-6
        if improved:
            best_val_mae = m["mae"]
            best_val_pape = m["pape"]
            # state_dict 의 CPU 사본을 보관 → GPU 메모리 확보 + 저장 직전 추가 작업 불필요.
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        flag = " *" if improved else ""
        msg = (
            f"  ep{epoch:02d} loss={rec['train_loss']:.4f} "
            f"val_mae={rec['val_mae']:.4f} val_pape={rec['val_pape']:.2f} "
            f"hr1={rec['val_hr@1']:.1f} ({rec['wall_s']}s){flag}"
        )
        if use_aux:
            msg += f"  aux={rec['train_aux']:.4f}"
        print(msg)
        # patience 만큼 연속으로 개선 실패하면 조기 종료 (overfitting/시간 낭비 방지).
        if bad >= patience:
            print(f"  early stop @ ep {epoch}")
            break

    # ---- 학습 종료 후 산출물 저장 ----
    # best.pt: state_dict 만 저장. 03/04 단계에서 strict=True 로 로드 가능.
    torch.save(best_state, out_dir / "best.pt")
    # training_log.json: 후속 분석/aggregator 가 읽는 구조화 로그.
    #   - norm: per-apt mean/std → 03 codebook fit / 04 cold inference 에서 동일 통계 재사용 가능.
    #   - history: epoch 별 loss/메트릭 → 학습 곡선 그릴 때 사용.
    #   - split_version="v02": v01 과 구분하기 위한 태그.
    with open(out_dir / "training_log.json", "w") as fh:
        json.dump(
            {
                "arm": arm,
                "seed": seed,
                "split_version": "v02",
                "lam": lam,
                "norm": norm,
                "history": history,
                "n_train_windows": n_train,
                "n_apts": len(present),
                "best_val_mae": best_val_mae,
                "best_val_pape": best_val_pape,
            },
            fh,
            indent=2,
        )
    print(f"[{arm}] saved best.pt; best_val_mae={best_val_mae:.4f} best_val_pape={best_val_pape:.2f}")


def main() -> None:
    """단일-시드 entrypoint.

    [컨벤션 — 매우 중요]
      이 스크립트는 한 호출당 **단일 seed** 만 처리한다. 멀티시드 스윕은
      스크립트 내부에서 for-loop 으로 돌리지 않고, 외부 launcher (또는
      사용자가 직접) `--seed 42`, `--seed 123`, `--seed 7` 으로 3번 호출하는
      구조를 사용한다 (프로젝트 메모리: feedback_argparse_per_seed).

    [argparse 인자]
      --seed     : 80:20 split YAML 조회용 + torch/np seed 모두에 사용.
                   default = config.RANDOM_SEED (=42).
      --arms     : ['T0','T2'] 의 부분집합. v02 는 T3 미지원.
      --epochs   : 최대 epoch (default 30; early-stop 으로 보통 그 전에 종료).
      --batch    : pooled DataLoader 배치 사이즈 (default 256).
      --lr       : Adam learning rate (default 1e-3).
      --patience : early-stop patience — val_mae 개선 없는 epoch 누적 한도 (default 8).
      --lam      : peak_aux 가중치 λ (default 0.3, T2 에서만 의미).

    [경로]
      입력 split: `outputs/v02_fl_8020_ratio/splits/v02_8020_seed{S}.yaml`
                  (없으면 splits.load_v02_split 가 FileNotFoundError 를 띄움 →
                  먼저 `01_make_split.py --seed {S}` 를 돌려야 함).
      출력 루트 : `outputs/v02_fl_8020_ratio/seed{S}/{T0,T2}/best.pt`
                  + 같은 폴더의 `training_log.json`.
    """
    ap = argparse.ArgumentParser(description="Train T0/T2 on v02 80-train-apt split for one seed.")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED, help="Seed for split lookup AND model init.")
    ap.add_argument("--arms", nargs="+", default=["T0", "T2"], choices=["T0", "T2"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.3)
    args = ap.parse_args()

    # 80개 train apt 리스트 로드 (파일 없으면 즉시 에러 — 01_make_split 선행 필수).
    apts = load_v02_split(args.seed)["train"]
    # seed 별로 출력 폴더를 분리 → 멀티시드 결과가 서로 덮어쓰지 않도록.
    out_root = V02_OUT_ROOT / f"seed{args.seed}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[setup] seed={args.seed}; {len(apts)} train apts; arms: {args.arms}")
    print(f"[setup] device={DEVICE}; out_root={out_root}")
    # arm 별 순차 학습. 둘 사이에는 가중치 공유 없음 (각각 from-scratch).
    for arm in args.arms:
        print(f"\n========== {arm} (seed {args.seed}) ==========")
        train_arm(
            arm,
            apts,
            args.epochs,
            args.lr,
            args.batch,
            args.patience,
            args.lam,
            args.seed,
            out_root,
        )


if __name__ == "__main__":
    main()
