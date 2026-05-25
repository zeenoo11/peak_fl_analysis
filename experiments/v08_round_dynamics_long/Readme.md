# v08 — round-dynamics long-round 재실행

v06 (R=20, E=40) 셀이 round 2~3에서 plateau 에 도달하는 문제를 FL 표준 (E, R) 비율
~10 영역 (McMahan 2017 / Li 2020 FedProx / Collins 2021 FedRep default) 으로 옮긴
v06 mirror. **(E, R) 만 변경 — 다른 모든 invariant (backbone, λ_aux=0.3, hr_weight=0.1,
W5 operating points, INPUT_SIZE=96, HORIZON=24, batch=512, lr=1e-3, weight_decay=1e-5,
algorithm-specific extras, federated codebook K_local=2 / M=32) 는 v06 과 동일.**

| 항목 | v06 | v08 |
|---|---|---|
| FL local epochs (E) | 40 | **5** |
| FL rounds (R) | 20 | **150** |
| T = E × R | 800 | **750** |
| R / E | 0.5 | **30** |
| Centralised epochs | 40 | 40 (변경 없음) |

> Centralised 셀은 user 지시에 따라 (E, R) 외 변경 없음 — v06 과 동일하게 epochs=40.
> 따라서 v06 의 compute-budget mismatch (centralised 40 vs FL 800 epoch-equiv) 가
> v08 에서도 유지 (centralised 40 vs FL 750 epoch-equiv). 절대 비교는 약한 claim,
> FL 알고리즘 간 상대 비교가 신뢰할 수 있는 결론.

## 실험 수행

```bash
# 0501 24시 수행
# FL 실험을 다섯 가지 종류의 FL로 수행
_ALGO_PRETTY = {
    "fedavg":   "FedAvg",
    "fedprox":  "FedProx",
    "fedrep":   "FedRep",
    "ditto":    "Ditto",
    "fedproto": "FedProto",
}
# seed 42
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedavg --local_epochs 5 --rounds 150
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedprox --local_epochs 5 --rounds 150
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedrep --local_epochs 5 --rounds 150
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm ditto --local_epochs 5 --rounds 150
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedproto --local_epochs 5 --rounds 150
# seed 123
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedavg --local_epochs 5 --rounds 150 --seed 123
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedprox --local_epochs 5 --rounds 150 --seed 123
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedrep --local_epochs 5 --rounds 150 --seed 123
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm ditto --local_epochs 5 --rounds 150 --seed 123
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedproto --local_epochs 5 --rounds 150 --seed 123
# seed 7
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedavg --local_epochs 5 --rounds 150 --seed 7
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedprox --local_epochs 5 --rounds 150 --seed 7
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedrep --local_epochs 5 --rounds 150 --seed 7
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm ditto --local_epochs 5 --rounds 150 --seed 7
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedproto --local_epochs 5 --rounds 150 --seed 7
```

### 결과 분석

필드 정의

- main — forecast main loss = MAE(ŷ, y) (z-norm scale, 학습 도중 client 평균; 라운드 내 모든 client × batch 의 weighted mean).
- aux — peak-aux loss = peak_amp_MSE + hr_weight · peak_hour_CE (가중치 적용 전, 단순 합). 결합 손실 L = main + 0.3·aux 으로 backprop.
- prox (FedProx) — proximal term (μ/2) ‖θ_local − θ_global‖² 의 평균값. McMahan/Li2020 의 FedProx anchoring 강도 측정.
- wall — 라운드 wall-clock 초. 1라운드 = (모든 client local-train E epoch + server aggregate + per-client val 포워드) 합.
- gm_main (Ditto) — global model 의 main_loss. Ditto 의 server-side FedAvg trunk.
- pm_main (Ditto) — personal model 의 main_loss. 각 client 가 보유하는 개인용 사본 (Ditto λ-regularised).
- pm_pull (Ditto) — personal "pull" term (λ/2) ‖θ_pers − θ_global‖². personal model 이 global 로부터 얼마나 끌어당겨지는지.
- proto (FedProto) — prototype regularisation ‖h_g − global_prototype_c*‖². h_g 가 cluster prototype 으로 얼마나 가까운지.


## 0502 옵션 A + λ=0 ablation 수행

### 변경 요지

- **옵션 A (round-by-round test trajectory + train loss trajectory 추가)**:
  `RoundLogger.log_round` 가 매 라운드마다 val 뿐 아니라 test 도 forward 하고,
  `train.loss_mean_last_epoch` 도 trajectories.npz 에 적립한다. McMahan2017 FedAvg
  Figure 2 (round vs test accuracy) / FedProx Figure 2 (round vs training loss)
  convention 에 직접 비교 가능. figure 추가: `F1b_round_vs_test_pape.png`,
  `F1c_round_vs_train_loss.png` (기존 F1/F2/F3 그대로 유지).

- **aux_lambda=0 ablation**: `--aux_lambda 0` 일 때 cell_name 이 자동으로
  `-MAEonly` suffix 가 붙어 `V6-Dyn-{A,B-*}-MAEonly/` 디렉토리로 분리 저장된다.
  default(λ=0.3) 결과는 덮어쓰지 않음. NBEATSx 본체만 학습 (aux_head 는 forward
  되지만 gradient=0 → 학습 거동은 backbone-only 와 동치).

cell_name 매핑:

| 호출 | cell_name | 디렉토리 |
|---|---|---|
| `--aux_lambda 0.3` (default) | `V6-Dyn-{A,B-Algo}` | 기존과 동일, 옵션 A trajectory 로 덮어쓰기 |
| `--aux_lambda 0`             | `V6-Dyn-{A,B-Algo}-MAEonly` | 신규 분리 |

### 실험 수행

```bash
# ====================================================================
# (1/3) default (λ=0.3) 재실행 — 옵션 A 의 새 trajectory 로 갱신
#       기존 V6-Dyn-{A,B-*}/ 디렉토리에 덮어씀 (multiseed numbers 는 그대로)
# ====================================================================
# seed 42
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/01_centralised.py" --seed 42 --epochs 40
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedavg --local_epochs 5 --rounds 150 --seed 42
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedprox --local_epochs 5 --rounds 150 --seed 42
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedrep --local_epochs 5 --rounds 150 --seed 42
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm ditto --local_epochs 5 --rounds 150 --seed 42
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedproto --local_epochs 5 --rounds 150 --seed 42
# seed 123
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/01_centralised.py" --seed 123 --epochs 40
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedavg --local_epochs 5 --rounds 150 --seed 123
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedprox --local_epochs 5 --rounds 150 --seed 123
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedrep --local_epochs 5 --rounds 150 --seed 123
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm ditto --local_epochs 5 --rounds 150 --seed 123
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedproto --local_epochs 5 --rounds 150 --seed 123
# seed 7
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/01_centralised.py" --seed 7 --epochs 40
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedavg --local_epochs 5 --rounds 150 --seed 7
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedprox --local_epochs 5 --rounds 150 --seed 7
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedrep --local_epochs 5 --rounds 150 --seed 7
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm ditto --local_epochs 5 --rounds 150 --seed 7
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedproto --local_epochs 5 --rounds 150 --seed 7

# ====================================================================
# (2/3) ablation (λ=0) — MAEonly suffix 자동 분리 (NBEATSx backbone-only)
#       V6-Dyn-{A,B-*}-MAEonly/ 디렉토리에 신규 저장
# ====================================================================
# seed 42
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/01_centralised.py" --seed 42 --epochs 40 --aux_lambda 0
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedavg --local_epochs 5 --rounds 150 --seed 42 --aux_lambda 0
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedprox --local_epochs 5 --rounds 150 --seed 42 --aux_lambda 0
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedrep --local_epochs 5 --rounds 150 --seed 42 --aux_lambda 0
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm ditto --local_epochs 5 --rounds 150 --seed 42 --aux_lambda 0
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedproto --local_epochs 5 --rounds 150 --seed 42 --aux_lambda 0
# seed 123
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/01_centralised.py" --seed 123 --epochs 40 --aux_lambda 0
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedavg --local_epochs 5 --rounds 150 --seed 123 --aux_lambda 0
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedprox --local_epochs 5 --rounds 150 --seed 123 --aux_lambda 0
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedrep --local_epochs 5 --rounds 150 --seed 123 --aux_lambda 0
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm ditto --local_epochs 5 --rounds 150 --seed 123 --aux_lambda 0
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedproto --local_epochs 5 --rounds 150 --seed 123 --aux_lambda 0
# seed 7
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/01_centralised.py" --seed 7 --epochs 40 --aux_lambda 0
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedavg --local_epochs 5 --rounds 150 --seed 7 --aux_lambda 0
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedprox --local_epochs 5 --rounds 150 --seed 7 --aux_lambda 0
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedrep --local_epochs 5 --rounds 150 --seed 7 --aux_lambda 0
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm ditto --local_epochs 5 --rounds 150 --seed 7 --aux_lambda 0
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/02_fl_dynamics.py" --algorithm fedproto --local_epochs 5 --rounds 150 --seed 7 --aux_lambda 0

# ====================================================================
# (3/3) aggregate + figure rendering
#       multiseed_summary.json + trajectories.npz 갱신,
#       F1/F1b/F1c/F2/F3 + (있으면) F4/F5 (-MAEonly variant) PNG 출력
# ====================================================================
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/06_aggregate.py" --seeds 42 123 7
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/07_make_figures.py"
```

### 예상 wall-clock (v06 측정값 기준 추정; v08 은 E=5/R=150 로 T=750 ≈ v06 T=800 와 유사 budget)

총 epoch-equivalent budget (T = E × R) 은 v06 대비 ≈6% 감소 (800 → 750) 이므로 학습
자체 wall-clock은 v06 과 유사하다. 다만 v08 은 라운드 수가 7.5× 늘어나 라운드별 overhead
(aggregation + per-client val/test forward + RoundLogger I/O) 가 누적되어 +30~50% 증가가
예상된다. test forward 비용 (Option A trajectory) 도 라운드 수에 비례.

| Cell | per-seed (v06 ref) | per-seed (v08 추정) | (1/3) default ×3 | (2/3) ablation ×3 |
|---|---|---|---|---|
| centralised (epochs=40, 변경 없음) | ~75s | ~75s | ~3.8m | ~3.8m |
| FedAvg | ~750s | ~1000s | ~50m | ~50m |
| FedProx | ~1600s | ~2100s | ~105m | ~105m |
| FedRep | ~700s | ~950s | ~48m | ~48m |
| Ditto | ~2900s | ~3800s | ~190m | ~190m |
| FedProto | ~900s | ~1200s | ~60m | ~60m |
| **합계** |  |  | **~7.5h** | **~7.5h** |

총 ≈15~16 시간. background 실행 필수. v06 대비 약 +35% 증가 (라운드 overhead 누적).


## 0503 Phase 2 codebook stacking (post-hoc Peak-VQ on 6 backbones, λ=0.3 only)

### 변경 요지

- v01-v05 의 Peak-VQ codebook 을 v06 Phase 1 의 6 backbone (centralised + 5 FL,
  λ=0.3 default 만 — Peak-VQ 의 코어 가정인 *peak-aware h_generic structure* 가
  aux head 학습으로 형성되므로 λ=0 (-MAEonly) variant 는 codebook 의 정당성이
  약함 → Phase 2 에서는 default cell 만 다룸) 에 *post-hoc stacking*. 모든
  backbone 은 frozen 으로 유지 (`final_state_dict.pt` 를 strict=True load).
- **Codebook fit protocol** = federation contract 그대로:
    - centralised cell (V6-Dyn-A_centralised) → pooled KMeans (모든 가구 train h_g 합쳐 KMeans++).
    - FL cell (V6-Dyn-B-*) → 2-stage hierarchical *federated* KMeans (`src/fl/codebook_fl.py`).
- **Correction** = CMO-only (Cluster-Mean Offset, Gaussian template α_w1 dropped):
  `ŷ_corr = ŷ_base + α_v0 · o_{c*}` (α_v0 = 1.0).
- **Test split** = per-client 20% test windows (학습에 사용된 적 없는 unseen 미래
  윈도우). v01-v05 의 cold-zero-shot 평가와 다르지만 v06 의 round-level FL 프로토콜
  에는 cold partition 이 없으므로 자연스러운 평가 타깃.
- 하이퍼파라미터 고정: M=32, K_local=2, stride=24 (CLAUDE.md / v01-v05 / FedCB 일치).

> **참고**: 08 driver 자체는 12 cell 모두 지원 (`--cell V6-Dyn-*-MAEonly` 도 valid choice).
> Phase 2 의 *현재 권장 plan* 만 6 cell 로 좁히는 것 — 향후 MAEonly 를 비교에
> 추가하고 싶으면 README 에 cell 추가만 하면 됨. `09_aggregate_codebook.py` 는
> `_discover_cells` 로 디렉토리를 자동 스캔하므로 추가/제거에 유연.

산출 파일 — per seed × per cell:

```
outputs/v08_round_dynamics_long/seed{S}/{cell}/codebook_lift.json
```

### 실험 수행 (6 cells × 3 seeds = 18 runs)

```bash
# ====================================================================
# (1/3) seed 42 — 6 cells (codebook stacking, λ=0.3 only)
# ====================================================================
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/08_codebook_stacking.py" --seed 42 --cell V6-Dyn-A_centralised
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/08_codebook_stacking.py" --seed 42 --cell V6-Dyn-B-FedAvg
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/08_codebook_stacking.py" --seed 42 --cell V6-Dyn-B-FedProx
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/08_codebook_stacking.py" --seed 42 --cell V6-Dyn-B-FedRep
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/08_codebook_stacking.py" --seed 42 --cell V6-Dyn-B-Ditto
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/08_codebook_stacking.py" --seed 42 --cell V6-Dyn-B-FedProto

# ====================================================================
# (2/3) seed 123 — 6 cells
# ====================================================================
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/08_codebook_stacking.py" --seed 123 --cell V6-Dyn-A_centralised
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/08_codebook_stacking.py" --seed 123 --cell V6-Dyn-B-FedAvg
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/08_codebook_stacking.py" --seed 123 --cell V6-Dyn-B-FedProx
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/08_codebook_stacking.py" --seed 123 --cell V6-Dyn-B-FedRep
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/08_codebook_stacking.py" --seed 123 --cell V6-Dyn-B-Ditto
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/08_codebook_stacking.py" --seed 123 --cell V6-Dyn-B-FedProto

# ====================================================================
# (3/3) seed 7 — 6 cells
# ====================================================================
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/08_codebook_stacking.py" --seed 7 --cell V6-Dyn-A_centralised
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/08_codebook_stacking.py" --seed 7 --cell V6-Dyn-B-FedAvg
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/08_codebook_stacking.py" --seed 7 --cell V6-Dyn-B-FedProx
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/08_codebook_stacking.py" --seed 7 --cell V6-Dyn-B-FedRep
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/08_codebook_stacking.py" --seed 7 --cell V6-Dyn-B-Ditto
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/08_codebook_stacking.py" --seed 7 --cell V6-Dyn-B-FedProto

# ====================================================================
# aggregate + figure rendering
#   codebook_lift_summary.json + figures/F6_codebook_lift.png 출력
#   (figure 의 MAEonly subplot 은 디렉토리에 codebook_lift.json 가 없어
#    자동으로 axis off 처리됨 — 10_make_codebook_figure.py:181-184)
# ====================================================================
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/09_aggregate_codebook.py" --seeds 42 123 7
& "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/.venv/Scripts/python.exe" "c:/Users/HOME/JW/Research Docs/FL_Peak_Project/experiments/v08_round_dynamics_long/10_make_codebook_figure.py"
```

### 예상 wall-clock

| 단계 | per-cell | per-seed (6 cells) | 18-run 합계 |
|---|---|---|---|
| 08_codebook_stacking | ~5s (CUDA) / ~15s (CPU) | ~30s | ~1.5m |
| 09_aggregate_codebook | n/a | ~3s | ~3s |
| 10_make_codebook_figure | n/a | ~5s | ~5s |

전체 ≈2~3 분. background 실행 불필요.
