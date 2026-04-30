# v05 — Fully-federated codebook construction (FedCB)

> Plan: `plans/v05-01_fedcb_codebook.md`
> Paper draft: `papers/v05_draft/v05_fedcb_codebook.md`
> Executor log: `papers/v05_draft/EXPERIMENT_LOG.md`

## 1. 계획 (Plan)

### 1.1 동기

`papers/pfl_unified/paper.md` §4.2 Table 1의 **Stacked-Aux** 행 (41.93 ± 1.30 PAPE)은 NBEATSxAux backbone을 federated 학습하지만 codebook 구성 단계는 80가구의 `h_generic`을 서버로 모아 KMeans를 1회 수행하는 **centralised** 단계다 (`experiments/v04_full_baseline_comparison/09_fix_rerun/02_fedavg_nbeatsx_aux.py:gather_train_segment_aux`).

KIIE oral 발표 직전 검토 (2026-04-29)에서 "fully FL이 아니다"라는 framing 약점 식별. v05는 codebook 구성 단계도 federated로 변환해 KIIE 발표에서 "fully-FL framework"로 격상 가능한지 결정한다.

### 1.2 방법 변경

| 단계 | v04 (기존) | v05 (변경) |
|---|---|---|
| Phase A (backbone) | FedAvg-NBEATSxAux | 동일 (재학습 안 함, 09_fix_rerun final_state_dict.pt 재사용) |
| Phase B (codebook) | 서버가 80×~3000 windows의 `h_g` raw 수집 → KMeans++ 1회 | **2-stage hierarchical KMeans**: (1) 각 client가 local KMeans, (2) 서버가 sample-count-weighted KMeans로 merge |
| Phase B residual | 서버가 raw window residual 수집 → cluster 평균 | client별 cluster 단위 partial sum + count만 업로드 → 서버가 평균 |
| Phase C (cold) | W5 hybrid (`y_hat + α_v0·o + α_w1·g`) at 두 op-point | **CMO-only** (`y_hat + α·o`) at 단일 α=1.0, R1 routing (h_g 1-NN) |

### 1.3 Gates

| Gate | 기준 | 확인 대상 |
|---|---|---|
| Gate 1 | v02 §B.3 paper anchor 44.18 ± 1.5 pp | V5-FedCB-0 (paper-load 검증) |
| Gate 2 | V5-FedCB-1 (K_local=4, α=1.0) PAPE ≤ 52 % | absolute 임계 |
| Gate 3 | K_local ∈ {2, 4, 8} 중 최소 하나 ≤ 52 % | sensitivity |

## 2. 과정 (Process)

### 2.1 구현

| 파일 | 역할 |
|---|---|
| `src/fl/codebook_fl.py` | 3개 helper: `local_codebook_step`, `merge_local_codebooks`, `federated_residual_offsets` (sklearn KMeans 직접 호출, Stage 2 sample_weight 사용) |
| `experiments/v05_fedcb_codebook/01_fedcb_codebook.py` | Per-seed driver (federated only). args: `--seed`, `--K_local`, `--alpha`, `--M`, `--batch_size`, `--stride` |
| `experiments/v05_fedcb_codebook/02_aggregate.py` | 3-seed mean ± std + V5-FedCB-0를 v02 W_component_results.json에서 직접 로드 + Gate 1/2/3 평가 |
| `experiments/v05_fedcb_codebook/03_communication.py` | seed-independent. K_local별 통신량 표 + paper.md §4.7 호환 context |
| `tests/test_codebook_fl.py` | 4 pytest (determinism / util-1.0 sanity / tiny-client fallback / residual offsets shape) |

### 2.2 Integrity check

- pytest 4 green (8.0 s)
- `strict=True` state_dict 로드 (FedAvg-NBEATSxAux), 모든 seed 체크포인트 존재 확인
- bf16 autocast on CUDA / fp32 fallback 확인
- 출력 경로 namespacing `outputs/v05_fedcb_codebook/seed{S}/...` 준수
- MLflow 없음 (repo 컨벤션 일치, result.json + print 로깅)

초기 검토에서 SEND-BACK 1회 발생: ① `--mode centralised`가 R1 routing 사용 (plan은 R0 명시), ② `02_aggregate.py`의 v02 anchor 로드 누락. Engineer 패치로 ① centralised 모드 통째로 제거 (federated-only driver), ② aggregator가 v02 W_component_results.json 직접 로드. pytest 4 green 유지, federated 경로 알고리즘 변경 없음 (smoke 결과 49.94 bit-identical 유지).

### 2.3 Sweep

| Stage | Cell | 실행 |
|---|---|---|
| A | V5-FedCB-0 | data-only (v02 JSON 로드, 재실행 0) |
| B | V5-FedCB-1 (K=4, α=1.0) | 3 seeds = 3 run |
| C | V5-FedCB-2a/b (K∈{2,8}, α=1.0) | 6 seeds = 6 run |
| D | V5-FedCB-3 (α sweep) | **skipped** — Gate 2 통과로 plan §실험매트릭스에서 conditional skip |
| E | aggregate + communication | 1-shot |

총 9 run, 13 분.

## 3. 결과 (Results)

### 3.1 Headline — 3-seed mean ± std

| Cell | Routing | α | PAPE (%) | HR@1 (%) | HR@2 (%) | MAE (kW) |
|---|---|---|---|---|---|---|
| V5-FedCB-0 (v02 anchor) | R0 | 1.5 | **44.18 ± 0.18** | 25.86 ± 2.59 | 37.12 ± 2.42 | 0.4487 ± 0.0252 |
| V5-FedCB-1 (K_local=4) | R1 | 1.0 | **50.17 ± 0.97** | 25.28 ± 1.30 | 37.24 ± 1.86 | 0.4392 ± 0.0230 |
| V5-FedCB-2a (K_local=2) | R1 | 1.0 | 50.70 ± 1.29 | 25.17 ± 1.73 | 37.08 ± 2.31 | 0.4390 ± 0.0229 |
| V5-FedCB-2b (K_local=8) | R1 | 1.0 | 50.37 ± 1.13 | 25.22 ± 1.76 | 37.27 ± 2.41 | 0.4389 ± 0.0229 |

Backbone-only (no codebook) reference (`fl_only` block, identical across K_local since same backbone): **PAPE 57.32 ± 1.55, HR@1 26.35 ± 1.67, HR@2 37.76 ± 1.56, MAE 0.4262 ± 0.0210**.

### 3.2 Per-seed PAPE (%)

| Cell | seed 42 | seed 123 | seed 7 | mean ± std |
|---|---|---|---|---|
| V5-FedCB-0 (v02 anchor) | 43.99 | 44.20 | 44.34 | 44.18 ± 0.18 |
| FL backbone only (`fl_only`) | 57.16 | 58.94 | 55.86 | 57.32 ± 1.55 |
| V5-FedCB-1 (K_local=4) | 49.94 | 51.23 | 49.33 | 50.17 ± 0.97 |
| V5-FedCB-2a (K_local=2) | 51.17 | 51.69 | 49.25 | 50.70 ± 1.29 |
| V5-FedCB-2b (K_local=8) | 50.54 | 51.40 | 49.16 | 50.37 ± 1.13 |

CMO 적용 효과: backbone-only 57.32 → V5-FedCB-1 50.17 = **PAPE −7.15 pp** (12.5 % 상대 개선).

### 3.3 VQ diagnostics — 시드별 codebook 진단

| K_local | seed | utilization | perplexity | k_min | k_max | n_empty | Stage 1 mean inertia | Stage 2 inertia |
|---|---|---|---|---|---|---|---|---|
| 2 | 42 | 1.000 | 26.17 | 29 | 1589 | 0 | 413.58 | 203.30 |
| 2 | 123 | 1.000 | 26.81 | 32 | 1311 | 0 | 282.10 | 221.65 |
| 2 | 7 | 1.000 | 25.08 | 31 | 1375 | 0 | 405.59 | 173.92 |
| 4 | 42 | 1.000 | 27.24 | 43 | 1246 | 0 | 130.19 | 424.70 |
| 4 | 123 | 1.000 | 26.84 | 32 | 1759 | 0 | 93.94 | 458.98 |
| 4 | 7 | 1.000 | 25.79 | 8 | 1807 | 0 | 125.95 | 364.41 |
| 8 | 42 | 1.000 | 27.67 | 28 | 1663 | 0 | 41.48 | 568.36 |
| 8 | 123 | 1.000 | 28.30 | 36 | 1516 | 0 | 36.12 | 615.38 |
| 8 | 7 | 1.000 | 27.23 | 9 | 1343 | 0 | 39.23 | 472.18 |

모든 셀에서 utilization=1.000, n_empty=0, perplexity 25-28 (M=32 중 약 78-89 % effective). Stage 1 inertia는 K_local 증가 시 단조 감소 (4.13×10² → 1.30×10² → 4.15×10¹), Stage 2는 단조 증가 — Stage 2가 받은 client centroid 풀이 커지면서 cluster당 평균 거리가 늘어남, 자연스러운 hierarchical 거동.

### 3.4 통신량 (V5-FedCB-4, seed-independent)

| 방식 | client당/round | round당 합계 | rounds | 총 bytes | boundary crosses |
|---|---|---|---|---|---|
| Centralised codebook 1-shot (v01-v03 anchor) | — | 4,939,264 | 1 | **4.94 MB** | 1 |
| **v05 federated K_local=2** | 3,720 | 305,792 | 1 | **0.31 MB** | 2 |
| **v05 federated K_local=4** | 4,240 | 347,392 | 1 | **0.35 MB** | 2 |
| **v05 federated K_local=8** | 5,280 | 430,592 | 1 | **0.43 MB** | 2 |
| (참고) FedAvg/FedProx/Ditto | 262,736 | 21,018,880 | 20 | 420 MB | 20 |
| (참고) FedRep | 224,256 | 17,940,480 | 20 | 359 MB | 20 |

v05 K_local=4 vs centralised codebook: **14.2× 감소** (4.94 MB → 0.35 MB), boundary cross +1. Stage 1 (centroid + count) + Stage 3 (residual sum + count). Stage 2 (server broadcast) 8,192 B는 v04 컨벤션상 boundary cross 카운트 제외.

### 3.5 Wall-clock per cell

| K_local | seed 42 | seed 123 | seed 7 | mean | Phase B (mean) | Phase C (mean) |
|---|---|---|---|---|---|---|
| 2 | 85.9 s | 100.1 s | 100.3 s | 95.4 s | 85.3 s | 10.1 s |
| 4 | 63.7 s | 73.3 s | 72.9 s | 70.0 s | 62.9 s | 7.1 s |
| 8 | 101.5 s | 98.6 s | 100.7 s | 100.3 s | 90.5 s | 9.8 s |

Stage B (3 run) ~210 s + Stage C (6 run) ~579 s + Stage E ~5 s = **약 13 분 합계**.

### 3.6 Gate 결과

| Gate | 기준 | 측정값 | Pass |
|---|---|---|---|
| Gate 1 (v02 anchor load) | mean PAPE ∈ [42.68, 45.68] | 44.18 ± 0.18 (3/3 JSON 로드) | ✓ true-by-construction |
| Gate 2 (V5-FedCB-1 default) | mean PAPE ≤ 52.0 | 50.17 | ✓ |
| Gate 3 (K_local sensitivity) | 최소 1개 cell ≤ 52.0 | K=2: 50.70 ✓ / K=4: 50.17 ✓ / K=8: 50.37 ✓ | ✓ (3/3) |

세 Gate 모두 통과 → "fully-FL framework via hierarchical KMeans" framing 격상 정당화됨.

### 3.7 MSE / RMSE 재forward (KIIE 발표용 추가 측정)

기존 multiseed 산출물은 PAPE / HR@1 / HR@2 / MAE만 저장했고 raw predictions를 보존하지 않아 MSE/RMSE는 재forward로만 얻을 수 있음. v05 + 모든 v04 baseline (Local-only 제외)을 cold pool에 다시 돌려 6 metric 전부 갖춘 표 작성.

| Method (cold PAPE 낮은 순) | PAPE (%) | MAE (kW) | **MSE (kW²)** | **RMSE (kW)** |
|---|---|---|---|---|
| **V5-FedCB-1 (K=4, Proposed)** | **50.17 ± 0.97** | 0.4392 ± 0.0230 | **0.5060 ± 0.0326** | **0.7111 ± 0.0228** |
| V5-FedCB-2a (K=2) | 50.70 ± 1.29 | 0.4390 ± 0.0229 | 0.5055 ± 0.0333 | 0.7107 ± 0.0234 |
| V5-FedCB-2b (K=8) | 50.37 ± 1.13 | 0.4389 ± 0.0229 | 0.5058 ± 0.0325 | 0.7110 ± 0.0228 |
| DLinear (NF) | 50.41 ± 0.80 | 0.4263 ± 0.0228 | 0.5167 ± 0.0350 | 0.7185 ± 0.0242 |
| Crossformer (NF) | 52.56 ± 1.71 | 0.4055 ± 0.0247 | 0.5199 ± 0.0368 | 0.7207 ± 0.0254 |
| NHITS_fixed (NF) | 52.73 ± 1.71 | 0.4053 ± 0.0254 | 0.5195 ± 0.0374 | 0.7205 ± 0.0258 |
| Chronos-Bolt small (FM) | 52.69 ± 1.56 | 0.4177 ± 0.0241 | 0.5451 ± 0.0360 | 0.7381 ± 0.0244 |
| TimesFM (FM) | 54.27 ± 2.15 | 0.4169 ± 0.0250 | 0.5450 ± 0.0399 | 0.7379 ± 0.0269 |
| FedAvg | 56.37 ± 1.39 | 0.4193 ± 0.0213 | 0.5283 ± 0.0302 | 0.7267 ± 0.0208 |
| FedProx | 56.30 ± 1.54 | 0.4202 ± 0.0209 | 0.5297 ± 0.0308 | 0.7276 ± 0.0211 |
| FedProto | 56.40 ± 1.43 | 0.4190 ± 0.0212 | 0.5288 ± 0.0308 | 0.7270 ± 0.0212 |
| Ditto | 56.39 ± 1.63 | 0.4187 ± 0.0213 | 0.5300 ± 0.0322 | 0.7278 ± 0.0221 |
| FedAvg-NBEATSxAux (no codebook) | 57.32 ± 1.55 | 0.4262 ± 0.0210 | 0.5300 ± 0.0314 | 0.7278 ± 0.0215 |
| FedRep | 57.19 ± 1.51 | 0.4235 ± 0.0205 | 0.5329 ± 0.0318 | 0.7298 ± 0.0217 |
| Chronos-T5 tiny (FM) | 63.09 ± 3.08 | 0.4533 ± 0.0262 | 0.7073 ± 0.0478 | 0.8407 ± 0.0284 |

15 method × 3 seed = 45 cell 재forward의 PAPE 재현치가 모두 published 결과와 일치 (≤ 0.04 pp). MSE/RMSE 신뢰 가능. 미측정: Local-only (per-cold-apt 학습, state 없음), peakvq_on_fedavg/fedrep (codebook 미저장).

#### 핵심 발견: MAE에서 안 보이던 ordering이 MSE에서 명확

- **MAE 범위**: 0.40-0.45 kW (좁아서 method 간 차이 미세).
- **MSE 범위**: 0.51-0.71 kW² (V5 < NF < FL < FM < Chronos-T5 일관 ranking).
- **Per-seed paired**: 모든 seed에서 V5-FedCB family가 top 3 점령 (45 cell 중 v05-K{2,4,8} = 9 cell). across-seed std (0.033)에 묻히지 않는 robust signal.

#### MAE ↔ MSE trade-off (codebook 효과의 메커니즘)

| 지표 | Backbone-only (FedAvg-NBEATSxAux) | Proposed (V5-FedCB-1) | Δ |
|---|---|---|---|
| MAE | 0.4262 | 0.4392 | **+3.05 %** |
| **MSE** | **0.5300** | **0.5060** | **−4.53 %** |
| RMSE | 0.7278 | 0.7111 | −2.30 % |

Peak-aware 보정이 평균 오차는 미세하게 키우되 (over-correction in some windows) outlier-heavy 큰 오차를 강하게 줄임 — 이게 PAPE 12.5 % 개선의 정확한 메커니즘. **MAE 단일로 "차이 없음" 결론은 잘못된 해석**.

#### FM의 PAPE-vs-MSE 역전 현상

Chronos-Bolt small이 FL 평균 (PAPE 56.51)보다 PAPE는 −3.82 pp 낮지만 MSE는 +2.87 % **높음** (0.5451 vs FL 평균 0.5299). zero-shot FM이 peak amplitude는 그럭저럭 맞추되 전반 fit은 약하다는 해석. v05는 양쪽 다 잡음 (peak-aware aux head + UMass 분포 학습된 backbone + cluster offset).

상세 표 / 가설 검증은 `papers/v05_draft/presentation_tables.md` 참조.

### 3.8 한계 (해석 시 유의)

V5-FedCB-0 (44.18) ↔ V5-FedCB-1 (50.17) 간 **6.0 pp 갭**은 세 변수 동시 변화의 합산이라 단일 요인 분해 불가:

- **backbone**: v02 T2 (centralised) → v04 FedAvg
- **routing**: R0 (KEY pool 1-NN) → R1 (h_g 1-NN)
- **codebook 구성**: centralised → federated hierarchical

Federation 자체의 순수 PAPE 비용을 분리하려면 추가 cell — 예: FedAvg-NBEATSxAux + centralised codebook + R1 + α=1.0 — 이 필요. KIIE 일정에 따라 별도 follow-up.

## 산출물 위치

```
outputs/v05_fedcb_codebook/
├── seed{42,123,7}/
│   └── fedcb_K{2,4,8}/{result.json, codebook.npz}
├── multiseed_summary.json
├── communication_summary.json
├── mse_recompute_summary.json        (12 method × 3 seed: FL/NF/v05/FedAvg-NBEATSxAux)
└── mse_recompute_fm_summary.json     (FM 3 × 3 seed: Chronos-Bolt/T5/TimesFM)
papers/v05_draft/
├── v05_fedcb_codebook.md             (paper-style draft, §1-6)
├── EXPERIMENT_LOG.md                 (chronological executor log)
└── presentation_tables.md            (KIIE 발표용 5 표 + 7 가설 검증)
```

## 시드별 실행 명령

```bash
# v05 sweep (Phase A reused, Phase B'/C only)
for s in 42 123 7; do
    for k in 2 4 8; do
        uv run python experiments/v05_fedcb_codebook/01_fedcb_codebook.py \
            --seed $s --K_local $k --alpha 1.0
    done
done
uv run python experiments/v05_fedcb_codebook/02_aggregate.py
uv run python experiments/v05_fedcb_codebook/03_communication.py

# MSE / RMSE 재forward
uv run python experiments/v05_fedcb_codebook/04_recompute_mse.py    # 12 method × 3 seed (~4 min)
uv run python experiments/v05_fedcb_codebook/05_recompute_mse_fm.py # 3 FM × 3 seed (~5 min)
```
