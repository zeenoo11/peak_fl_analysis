
# Title

연합학습 환경에서 가구별 피크 부하 예측을 위한 라운드 단위 학습 동역학 분석과 사후 코드북 스태킹

경북대학교 데이터사이언스대학원 전진우

2026.06.04

발표 도입

안녕하십니까. 경북대학교 데이터사이언스대학원 소속 석사과정 전진우 입니다. 발표 주제는 연합학습 환경에서 가구별 피크 부하 예측 정확도를 끌어올리는 두 가지 축, 즉 라운드 단위 FL 학습 동역학과 사후 코드북 스태킹입니다. 시작하겠습니다.

---

# Introduction

## 전력 계통 관리에서 피크의 의미

전력 계통 운영은 본질적으로 발전과 소비의 실시간 균형을 맞추는 일입니다. 이때 결정적인 변수가 피크입니다. 일별 최대 부하 시점에 발전 용량이 부족하면 곧바로 정전이나 비상 발전으로 이어지므로, 계통 운영자는 피크를 기준으로 예비력을 산정하고 송배전 설비를 설계합니다.

피크 정보가 가지는 영향은 다음 세 가지 운영 의사결정에 직접 들어갑니다. 첫째, 발전기 기동정지 계획에서 피크 시점의 발전 자원 동원 순서를 결정합니다. 둘째, ESS 충방전 스케줄링에서 피크 절감을 위한 방전 타이밍을 정합니다. 셋째, 송배전 설비 증설 계획에서 변압기와 선로 용량을 피크 부하 기준으로 산정합니다.

## 가구 단위로 내려가야 하는 이유

전통적으로 피크 예측은 시군구 단위 또는 변전소 단위 집계 부하를 대상으로 수행되었습니다. 그러나 최근 분산 전원과 V2G, 가정용 ESS의 보급으로 가구 단위 피크 정보의 중요성이 커지고 있습니다.

특히 본 연구가 목표로 하는 응용은 가정용 ESS의 자동 운영입니다. 가구별 피크 시점과 크기를 사전에 알면 ESS가 최적 충방전 스케줄을 수립할 수 있고, 이는 가구 전기 요금 절감과 동시에 계통의 분산형 피크 절감에 기여합니다.

## 가구 단위 피크 예측의 두 가지 본질적 어려움

첫째, 가구별 부하의 변동성이 매우 큽니다. 같은 단지의 가구라도 거주자 수, 생활 패턴, 가전 보유에 따라 피크 시점과 크기가 매우 다르며, 같은 가구 내에서도 일별 변동이 큽니다. 평균 부하 기준으로 학습한 모델은 이 피크 영역의 신호를 충분히 잡지 못합니다.

둘째, 가구 단위 부하 데이터는 프라이버시 민감 정보입니다. 시간 단위 소비 데이터로부터 거주자 수, 재실 여부, 출퇴근 시각, 주요 활동 패턴이 추정 가능합니다. 따라서 다수 가구의 raw 데이터를 중앙 서버로 모으는 방식은 점점 어려워지고 있으며, 한국에서도 개인정보보호법 적용 대상으로 다루어집니다.

## 연합학습이 자연스러운 이유

이 두 제약 조건을 함께 다루는 자연스러운 framework가 연합학습입니다. 가구의 raw 데이터는 가구에 머물고, 모델 학습에 필요한 정보만 서버와 교환합니다. 본 연구는 이 연합학습 framework 안에서 (1) 라운드 단위로 학습이 진행될 때 어떤 동역학이 관찰되는지, 그리고 (2) 학습 종료 후 사후 코드북 모듈로 어디까지 정확도를 끌어올릴 수 있는지를 동시에 분석합니다.

---

# Goal

## 연구 목적

본 연구는 다음 세 가지 목적을 동시에 달성하는 것을 목표로 합니다.

1. raw 가구 부하 데이터를 가구 외부로 반출하지 않는 연합학습 환경에서 가구별 피크 예측 모델을 라운드 단위로 학습하고, 다섯 가지 표준 FL 알고리즘 (FedAvg, FedProx, FedRep, Ditto, FedProto) 의 학습 동역학을 동일 backbone 상에서 직접 비교
2. 표준 연합학습이 가구 피크 예측에서 보이는 정확도 한계 (centralised 대비 약 +2 PAPE gap) 를 정확히 식별하고, 그 한계를 보완할 수 있는 사후 코드북 모듈을 설계
3. 사후 코드북 모듈이 모든 FL 알고리즘 backbone 위에 동일하게 결합 가능한 직교 component이며, 추가 통신 비용은 federated 학습 단계의 통신량 대비 무시할 수 있는 수준임을 검증

## 평가 지표 정의

본 연구의 핵심 평가 지표는 PAPE, peak absolute percentage error입니다. 정의는 다음과 같습니다.

PAPE = (1/N) × Σ |max(y_true_i) − max(y_pred_i)| / max(y_true_i) × 100

여기서 y_true_i와 y_pred_i는 i번째 24시간 forecast 윈도우의 ground truth와 예측이며, max는 그 24시간 안의 최대값을 가져옵니다. 즉 PAPE는 예측된 24시간 부하의 최대값이 실제 최대값과 얼마나 다른지를 백분율로 측정합니다.

PAPE는 ESS 운영 의사결정에 직결되는 지표입니다. ESS는 24시간 안의 피크 시점에 방전하여 피크를 깎는 것이 목적이므로, 피크 크기 추정 오차가 곧 ESS 운영 손실로 이어집니다.

피크 시점의 정확도는 HR@1, HR@2 (Hit Rate within ±1 hour, ±2 hour)로 측정합니다. 또한 보조 지표로 MAE, MSE (kW²)를 함께 보고합니다. MSE는 outlier 큰 오차에 가중치를 두어 method 간 전반 fit 차이를 식별하는 보조 지표입니다.

---

# Related Works

## 연합학습 표준 알고리즘

표준 연합학습 알고리즘은 다음 세 갈래로 발전해 왔습니다.

첫째, FedAvg 계열로 모든 client의 모델 weight를 평균하는 가장 단순한 형태입니다.
둘째, FedProx로 client drift를 억제하는 proximal term을 추가합니다.
셋째, personalised FL 계열로 FedRep, Ditto, FedProto가 대표적이며, encoder는 공유하고 head는 client별로 두거나 global 모델과 local 모델을 동시에 유지하거나 prototype 정규화를 추가하는 구조입니다.

이 알고리즘들은 공통적으로 forecast loss의 평균값 최적화에 초점을 둡니다. 즉 피크 영역의 신호를 따로 강조하지 않으며, McMahan 2017과 FedProx 2020 모두 라운드 단위 학습 trajectory 비교를 통해 알고리즘 차이를 진단하는 관행을 정립했습니다. 본 연구는 동일한 라운드 단위 trajectory framework 위에서 다섯 알고리즘을 한 자리에 비교합니다.

## 피크에 특화된 시계열 예측 연구

피크 부하 예측에 특화된 선행 연구로 Zhang 등 2023의 Seq2Peak이 있습니다. 이 연구는 forecast loss에 peak loss를 가중 결합한 hybrid loss를 제안하며, ETTh, Electricity 등 집계 데이터셋에서 검증되었습니다. 그러나 가구 단위 및 연합학습 환경 검증은 없습니다.

## 가구 단위 부하 예측의 일반화 한계

가구 단위 부하 예측에서 강한 한계가 두 선행 연구에서 확인됩니다. BuildingsBench 2023은 90만 시뮬레이션 건물로 사전학습된 Transformer가 실제 거주 건물에서 zero-shot NRMSE 79%를 보고하며, 단순 persistence가 78%로 거의 동등한 수준임을 보였습니다. Peng 등 2019는 approximate entropy 분석으로 개별 가구 부하가 본질적으로 예측 어려움이 큰 신호임을 보였습니다.

이는 가구 단위에서는 모델 크기나 사전학습 양보다 task에 맞춘 inductive bias가 중요함을 시사합니다.

## 벡터 양자화의 시계열 활용

벡터 양자화는 self-supervised pretraining에서 시계열 token화에 사용되어 왔습니다. VQ-MTM 2024가 대표적입니다. 본 연구는 이 방향과 다르게, 학습된 backbone의 hidden representation 위에서 사후 KMeans를 통해 inference time correction module을 만드는 방식으로 벡터 양자화를 활용합니다. 또한 iterative federated KMeans는 TAR attack (arxiv:2511.07073) 위험으로 인해 회피하고, 2-stage hierarchical KMeans의 single-shot 구성을 채택합니다.

---

# Method

## 전체 구조 요약

본 framework는 두 phase로 구성됩니다. Phase 1에서 NBEATSx + peak-aux head backbone을 다섯 가지 FL 알고리즘으로 라운드 단위 학습하고, 모든 라운드에서 val/test forward, 통신 비용, client drift를 함께 기록합니다. Phase 2에서 학습이 종료된 backbone을 frozen 상태로 두고, 학습된 hidden representation 위에 federated codebook을 한 번에 구성하여 inference time correction module을 만듭니다.

학습 phase와 codebook 구성 phase 모두 fully-federated이며, raw 가구 데이터와 raw hidden representation, raw forecast residual 모두 서버로 전송되지 않습니다.

## Backbone — NBEATSxAux

Backbone으로 NBEATSx를 채택했습니다. 선택 근거는 NBEATSx의 stack-wise decomposition 구조가 보조 헤드와 자연스럽게 결합되기 때문입니다. NBEATSx는 trend, seasonal, generic 세 stack으로 forecast를 분해하는데, 보조 헤드를 generic stack hidden 위에만 부착하면 trend와 seasonal component는 보존하면서 generic 부분만 피크 task에 align됩니다.

[NBEATSx Architecture 삽입]

NBEATSx의 generic stack hidden을 h_g ∈ R^64라 할 때, 보조 헤드는 두 가지 출력을 가집니다.

(â, ĥ) = AuxHead(h_g)

여기서 â은 24시간 forecast 윈도우의 예측 피크 진폭이고, ĥ은 예측 피크 시점입니다.

학습 손실은 다음과 같습니다.

L_aux = MSE(â, max(y)) + 0.1 × CE(ĥ, argmax(y))

L_total = MAE(ŷ, y) + λ_aux × L_aux, λ_aux ∈ {0.3, 0}

여기서 λ_aux = 0.3은 v01–v05에서 carry-over한 default 값이며, λ_aux = 0은 본 연구가 라운드 단위 FL 환경에서 새롭게 도입한 MAE-only ablation입니다. 후술하듯이 이 ablation은 핵심 negative result로 이어집니다.

## Phase 1 — 라운드 단위 FL 학습 동역학

114개 UMass Smart* 가구가 모두 학습에 참여합니다. 각 가구의 시간 단위 부하 series를 chronological하게 train 70% / val 10% / test 20%로 분할하고, 라운드별로 val과 test 양쪽을 forward해 trajectory를 적립합니다. 입력 윈도우는 96시간, horizon은 24시간, stride는 24입니다.

| Hyperparameter | Value |
|---|---|
| Rounds          | 20 |
| Local epochs    | 40 |
| Optimiser       | Adam (lr=1e-3, weight_decay=1e-5) |
| Batch           | 512 |
| Participation   | full (C=1.0; 114 of 114 clients) |
| AMP             | bf16 on CUDA |

라운드 logger는 매 라운드마다 다음을 기록합니다.

- `val` / `test` 블록 — per-client forward 결과를 across-client mean/std (ddof=1) 로 집계
- `train` 블록 — last-epoch loss mean, n_steps_round
- `comm` 블록 — 알고리즘별 upload + broadcast bytes (cumulative)
- `drift_l2` — 라운드 종료 시점 client end-state간 평균 L2 distance
- `wall_seconds_round`

알고리즘 별 추가 hyperparameter는 paper-default를 그대로 사용합니다. FedProx μ=0.01, FedRep head_epochs=1, Ditto λ=0.1, FedProto K=32 / λ_proto=0.1.

## Phase 2 — Federated Codebook Stacking

학습이 종료된 backbone (`final_state_dict.pt`, strict=True load) 을 frozen 상태로 두고, 모든 가구의 train 윈도우에서 h_generic ∈ R^64와 forecast residual을 추출해 codebook을 구성합니다. Codebook 구성 protocol은 backbone이 centralised인지 federated인지에 따라 다릅니다.

- **Centralised cell** (V6-Dyn-A_centralised) → 모든 가구의 h_g를 pool하여 KMeans++(M=32, n_init=10) 한 번에 fit.
- **FL cells** (V6-Dyn-B-{FedAvg, FedProx, FedRep, Ditto, FedProto}) → 2-stage hierarchical *federated* KMeans (`src/fl/codebook_fl.py`).

Federated KMeans의 세 stage는 다음과 같습니다.

Stage 1에서 각 가구가 자기 train 윈도우의 h_g 위에 local KMeans++를 수행합니다. K_local=2 centroid가 만들어지며, raw h_g는 가구 내에 머물고 centroid 2개와 cluster sample count만 서버로 전송됩니다. 가구당 약 1KB.

Stage 2에서 서버가 114가구의 228개 local centroid를 모아 sample-count-weighted KMeans++로 다시 클러스터링해 32-entry global codebook C_global ∈ R^(32 × 64)을 만듭니다. 이 global codebook이 모든 가구로 broadcast됩니다.

Stage 3에서 각 가구가 자기 train 윈도우들을 global codebook으로 routing해 cluster assignment를 결정한 뒤, cluster별 forecast residual의 partial sum과 count를 서버로 업로드합니다. 서버는 cluster별로 합산해 cluster mean residual offsets ∈ R^(32 × 24)을 계산합니다. Individual residual은 서버에 노출되지 않으며 cluster-aggregated 값만 보입니다.

세 stage 모두 single-shot이며 라운드별 갱신이 없습니다. 가구당 총 통신량은 약 4.2KB.

## Cluster-wise Forecast Correction

Test 가구의 입력에 대해 forecast ŷ_base를 얻고, 입력의 h_g_test를 codebook과 매칭하여 cluster c*를 결정합니다. 그 cluster의 mean residual offset을 forecast에 더해 보정합니다.

ŷ_corr = ŷ_base + α_v0 · offsets[argmin_c ‖h_g_test − codebook[c]‖₂]

α_v0는 보정 강도를 조절하는 단일 hyperparameter이며, 본 연구는 α_v0 = 1.0을 default로 사용합니다. 같은 cluster에 속한 학습 가구들의 평균 forecast bias를 test 가구에 prior로 적용하는 역할을 합니다. Gaussian template 항 (v01 W5의 α_w1) 은 Phase 2의 codebook 기여를 isolate하기 위해 0으로 설정합니다.

## Codebook 모듈의 직교성

본 framework의 codebook은 특정 FL 알고리즘에 종속되지 않습니다. FedAvg, FedProx, FedRep, Ditto, FedProto backbone 어느 쪽 위에도 동일하게 적용 가능하며, Phase 1의 라운드 단위 학습을 그대로 둔 채 Phase 2 codebook을 추가하는 직교적 모듈 구조입니다. 후술 §5.1 결과에서 이 직교성이 수치로 확인됩니다.

---

# Experiments

## 데이터셋과 설정

UMass Smart* 데이터셋의 114가구 시간 단위 부하 데이터를 사용합니다 (`filter_valid_apartments(min_hours=7000)` 통과 가구). 각 가구를 chronological train 70% / val 10% / test 20%로 per-client 분할하며, 모든 가구가 학습에 참여합니다 (cold partition 없음). 입력 윈도우는 96시간, 예측 horizon은 24시간이며, 모든 결과는 seed 42, 123, 7의 3-seed mean ± std (ddof=1) 입니다.

비교군은 centralised pooled SGD upper bound 1종과 federated learning 5종, 그리고 각 backbone 위의 Phase 2 codebook stacking 6종으로 구성됩니다.

## Phase 1 — Round-Level FL 학습 동역학 (λ_aux = 0.3 default)

| Cell | val.PAPE | **test.PAPE** | HR@1 (test) | drift L2 | Upload (MiB) | wall (s) |
|---|---|---|---|---|---|---|
| V6-Dyn-A centralised | 66.33 ± 0.80 | **49.43 ± 0.36** | 20.81 ± 0.03 | 0 | 0 | 35 |
| FedAvg               | 81.62 ± 3.00 | 51.36 ± 0.61 | 13.46 ± 0.16 | 2.42 | 611  | 725 |
| FedProx (μ=0.01)     | 81.73 ± 2.35 | 51.40 ± 0.63 | 13.78 ± 0.18 | **1.71** | 611 | 2384 |
| FedRep (head_ep=1)   | 78.24 ± 1.79 | 51.36 ± 0.68 | 13.78 ± 0.62 | 2.20 | **502** | 897  |
| Ditto (λ=0.1)        | 84.49 ± 2.42 | 51.79 ± 0.47 | 13.84 ± 0.35 | 2.42 | 611  | 4410 |
| FedProto (K=32)      | 80.93 ± 2.95 | 51.50 ± 0.54 | 13.46 ± 0.34 | 2.43 | 629  | 1574 |

[F1b 삽입: round vs test PAPE trajectory]

**Centralised vs FL.** Centralised SGD가 test PAPE 49.43%를 달성하고, 가장 우수한 FL은 FedAvg와 FedRep의 51.36%입니다. 약 +2 PAPE 의 deficit이 다섯 알고리즘 전반에서 std ~0.6 내로 일관되게 나타나, 노이즈가 아닌 실제 gap임을 확인합니다.

**알고리즘 동등성.** 다섯 FL 알고리즘의 test PAPE는 51.36 ~ 51.79 구간에 모여 있어, 알고리즘 종류가 PAPE에 대해 *유의미한 discriminator가 아닙니다*. FedProx는 client drift를 절반 가까이 (2.42 → 1.71) 억제하지만 그 drift control이 PAPE 개선으로 전환되지 않는데, 이는 FedProx 2020이 보고한 "drift control ≠ accuracy gain on convex-ish workloads" 관찰과 일치합니다.

**통신 효율.** FedRep은 head를 broadcast하지 않아 총 502 MiB를 사용하며, 이는 FedAvg의 611 MiB 대비 18% 절감입니다. PAPE는 동등하므로 FedRep이 라운드 단위 학습에서 Pareto dominant choice입니다.

[F2 삽입: comm bytes vs val PAPE Pareto]

[F3 삽입: client drift L2 trajectory]

## Phase 1 — MAE-Only Ablation (λ_aux = 0)

다섯 FL 셀과 centralised 셀 모두에서 peak-aux loss term을 0으로 끄고 backbone-only로 재학습한 결과입니다.

| Cell | val.PAPE | **test.PAPE** | HR@1 (test) | drift L2 | wall (s) |
|---|---|---|---|---|---|
| V6-Dyn-A-MAEonly       | 64.94 ± 1.76 | 48.91 ± 0.70 | 20.97 ± 0.54 | 0   | 61   |
| FedAvg-MAEonly         | 80.94 ± 1.26 | 48.42 ± 0.37 | 15.68 ± 0.37 | 2.56 | 1086 |
| FedProx-MAEonly        | 78.41 ± 1.05 | 48.51 ± 0.03 | 15.86 ± 0.42 | **1.67** | 2361 |
| FedRep-MAEonly         | 84.85 ± 0.87 | 49.08 ± 0.50 | 15.27 ± 0.37 | 2.45 | 878  |
| **Ditto-MAEonly**      | 78.79 ± 1.55 | **48.28 ± 0.32** | **16.09 ± 0.69** | 2.56 | 3676 |
| FedProto-MAEonly       | 81.03 ± 1.05 | 48.49 ± 0.31 | 15.81 ± 0.71 | 2.57 | 751  |

[F4 삽입: round vs test PAPE — MAE-only]

**Negative result.** v01–v05 carry-over invariant인 peak-aux loss (λ_aux=0.3) 가 *라운드 단위 FL 학습에서는 PAPE를 악화시킵니다*. Centralised는 0.5 PAPE, FL 셀들은 2.3 ~ 3.5 PAPE 개선되며, HR@1 또한 동시에 0.2 ~ 2.4 points 향상됩니다. 즉 cold-zero-shot 환경 (v01) 에서 긍정적이던 peak-aux head가 라운드 단위 FL에서는 부정적입니다.

**메커니즘 해석.** Peak-aux head의 24-class hour-classification CE는 가구별 peak hour 분포가 매우 heterogeneous한데, FedAvg가 client gradient를 평균내면서 이 heterogeneous label signal이 dilute되는 것으로 보입니다. λ_aux = 0이면 이 noise가 제거되어 MAE signal이 깨끗하게 평균됩니다.

**FL 격차의 붕괴.** λ_aux=0.3 default에서 +2 PAPE이던 centralised-FL gap이 MAE-only에서는 −0.6 ~ +0.2로 collapse합니다. Ditto-MAEonly (48.28) 와 FedAvg-MAEonly (48.42) 가 centralised-MAEonly (48.91) 를 오히려 능가하는 결과는, "FL이 본질적으로 centralised보다 못하다"는 통념이 부분적으로 loss term의 artefact였음을 시사합니다.

## Phase 2 — 사후 Codebook Stacking (λ_aux = 0.3)

여섯 backbone 모두에 동일한 federated codebook (centralised는 pooled KMeans, FL은 2-stage federated KMeans) 을 사후 stacking한 결과입니다.

| Cell | test.PAPE BEFORE | **test.PAPE AFTER** | **ΔPAPE** | ΔHR@1 | ΔHR@2 | ΔMAE | ΔMSE(kW²) |
|---|---|---|---|---|---|---|---|
| V6-Dyn-A centralised | 49.43 ± 0.35 | **44.92 ± 0.14** | **−4.51 ± 0.21** | +0.78 | +0.97 | +0.0095 | −0.0183 |
| V6-Dyn-B-FedAvg      | 51.36 ± 0.63 | 45.92 ± 0.51 | −5.44 ± 0.15 | +0.43 | +0.72 | +0.0065 | −0.0202 |
| V6-Dyn-B-FedProx     | 51.42 ± 0.64 | 46.00 ± 0.45 | −5.42 ± 0.20 | +0.40 | +0.86 | +0.0067 | −0.0206 |
| V6-Dyn-B-FedRep      | 51.37 ± 0.66 | **45.77 ± 0.22** | −5.60 ± 0.45 | +0.41 | +0.74 | +0.0064 | −0.0209 |
| **V6-Dyn-B-Ditto**   | 51.80 ± 0.44 | 45.92 ± 0.26 | **−5.88 ± 0.27** | +0.17 | +0.47 | +0.0057 | **−0.0253** |
| V6-Dyn-B-FedProto    | 51.51 ± 0.56 | 45.89 ± 0.29 | −5.61 ± 0.33 | +0.77 | **+1.23** | +0.0062 | −0.0208 |

[F6 삽입: codebook lift on test PAPE — BEFORE/AFTER]

**Lift 보편성.** Centralised와 다섯 FL 셀 모두에서 4.5 ~ 5.9 PAPE의 개선이 일관되게 나타납니다. Lift 범위가 1.4 PAPE 이내, 셀별 std 0.15 ~ 0.45로 좁아 알고리즘 종류가 codebook 효과를 좌우하지 않습니다.

**FL deficit의 추가 closing.** Phase 1의 +2.06 PAPE gap (centralised 49.43 vs FL avg 51.49) 이 Phase 2에서 +0.98 PAPE (44.92 vs 45.90) 로 줄어듭니다. FL 셀이 centralised보다 *더 큰 codebook lift* 를 받는다는 점에서, codebook이 FL 학습이 under-specify한 peak-relevant structure를 회복시키는 역할을 한다고 해석할 수 있습니다.

**Trade-off.** 모든 셀이 ΔMAE +0.006 ~ +0.010 kW를 보입니다. Cluster-mean offset이 예측을 cluster-typical peak shape 쪽으로 끌어당기면서 non-peak hour의 MAE가 약간 악화되는, v01에서 보고된 W5 trade-off와 동일한 현상입니다. PAPE −5 vs MAE +0.007이라는 비율은 peak-prioritised application에서 유리한 trade입니다.

## Phase 2 — Federated Codebook 품질 vs Pooled Codebook 품질

| Codebook diagnostic | centralised (pooled) | FL avg (5 cells, federated) |
|---|---|---|
| utilization        | 1.000 ± 0.000 | 1.000 ± 0.000 |
| perplexity (M=32)  | 26.74 ± 0.95  | 26.18 ± 0.40  |
| n_empty_clusters   | 0             | 0             |
| k_max              | 2991 ± 406    | 2026 ± 180    |
| k_min              | 109 ± 45      | 46 ± 19       |

Federated 2-stage hierarchical KMeans가 pooled KMeans와 *수치적으로 동등한 quality* 의 codebook을 생산합니다. 두 codebook 모두 utilization 1.0과 81% of ideal perplexity를 달성하며, federated path는 raw h_g 가 가구를 떠나지 않는 프라이버시 계약을 0의 정확도 비용으로 만족합니다.

## MAEonly + Codebook — Negative Result의 지속

§5.1 결과에 §4.2 ablation을 결합한 추가 실험입니다. 만약 codebook이 peak-aux head가 만든 peak-aware h_g 구조에 의존한다면 MAEonly backbone에서는 lift가 줄거나 사라져야 합니다.

| Cell | AFTER default (λ_aux=0.3) | AFTER MAEonly (λ_aux=0) | MAEonly 더 우수 |
|---|---|---|---|
| centralised | 44.92 | **44.41** | −0.51 |
| FedAvg      | 45.92 | **44.59** | **−1.33** |
| FedProx     | 46.00 | **44.84** | −1.16 |
| FedRep      | 45.77 | 45.49     | −0.28 |
| **Ditto**   | 45.92 | **44.20** | **−1.72** |
| FedProto    | 45.89 | **44.54** | −1.35 |

[F8 또는 §5.3 표 시각화]

**Peak-aux negative result는 codebook으로 erase되지 않습니다.** 모든 셀에서 "λ_aux=0 + codebook"이 "λ_aux=0.3 + codebook"보다 *strictly* 더 낮은 test PAPE를 달성합니다. v06의 최종 권장 operating recipe는 따라서 **λ_aux = 0 + federated codebook stacking**이며, FL 셀 44.2 ~ 44.8% / centralised 44.4%의 test PAPE를 달성합니다.

## Codebook 보정 강도 α_v0 — Pareto Curve

α_v0를 {0.5, 1.0, 1.5, 2.0} 으로 sweep한 결과입니다.

| α_v0 | centralised PAPE / ΔMAE | FL avg PAPE / ΔMAE |
|---|---|---|
| 0.5  | 47.18 / +0.0018 | 48.66 / **−0.0004** |
| 1.0  | 44.92 / +0.0095 | 45.90 / +0.0064 |
| 1.5  | 42.79 / +0.023  | 43.37 / +0.019 |
| 2.0  | **40.78** / +0.040 | **41.21** / +0.037 |

[F7 삽입: α_v0 PAPE/MAE Pareto curve]

α_v0 = 0.5는 v01에서 문서화되지 않았던 새로운 Pareto point로, FL 셀에서 ΔMAE가 *오히려 negative* (개선) 이면서 ΔPAPE −3을 달성합니다. α_v0 = 1.0은 v01 carry-over operating point이며, α_v0 = 2.0은 centralised 40.78까지 (Phase 1 대비 17.5% relative reduction) 도달하지만 ΔMAE +0.04를 지불합니다. Trade-off는 monotonic이며, 응용의 PAPE/MAE 가중치에 따라 operating point를 선택할 수 있습니다.

## Federated Codebook K_local Sweep

Stage 1의 client local centroid 개수 K_local을 {1, 2, 4, 8}로 sweep한 결과입니다.

| Cell | K=1 | **K=2** (baseline) | K=4 | K=8 |
|---|---|---|---|---|
| FedAvg   | −4.36 | **−5.44** | −5.58 | −5.71 |
| FedProx  | −4.41 | **−5.42** | −5.53 | −5.76 |
| FedRep   | −4.42 | **−5.60** | −5.69 | −5.79 |
| Ditto    | −5.01 | **−5.88** | −6.00 | **−6.19** |
| FedProto | −4.51 | **−5.61** | −5.85 | −5.90 |

[F8 삽입: K_local sweep]

K=1 → K=2 step에서 약 +1.0 PAPE의 lift가 추가됩니다 (client mean h_g 한 점만으로는 intra-client peak diversity가 손실됨). K=2 → K=4, K=4 → K=8 step은 각각 ~+0.1 PAPE에 불과해, **v05 FedCB의 K_local = 2 선택이 v06의 라운드 단위 FL 프로토콜에서도 그대로 robust** 함을 확인합니다. K=8은 upload payload를 4배로 늘리지만 sub-noise gain만 제공하므로 Pareto dominated입니다.

## 통신 효율

Phase 1과 Phase 2의 통신 비용을 비교합니다.

| Method                                  | Per round (MiB) | Rounds | Total (MiB) |
|---|---|---|---|
| FedAvg backbone training (Phase 1)      | ~30.6           | 20     | 611         |
| FedRep backbone training (Phase 1)      | ~25.1           | 20     | 502         |
| Phase 2 federated codebook (K_local=2)  | n/a             | 1      | ~0.5 (≈4.2KB × 114 가구) |

Phase 2 federated codebook의 총 통신량은 Phase 1 FedAvg의 약 0.08%에 불과합니다. 사후 codebook 모듈은 학습 통신 비용에 사실상 추가 부담을 주지 않으면서 ΔPAPE −5 ~ −6의 추가 lift를 제공합니다.

---

# Conclusion

## 연구 정리

본 연구는 fully-federated 환경에서 가구별 피크 부하 예측을 두 phase로 분석했습니다. Phase 1에서 다섯 가지 FL 알고리즘을 동일 NBEATSx + peak-aux backbone 상에서 라운드 단위로 학습하고 비교했으며, Phase 2에서 학습된 여섯 backbone 위에 federated codebook을 사후 stacking했습니다.

## Contribution

첫째, **라운드 단위 FL 학습 동역학에서 peak-aux head loss가 PAPE를 악화시키는 negative result** 를 발견했습니다. v01–v05의 cold-zero-shot 환경에서는 positive contributor였던 λ_aux=0.3이 라운드 단위 FL에서는 +2 ~ +3.5 PAPE 손실을 야기합니다. MAE-only ablation이 centralised-FL gap을 +2 PAPE에서 −0.6 ~ +0.2로 collapse시킵니다.

둘째, **FL 알고리즘 선택은 PAPE의 유의미한 discriminator가 아닙니다.** 다섯 알고리즘이 0.5 PAPE 이내로 군집하며, 통신 비용 18% 절감을 제공하는 FedRep이 Pareto dominant입니다.

셋째, **사후 codebook stacking이 모든 backbone에 4.5 ~ 5.9 PAPE의 보편적 lift를 제공합니다.** Federated codebook이 pooled codebook과 numerical하게 동등한 quality를 달성하면서 raw h_g가 가구를 떠나지 않는 프라이버시 계약을 만족합니다. Phase 1의 +2.06 FL gap이 Phase 2 적용 후 +0.98로 줄어, codebook은 FL 학습과 직교적 contributor입니다.

넷째, **권장 operating recipe는 λ_aux = 0 + federated codebook stacking** 이며, FL 셀 44.2 ~ 44.8% / centralised 44.4%의 test PAPE를 달성합니다. 이는 default Phase 1 (λ_aux=0.3) 대비 ~5 PAPE의 absolute improvement이며, Phase 2 통신 추가 비용은 Phase 1의 0.08%에 불과합니다.

다섯째, **codebook hyperparameter는 robust합니다.** α_v0는 MAE-zero-cost (0.5) ~ PAPE-extreme (2.0) 의 informative Pareto curve를 그리며, K_local=2는 v05 FedCB의 선택이 라운드 단위 FL 프로토콜에서도 그대로 유효함을 재확인합니다.

## Limitation

본 연구는 다음 한계를 가집니다.

첫째, **검증 데이터셋이 UMass Smart* 로 한정** 되어 있으며, 한국 가정용 부하 데이터에서의 일반화는 추가 검증이 필요합니다. 가구간 amplitude 분포나 trend 강도 같은 도메인 특성이 한미 간 다를 수 있어, 한전 AMI 데이터 등을 통한 국내 검증을 우선 과제로 두고 있습니다.

둘째, **compute budget mismatch** 가 있습니다. FL 셀은 R=20 × E=40 = 800 epoch-equivalent를 사용한 반면 centralised는 40 epoch만 사용했습니다. 따라서 centralised upper bound와 FL의 절대 비교는 약한 claim이며, FL 알고리즘 간 상대 비교 (round-vs-PAPE shape, drift trajectory) 가 더 신뢰할 수 있는 결론입니다.

셋째, **λ_aux sweep과 (E, R) budget sweep은 v07로 이연** 했습니다. Peak-aux negative result의 메커니즘 (heterogeneous label dilution vs λ 자체의 mis-tune) 을 discriminate하려면 λ_aux ∈ {0, 0.05, 0.1, 0.2, 0.3} sweep이 필요하며, FL 알고리즘 동등성이 budget 변화에서도 유지되는지는 v07 (E, R) sweep으로 확인됩니다.

넷째, **라운드 trajectory codebook은 후속 연구로 남깁니다.** Codebook lift가 backbone 라운드 수에 따라 monotonic하게 자라는지 또는 조기에 plateau하는지는 intermediate checkpointing이 필요한 별도 실험입니다.

이상으로 발표를 마치겠습니다. 감사합니다.
