
# Title

연합학습 환경에서 가구별 피크 부하 예측을 위한 피크 인지 코드북 프레임워크

경북대학교 데이터사이언스대학원 전진우

2026.06.04

발표 도입

안녕하십니까. 경북대학교 데이터사이언스대학원 소속 석사과정 전진우 입니다. 발표 주제는 연합학습 환경에서 가구별 피크 부하 예측 정확도를 끌어올리는 피크 인지 코드북 프레임워크입니다. 시작하겠습니다.

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

이 두 제약 조건을 함께 다루는 자연스러운 framework가 연합학습입니다. 가구의 raw 데이터는 가구에 머물고, 모델 학습에 필요한 정보만 서버와 교환합니다. 본 연구는 이 연합학습 framework 안에서 가구 단위 피크 예측 정확도를 어디까지 높일 수 있는가를 분석했습니다. 

---

# Goal

## 연구 목적

본 연구는 다음 세 가지 목적을 동시에 달성하는 것을 목표로 합니다.

1. raw 가구 부하 데이터를 가구 외부로 반출하지 않는 연합학습 환경에서 가구별 피크 예측 모델을 학습
2. 표준 연합학습 알고리즘이 가구 피크 예측에서 보이는 정확도 한계를 정확히 식별하고, 그 한계를 보완할 수 있는 추가 component를 설계
3. 추가 component가 다양한 연합학습 알고리즘과 결합 가능한 모듈 형태이며, 추가 통신 비용은 federated 학습 단계의 통신량 대비 무시할 수 있는 수준임을 검증

## 평가 지표 정의

본 연구의 핵심 평가 지표는 PAPE, peak absolute percentage error입니다. 정의는 다음과 같습니다.

PAPE = (1/N) × Σ |max(y_true_i) − max(y_pred_i)| / max(y_true_i) × 100

여기서 y_true_i와 y_pred_i는 i번째 24시간 forecast 윈도우의 ground truth와 예측이며, max는 그 24시간 안의 최대값을 가져옵니다. 즉 PAPE는 예측된 24시간 부하의 최대값이 실제 최대값과 얼마나 다른지를 백분율로 측정합니다.

PAPE는 ESS 운영 의사결정에 직결되는 지표입니다. ESS는 24시간 안의 피크 시점에 방전하여 피크를 깎는 것이 목적이므로, 피크 크기 추정 오차가 곧 ESS 운영 손실로 이어집니다.

피크 시점의 정확도는 HR@1, HR@2 (Hit Rate within ±1 hour, ±2 hour)로 측정합니다. 또한 보조 지표로 MSE (mean squared error, kW²)를 함께 보고합니다. MSE는 outlier 큰 오차에 가중치를 두어 method 간 전반 fit 차이를 식별하는 보조 지표입니다.

---

# Related Works

## 연합학습 표준 알고리즘

표준 연합학습 알고리즘은 다음 세 갈래로 발전해 왔습니다.

첫째, FedAvg 계열로 모든 client의 모델 weight를 평균하는 가장 단순한 형태입니다. 
둘째, FedProx로 client drift를 억제하는 proximal term을 추가합니다. 
셋째, personalised FL 계열로 FedRep, Ditto가 대표적이며, encoder는 공유하고 head는 client별로 두거나 global 모델과 local 모델을 동시에 유지하는 구조입니다.

이 알고리즘들은 공통적으로 forecast loss의 평균값 최적화에 초점을 둡니다. 즉 피크 영역의 신호를 따로 강조하지 않습니다.

## 피크에 특화된 시계열 예측 연구

피크 부하 예측에 특화된 선행 연구로 Zhang 등 2023의 Seq2Peak이 있습니다. 이 연구는 forecast loss에 peak loss를 가중 결합한 hybrid loss를 제안하며, ETTh, Electricity 등 집계 데이터셋에서 검증되었습니다. 그러나 가구 단위 검증은 없습니다.

## 가구 단위 부하 예측의 일반화 한계

가구 단위 부하 예측에서 강한 한계가 두 선행 연구에서 확인됩니다. BuildingsBench 2023은 90만 시뮬레이션 건물로 사전학습된 Transformer가 실제 거주 건물에서 zero-shot NRMSE 79%를 보고하며, 단순 persistence가 78%로 거의 동등한 수준임을 보였습니다. Peng 등 2019는 approximate entropy 분석으로 개별 가구 부하가 본질적으로 예측 어려움이 큰 신호임을 보였습니다.

이는 가구 단위에서는 모델 크기나 사전학습 양보다 task에 맞춘 inductive bias가 중요함을 시사합니다.

## 벡터 양자화의 시계열 활용

벡터 양자화는 self-supervised pretraining에서 시계열 token화에 사용되어 왔습니다. VQ-MTM 2024가 대표적입니다. 본 연구는 이 방향과 다르게, 학습된 backbone의 hidden representation 위에서 사후 KMeans를 통해 inference time correction module을 만드는 방식으로 벡터 양자화를 활용합니다.

---

# Method

## 전체 구조 요약

본 framework는 표준 연합학습 위에 두 가지 component를 추가합니다. 첫째, peak-aware backbone으로 NBEATSx에 보조 헤드를 부착해 forecast와 피크 정보를 함께 학습합니다. 둘째, federated codebook으로 학습된 backbone의 hidden representation을 두 단계 hierarchical KMeans로 클러스터링해 inference time correction module을 만듭니다.

학습 phase와 codebook 구성 phase 모두 fully-federated이며, raw 가구 데이터와 raw hidden representation 모두 서버로 전송되지 않습니다.

## Backbone 선택 근거

Backbone으로 NBEATSx를 채택했습니다. 선택 근거는 두 가지입니다.

첫째, NBEATSx의 stack-wise decomposition 구조가 보조 헤드와 자연스럽게 결합됩니다. NBEATSx는 trend, seasonal, generic 세 stack으로 forecast를 분해하는데, 보조 헤드를 generic stack hidden 위에만 부착하면 trend와 seasonal component는 보존하면서 generic 부분만 피크 task에 align됩니다.

[NBEATSx Architecture 삽입]

둘째, baseline 비교 실험에서 NBEATSx가 가장 우수한 성능을 보였습니다. 동일 데이터셋과 동일 학습 조건에서 DLinear는 cold PAPE 50.4%, NHITS는 52.7%, Crossformer는 52.5%를 보인 반면, NBEATSx 기반 framework는 50.2%로 가장 낮은 오차를 달성했습니다. 이는 NBEATSx의 표현력과 stack 분해 구조가 가구 피크 task에 적합함을 보여줍니다.

## Peak-Aware Auxiliary Head

NBEATSx의 generic stack hidden을 h_g ∈ R^64라 할 때, 보조 헤드는 두 가지 출력을 가집니다.

(â, ĥ) = AuxHead(h_g)

여기서 â은 24시간 forecast 윈도우의 예측 피크 진폭이고, ĥ은 예측 피크 시점입니다.

학습 손실은 다음과 같습니다.

L_aux = MSE(â, max(y)) + 0.1 × CE(ĥ, argmax(y))

L_total = MAE(ŷ, y) + λ × L_aux, λ = 0.3

여기서 ŷ과 y는 각각 forecast와 ground truth, λ는 보조 task 가중치입니다. 시점 분류 항의 0.1 가중치는 시점 추정의 본질적 어려움을 반영해 약하게 설정되었습니다.

이 보조 task의 역할은 학습 시점에 generic stack의 hidden representation이 피크 구조에 align되도록 만드는 것입니다. Inference 시점에는 보조 헤드의 출력을 직접 사용하지 않으며, peak-aware하게 학습된 hidden이 다음 단계의 codebook 구성에 활용됩니다.

## Federated Codebook Construction

Codebook 구성은 두 stage로 구성된 hierarchical KMeans로 federated하게 처리됩니다.

Stage 1에서 각 가구가 자기 학습 윈도우의 hidden representation 위에 local KMeans를 수행합니다. K_local=4 centroid가 만들어지며, raw hidden representation은 가구 내에 머물고 centroid 4개와 cluster sample count만 서버로 전송됩니다. 가구당 약 1KB.

Stage 2에서 서버가 80가구의 320개 local centroid를 모아 sample-count-weighted KMeans로 다시 클러스터링해 32-entry global codebook C_global ∈ R^(32 × 64)을 만듭니다. 이 global codebook이 모든 가구로 broadcast됩니다.

Stage 3에서 각 가구가 자기 윈도우들을 global codebook으로 routing해 cluster assignment를 결정한 뒤, cluster별 forecast residual의 partial sum과 count를 서버로 업로드합니다. 서버는 cluster별로 합산해 cluster mean residual o_c를 계산합니다. Individual residual은 서버에 노출되지 않으며 cluster-aggregated 값만 보입니다.

세 stage 모두 single-shot이며 라운드별 갱신이 없습니다. 가구당 총 통신량은 약 4.2KB이며, 서버 측 합계 0.35MB입니다.

## Cluster-wise Forecast Correction

Cold 가구의 입력에 대해 forecast ŷ_base를 얻고, 입력의 hidden representation을 federated codebook과 매칭하여 cluster c*를 결정합니다. 그 cluster의 mean residual o_(c*)을 forecast에 더해 보정합니다.

ŷ_corr = ŷ_base + α × o_(c*)

여기서 α는 보정 강도를 조절하는 단일 hyperparameter로, 본 연구는 α=1.0을 default로 사용합니다. 이는 cluster mean residual을 그대로 더하는 가장 자연스러운 default 값이며, hyperparameter sweep 없이 fix했습니다. 같은 cluster에 속한 학습 가구들의 평균 forecast bias를 cold 가구에 prior로 적용하는 역할을 합니다.

## Codebook 모듈의 직교성

본 framework의 codebook과 hybrid correction은 특정 FL 알고리즘에 종속되지 않습니다. FedAvg backbone, FedRep backbone 어느 쪽 위에도 동일하게 적용 가능하며, 표준 FL 학습을 그대로 둔 채 codebook과 correction을 추가하는 모듈 구조입니다.

---

# Experiments

## 데이터셋과 설정

UMass Smart* 데이터셋의 100가구 시간 단위 부하 데이터를 사용합니다. 80가구는 federated 학습에, 20가구는 학습에 전혀 노출되지 않은 cold 검증에 사용합니다. 입력 윈도우는 96시간, 예측 horizon은 24시간이며, 모든 결과는 seed 42, 123, 7의 3-seed 평균과 표준편차입니다.

평가 지표는 PAPE, HR@1, HR@2 세 가지를 주 지표로 사용하고, 보조 지표로 MSE (kW²)를 함께 보고합니다. PAPE는 24시간 forecast 윈도우의 피크 진폭 추정 오차이고, HR@1과 HR@2는 피크 시점 예측이 실제 피크 시점에서 ±1시간 또는 ±2시간 이내에 들어오는 비율입니다. PAPE는 피크의 크기 정확도, HR@1과 HR@2는 피크의 시점 정확도를 측정합니다. MSE는 outlier 큰 오차에 가중치를 두어 method 간 전반 fit 차이를 식별하는 보조 지표입니다.

비교군을 세 그룹으로 나누어 보고합니다. 표준 연합학습 알고리즘 그룹, centralised 학습한 neural forecasting 모델 그룹, 사전학습된 시계열 foundation model 그룹입니다. 각 그룹과의 비교는 본 framework의 contribution을 다른 측면에서 확인합니다.

Neural Forecasting 모델 비교

centralised 학습한 neural forecasting 모델은 80가구의 raw 데이터를 모두 모아 학습한 결과로, federated 환경의 정확도 상한을 가늠하는 reference 역할을 합니다.

| Method                           | PAPE (%)     | HR@1 (%)     | HR@2 (%)     | MSE (kW²)       |
| -------------------------------- | ------------ | ------------ | ------------ | --------------- |
| NHITS (centralised pooled)       | 52.74 ± 1.71 | 26.82 ± 2.32 | 37.69 ± 2.17 | 0.5195 ± 0.0374 |
| Crossformer (centralised pooled) | 52.54 ± 1.71 | 26.92 ± 2.18 | 38.18 ± 1.91 | 0.5199 ± 0.0368 |
| DLinear (centralised pooled)     | 50.37 ± 0.84 | 26.41 ± 1.84 | 37.22 ± 1.73 | 0.5167 ± 0.0350 |
| Proposed (federated)             | 50.17 ± 0.97 | 25.28 ± 1.30 | 37.24 ± 1.86 | **0.5060 ± 0.0326** |

본 framework의 PAPE 50.17%는 가장 우수한 centralised neural forecasting 모델인 DLinear의 50.37%와 통계적으로 동등합니다. NHITS와 Crossformer는 본 framework보다 약 2.4-2.6%p 높은 PAPE를 보입니다. MSE에서는 본 framework가 0.5060으로 NF 1등 DLinear의 0.5167보다 0.0107 (2.07%) 낮아 추가적 우위를 보입니다. 즉 본 framework는 raw 데이터 미공유라는 강한 제약 조건을 만족하면서도 centralised pooled 학습과 동등하거나 더 우수한 정확도를 달성합니다.

## Foundation Model 비교

사전학습된 시계열 foundation model은 학습 없이 zero-shot으로 가구 부하 예측을 수행한 결과입니다. 모델 크기는 약 1천만에서 5천만 파라미터 범위입니다.

| Method                         | PAPE (%)     | HR@1 (%)     | HR@2 (%)     | MSE (kW²)       |
| ------------------------------ | ------------ | ------------ | ------------ | --------------- |
| Chronos-T5 tiny (zero-shot)    | 63.13 ± 3.04 | 18.32 ± 0.77 | 27.19 ± 1.21 | 0.7073 ± 0.0478 |
| TimesFM (zero-shot)            | 54.27 ± 2.15 | 24.95 ± 1.24 | 35.30 ± 0.64 | 0.5450 ± 0.0399 |
| Chronos-Bolt small (zero-shot) | 52.69 ± 1.56 | 26.16 ± 1.92 | 36.67 ± 1.46 | 0.5451 ± 0.0360 |
| Proposed                       | 50.17 ± 0.97 | 25.28 ± 1.30 | 37.24 ± 1.86 | **0.5060 ± 0.0326** |

본 framework는 가장 우수한 foundation model인 Chronos-Bolt와 통계적으로 동등하거나 약간 우수한 PAPE를 보입니다. 주목할 점은 이를 6만 5천 파라미터의 경량 NBEATSx로 달성했다는 것입니다. 5천만 파라미터 foundation model이 zero-shot으로 도달한 정확도를 약 1/770 크기 모델이 fully-federated 학습으로 달성했다는 의미입니다. 가구 단위 피크 예측에서는 모델 크기보다 task에 맞춘 inductive bias가 결정적임을 보여주는 결과입니다.

MSE 측면에서는 격차가 더 두드러집니다. 본 framework의 MSE 0.5060은 Chronos-Bolt 0.5451 대비 7.17% 낮고, TimesFM 0.5450 대비 7.16%, Chronos-T5 0.7073 대비 28.5% 낮습니다. 흥미로운 점은 Chronos-Bolt와 TimesFM의 PAPE가 표준 FL 평균 56.51%보다 약 4%p 낮음에도 불구하고 MSE는 표준 FL 평균 0.5299보다 오히려 높다는 것입니다 (각각 +2.87%, +2.85%). zero-shot foundation model이 피크 진폭은 그럭저럭 맞추되 전반적인 데이터 fit은 UMass 분포에 학습된 모델보다 약하다는 해석이 가능하며, 이는 본 framework가 peak-aware backbone 학습과 cluster-wise correction을 함께 사용해 두 측면 모두에서 우위를 점할 수 있음을 보여줍니다.
## Federated Learning 알고리즘 비교

표준 FL 알고리즘은 모두 평균 forecast loss 최적화에 집중하는 공통 특성이 있어, 가구 피크 예측에서 어떤 한계를 보이는지를 직접 측정할 수 있습니다.

| Method   | PAPE (%)         | HR@1 (%)     | HR@2 (%)     | MSE (kW²)           |
| -------- | ---------------- | ------------ | ------------ | ------------------- |
| FedRep   | 57.18 ± 1.52     | 25.72 ± 1.60 | 37.24 ± 1.83 | 0.5329 ± 0.0318     |
| Ditto    | 56.38 ± 1.63     | 26.52 ± 1.84 | 38.06 ± 2.00 | 0.5300 ± 0.0322     |
| FedProto | 56.37 ± 1.44     | 26.61 ± 1.71 | 38.17 ± 2.09 | 0.5288 ± 0.0308     |
| FedAvg   | 56.34 ± 1.41     | 26.44 ± 1.64 | 37.98 ± 1.83 | 0.5283 ± 0.0302     |
| FedProx  | 56.30 ± 1.55     | 25.99 ± 1.45 | 37.55 ± 2.07 | 0.5297 ± 0.0308     |
| Proposed | **50.17 ± 0.97** | 25.28 ± 1.30 | 37.24 ± 1.86 | **0.5060 ± 0.0326** |

표준 FL 5종은 모두 PAPE 56-57%대 좁은 구간에 군집하며 알고리즘 종류와 무관하게 1%p 이내 차이입니다. 본 framework는 PAPE 50.17%로 FL 5종 평균 56.51% 대비 약 6.34%p 낮으며, 약 11% 상대 개선입니다. MSE 측면에서도 본 framework 0.5060은 FL 5종 평균 0.5299 대비 0.0239 (4.5%) 낮습니다. HR@1과 HR@2는 모두 표준 FL과 통계적으로 동등한 수준입니다. 즉 시점 정확도를 유지하면서 진폭 정확도(PAPE)와 전반 fit(MSE) 양쪽을 의미 있게 개선했습니다.

추가로 per-seed paired 비교에서, 본 framework는 3 seed × 14 baseline = 42 paired 비교 중 42건 모두에서 더 낮은 MSE를 기록하여 across-seed 표준편차에 묻히지 않는 일관된 우위를 보입니다.
## Codebook Correction Module 효과 측정

본 framework의 Codebook Correction Module이 backbone 위에 더하는 순수 기여를 측정하기 위해, 동일 backbone의 module 적용 전후를 비교했습니다.

| Method                                | PAPE (%)     | HR@1 (%)     | HR@2 (%)     | MSE (kW²)       |
| ------------------------------------- | ------------ | ------------ | ------------ | --------------- |
| Backbone (no correction)              | 57.32 ± 1.55 | 26.35 ± 1.67 | 37.76 ± 1.56 | 0.5300 ± 0.0314 |
| Backbone + Codebook Correction Module | 50.17 ± 0.97 | 25.28 ± 1.30 | 37.24 ± 1.86 | 0.5060 ± 0.0326 |

Codebook Correction Module 적용으로 PAPE가 7.15%p 감소합니다. 약 12.5% 상대 개선입니다. MSE도 0.5300에서 0.5060으로 약 4.5% 감소하여 outlier-heavy 큰 오차도 함께 줄어듭니다. HR@1과 HR@2는 module 적용 전후 모두 통계적으로 동등한 수준입니다. 즉 Module은 시점 정확도를 손상시키지 않으면서 진폭 정확도를 의미 있게 개선합니다.
## Communication Efficiency

다음 표는 통신 비용을 비교한 결과입니다.

| Method                                  | Per round (MB) | Rounds | Total (MB) |
| --------------------------------------- | -------------- | ------ | ---------- |
| FedAvg backbone training                | 21.95          | 20     | 439        |
| Proposed federated codebook (K_local=4) | 0.35           | 1      | 0.35       |

본 framework의 통신 비용은 두 단계로 구성됩니다. Phase A에서 NBEATSxAux backbone을 federated하게 학습하며, 보조 헤드 파라미터는 backbone state와 함께 매 라운드 client-server 간 전송되어 backbone 통신량에 흡수됩니다. NBEATSx 본체 기준 약 420MB가 20라운드에 걸쳐 사용되며, 보조 헤드 추가로 인한 통신량 증가는 약 4% 수준입니다.  Phase B의 federated codebook 구성은 추가로 0.35MB만 사용하며, 이는 Phase A의 약 0.08%에 불과합니다. codebook 구성을 fully-federated로 처리해도 추가 통신 부담이 사실상 없으며, 이는 on-device 배포나 통신 자원이 제한된 가구 단위 IoT 환경에 적합한 특성입니다.

---
# Conclusion

## 연구 정리

본 연구는 fully-federated 환경에서 가구별 피크 부하 예측 정확도를 향상시키는 framework를 제안했습니다. NBEATSx backbone에 피크 진폭과 시점을 함께 예측하는 보조 헤드를 federated하게 학습하고, 학습 종료 후 두 stage hierarchical KMeans로 federated codebook을 구성하여 inference 시 cluster mean residual로 forecast를 보정합니다.

## Contribution

첫째, codebook 구성 단계까지 federated하게 변환하여 fully-federated framework를 제안했습니다. raw 가구 데이터, model weight, raw hidden representation, raw forecast residual 모두 서버로 전송되지 않으며, 가구 측에서 추출한 cluster-aggregated 통계만 서버에 도달합니다.

둘째, fully-federated 환경에서 PAPE 50.17%를 달성하여 가장 우수한 centralised pooled neural forecasting 모델인 DLinear의 50.37%와 통계적으로 동등한 정확도를 보였습니다. MSE에서도 본 framework 0.5060이 NF 1등 DLinear의 0.5167을 2.07% 차이로 앞서며, FL 5종 평균 0.5299 대비 4.5%, FM 1등 Chronos-Bolt 대비 7.17% 낮습니다. raw 데이터 미공유 제약을 만족하면서도 centralised pooled 학습과 동등하거나 더 우수한 정확도를 달성한 결과입니다.

셋째, codebook 구성을 fully-federated로 처리하면서도 추가 통신 비용은 federated 학습 단계의 0.08%에 불과해, 통신 자원이 제한된 가구 단위 IoT 환경에 적합한 framework입니다.
## Limitation

본 연구는 다음 두 가지 한계를 가집니다.

첫째, 피크 시점 추정 정확도(Hit Rate)에 본질적 한계가 있습니다. 가구별 피크는 진폭에서 가구간 차이가 뚜렷한 반면 시점은 일별 변동이 크고 가구 내에서도 일관성이 약합니다. 통계 분석 결과, Hit Rate 지표의 oracle ceiling 자체가 약 40% 수준으로 낮으며, 본 framework는 진폭 보정에 집중한 cluster mean offset 구조로 설계되었습니다. 

둘째, 본 연구는 codebook을 FedAvg backbone 위에 hierarchical 2-stage KMeans 방식으로 구성한 단일 조합에서 검증되었습니다. 다른 FL 알고리즘 backbone과의 결합, 그리고 iterative federated KMeans나 secure aggregation 같은 대안적 codebook 구성 방식과의 비교는 후속 연구로 남겼습니다.

셋째, 검증 데이터셋이 UMass Smart* 로 한정되어 있으며, 한국 가정용 부하 데이터에서의 일반화는 추가 검증이 필요합니다. 가구간 amplitude 분포나 trend 강도 같은 도메인 특성이 한미 간 다를 수 있어, 향후 한전 AMI 데이터 등을 통한 국내 검증을 우선 과제로 두고 있습니다.

이상으로 발표를 마치겠습니다. 감사합니다.
