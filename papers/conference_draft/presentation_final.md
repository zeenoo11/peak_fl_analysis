# 1. Introduction
## Abstract
Accurate household peak load forecasting is important for demand-side management and grid operation, yet it remains challenging due to highly volatile and heterogeneous consumption patterns. Federated learning (FL) enables collaborative training without centralizing privacy-sensitive consumption data, but conventional FL methods often underfit rare and sharp peak events, particularly for households with limited historical observations. 
## 키워드
- 연합학습, 보안, collaborative training

- 전력망 관리의 수요 측면에서 정확한 가구 부하의 피크 예측은 중요
- 변동성 크고 다종성의 소비 패턴을 가져 어려움
- 연합학습은 민감 정보 없이 중앙 집중 학습이 가능하지만, 기존 FL은 희소하고 뾰족한 Peak 지점에 underfit 해짐
- 제한적인 관측 영역에서도 특히 어려움

## 1-1. 가구 단위 피크 예측의 중요성

### 전력 그리드 관리에서 피크 부하의 의미

- 전력 계통 운영에서는 발전과 소비의 실시간 균형을 맞추는 것이 필수적이며, 최대 부하(피크)의 크기와 발생 시점이 결정적인 변수로 작용합니다.
- 최대 부하 시점에 발전 용량이 부족할 경우, 대규모 정전이나 고비용 비상 발전기 가동으로 직결됩니다.
- 전력 계통의 예비력 확보 및 송배전 설비 용량 설계는 모두 이 피크 부하를 기준으로 결정됩니다.
### 피크 부하 기반의 핵심 의사결정

- 발전기 기동정지 계획(Unit Commitment): 피크 시점에 맞춰 어떤 발전 자원을 어떤 순서로 동원할지 결정합니다.
- ESS 충방전 스케줄링: 피크 부하를 효과적으로 절감(Peak Shaving)하기 위한 최적의 방전 타이밍을 설정합니다.
- 송배전 설비 증설 계획: 변압기 및 송전 선로의 물리적 용량을 산정하는 기준이 됩니다.
### 가구 단위 예측으로의 패러다임 전환 이유

- 기존의 전통적인 방식은 시·군·구 또는 변전소 단위의 대규모 집계 부하를 대상으로 예측을 수행해 왔습니다.
- 최근 분산 전원(DER), V2G(Vehicle-to-Grid), 가정용 ESS 등의 보급이 확대되면서 개별 가구 단위의 전력 정보와 예측 중요성이 급격히 커지고 있습니다.

### 가구 단위 피크 예측의 구체적 활용 방안

- 가정용 ESS 및 전기차(EV) 충전 스케줄의 자동 최적화를 통해 개별 수용가의 에너지 비용을 절감합니다.
- 수요반응(DR) 인센티브 설계 시, 피크 감축에 실질적으로 기여한 가구를 정확히 식별하고 합리적인 보상을 제공합니다.
- 배전망 운영 측면에서 변압기 단위의 부하 집계 정확도를 향상시켜 국소적 과부하와 설비 고장을 예방합니다.
## 1-2. Trend, Seasonality 부족한 개별 가구의 Peak 정보

- 각 가구의 Peak 시간대는 가구별로 일정할거라 생각하지만, 실제 데이터에선 달랐음. 
	- 연간, 분기 단위 시점에선 Trend 정보는 존재함 (여름, 겨울 부하 큰 것)
	- 하지만 일주일, 한 달 시점에선 trend, seasonal 정보가 희소
	- = 개별 가구는 날마다 전력 사용 시간대를 크게 바꾼다 (Variability)
	- 
	- 오히려 가구 간 정보를 수집했을 때 클러스터 간 유사성이 더 크다[^1]
- 전력망 관리 관점에서의 Peak 지점은 하루 최대 부하의 시점과 크기 
- 
# 2. Goal 

> 키워드 위주로 핵심 목표를 설정하기 

- **G1 — RoundFL (Round-level Federated Learning)**: 모든 가구(114)가 매 라운드 학습에 참여하고, 각 가구의 self test 구간에서 라운드 단위로 peak 정확도를 추적. cold partition을 폐기하고 FedAvg / FedProx / FedRep / Ditto / FedProto를 동일 프로토콜로 비교.
- **G2 — RoundCB (Round-wise federated Codebook)**: 매 라운드 종료 시 backbone hidden `h_g`로부터 federated codebook을 구성하고, cluster 평균 잔차(Offset)로 예측을 보정. backbone forward를 거치지 않아 **모든 FL 알고리즘에 직교적**으로 적용.
- **G3 — Efficiency**: 큰 모델·추가 학습 없이, task-aligned inductive bias + inference-time personalization 만으로 standard FL 대비 peak 정확도를 향상.

# 3. Method
## Abstract
We propose a peak-aware FL framework for household load forecasting that combines representation-level peak learning with codebook-based inference-time personalization. 
First, we introduce a peak-aware NBEATSx backbone with an auxiliary objective for predicting peak amplitude and timing, encouraging hidden representations to encode peak-related dynamics during federated training. 
Second, we construct a fully federated hierarchical codebook from backbone hidden representations using two-stage KMeans and use cluster-wise mean residuals to correct forecasts for cold households at inference time. 

RoundCB는 세 모듈로 구성됩니다. 
1. NBEATSx Residual 분해로 Representation Extraction
2. 2-stage Federate Aggregation (Codebook)
3. 평균 예측 잔차(Offset)으로 예측 보정 

Codebook의 큰 흐름으로는 R(Representation formation), A(Aggregation), C(Correction)의 세 가지 메커니즘으로 분석해왔다. 여기서 사용할 RoundCB는 A, C에 초점을 두었다. 

## 3-1. Representation-Level Peak Infomation

>a peak-aware NBEATSx backbone with an auxiliary objective for predicting peak amplitude and timing, encouraging hidden representations to encode peak-related dynamics during federated training
### Residual Extraction from NBEATS Stack

- Backbone: **NBEATSx** (Olivares et al., 2023) — `MinimalNBEATSx`의 3-stack (trend / seasonal / generic) 구조.
- **Doubly-residual stacking**: 각 stack이 backcast로 입력 잔차를 제거(`residual ← residual − backcast`)하고 forecast를 누적.
- 잔차 분해의 **마지막 단계인 generic stack의 hidden vector `h_g ∈ ℝ⁶⁴`** 가 peak-relevant dynamics를 담는 표현이며, 이것이 codebook의 입력이 됨.
- (R/A/C 중 **R축**): RoundCB는 representation을 별도로 shaping하지 않고 backbone이 만든 `h_g`를 **그대로** 사용 — codebook이 backbone forward를 거치지 않으므로 commitment loss가 없음.

## 3-2. Federate Latent Learning with Codebook

> Second, we construct a fully federated hierarchical codebook from backbone hidden representations using two-stage KMeans and use cluster-wise mean residuals to correct forecasts for cold households at inference time. 

핵심 컨셉: RoundCB
### Codebook 3대 요소 R, A, C 기준 설명

1. **Representation Formation** - 없음; 그대로 Residual을 사용
	1. 매 라운드 종료 시 `hidden h_g` 추출
	2. Residual Decomposition의 마지막 단계
2. **Aggregate** — 2-stage hierarchical **federated KMeans**
	1. Stage-1 (client): 각 가구가 자기 `h_g`에 KMeans++(K_local=2) → `(centroid, count)`만 업로드. raw `h_g`는 가구를 벗어나지 않음.
	2. Stage-2 (server): 업로드된 centroid들에 **count-weighted** KMeans++(M=32) → 단일 `(32, 64)` 글로벌 codebook을 브로드캐스트.
	3. 이 count-weighted aggregation은 **FedProto**(Tan et al., 2022)의 per-class prototype 집계를 codebook으로 일반화한 것.
3. **Correction** — Cluster-Mean Offset (CMO)
	1. cluster별 평균 training 잔차 `offset ∈ ℝ^{32×24}` 를 federated하게 산출 (cluster별 잔차합/count만 업로드).
	2. 추론 시 보정: `ŷ_corr = ŷ_base + α · offset[c*]`,  `c* = argmin_c ‖h_g − codebook[c]‖₂`,  α=1.0.


강점
- **Backbone-agnostic**: forward가 codebook을 거치지 않아 어떤 FL backbone(FedAvg~FedProto)에도 그대로 얹힘 → 5개 알고리즘 전부 −5.7 ~ −6.5 PAPE lift (보편성).
- **Privacy-preserving**: raw `h_g` 미전송 (centroid·count·잔차합만 업로드). federated codebook이 pooled KMeans와 동등 품질 (utilization 1.0, perplexity ≈ 26).
- **No extra training**: backbone freeze 후 post-hoc 구성 — 추가 backprop 없음.

## 3-3. Inference-time Personalization

- cold-start 없이, 각 가구 test window의 `h_g`를 글로벌 codebook에 **1-NN routing** → 해당 cluster의 offset으로 개인화 보정.
- 학습 재개·파라미터 업데이트 불필요 → **추론 시점(inference-time) personalization**.
- **α_v0 Pareto** (보정 강도): α=0.5 (MAE-zero-cost, ΔPAPE≈−3, ΔMAE≈0) ~ α=1.0 (기본 운영점) ~ α=2.0 (PAPE-extreme, ΔPAPE≈−8.6, ΔMAE+0.04). 응용의 peak-vs-MAE 가중에 따라 운영점 선택.

# 4. Experiments

~~4-1. Data~~ -> Framework 에서 다 설명하기 
## 4-1. Round-Level Federate Learning 

- Setup: UMass Smart* 2016, 114 가구, per-client 70/10/20 (chronological), per-apt z-norm(train only). R=20 rounds, 5 FL 알고리즘, seeds {42, 123, 7} (mean ± std).
- **알고리즘 등가성**: codebook 적용 *전* test PAPE가 5개 알고리즘에서 ~1 PAPE 내로 수렴.

| FL backbone | test PAPE (codebook 전) |
|---|---|
| FedAvg | 52.58 ± 0.05 |
| FedProx | 52.48 ± 0.05 |
| FedRep | 53.50 ± 1.26 |
| Ditto | 53.45 ± 0.88 |
| FedProto | 52.65 ± 0.10 |

- 알고리즘 선택은 v09 스케일(114가구, R=20)에서 PAPE discriminator가 아님 → 비용 효율(FedRep, comm −18%)이 실질적 선택 기준.
- centralised 상한(≈ 49.4 PAPE)은 v06 참조. [TODO: v09 centralised cell 미실행]

## 4-2. Codebook and Aux Head 

- **말씀드릴 고민되는 점**: 처음엔 보조MLP(Aux head)로 Local Peak 정보를 보완할 수 있을까 했는데, 실제로는 큰 영향이 없고 일반화시키는 경향을 보임. 
- 정량 확인 (MAE-only ablation, λ_aux 0.3 → 0, 3-seed):
	- codebook 적용 **전** PAPE는 aux 유무와 무관하게 거의 동일 (52.5 vs 52.5) → aux head 단독 효과 미미.
	- codebook 적용 **후**: default(46.9) ≤ MAE-only(47.8) → lift의 주된 원천은 aux head가 아니라 **codebook 보정(RoundCB)**.
	- 해석: round-level FL에서 aux head는 peak를 직접 끌어올리기보다 표현을 일반화시키는 쪽으로 작용 → 본 framework의 핵심 기여는 RoundCB의 A+C 메커니즘.
# 5. Result
**Abstract**
We evaluate peak-region accuracy using Peak Absolute Percentage Error (PAPE). Against local, centralized, standard FL, and time-series foundation-model baselines, the proposed framework achieves the best peak accuracy, reducing PAPE by 11% compared with the best standard FL baseline. 
These results suggest that task-aligned inductive bias and inference-time personalization can be more effective than increasing model scale for privacy-constrained household peak forecasting.

> [TODO: TimesFM 등 TSFM(time-series foundation model) baseline 수치 미확보 — 실행 후 채울 것. 현재 비교군은 local / centralized / standard FL.]

## 5-1. Experiment Result
### figure1. Round-level Federated Learning Baseline
![[Pasted image 20260526175013.png]]
### figure2. Round-level FL with Codebook
![[Pasted image 20260526175009.png]]
### figure3. Codebook Lift 
![[Pasted image 20260527031315.png]]

### Table. RoundCB Codebook Lift (v09, R=20, 3-seed mean ± std)

| FL backbone | BEFORE PAPE | **AFTER PAPE** | **ΔPAPE** |
|---|---|---|---|
| FedAvg | 52.58 ± 0.05 | 46.88 ± 0.28 | −5.70 ± 0.25 |
| FedProx | 52.48 ± 0.05 | 46.83 ± 0.16 | −5.66 ± 0.15 |
| FedRep | 53.50 ± 1.26 | 47.04 ± 0.21 | −6.46 ± 1.05 |
| Ditto | 53.45 ± 0.88 | 46.92 ± 0.39 | −6.53 ± 0.55 |
| FedProto | 52.65 ± 0.10 | 46.91 ± 0.24 | −5.74 ± 0.24 |

- **Lift 보편성**: 5개 FL backbone 전부 −5.7 ~ −6.5 PAPE → codebook은 backbone-agnostic 보정 모듈.
- best standard FL 대비: 52.48 → 46.83 = **약 11% 상대 PAPE 감소** (모델 규모 확대 없이).

## 5-2. Ablation

- **Aux head (λ_aux 0.3 → 0)**: §4-2 참조 — codebook 전 PAPE 거의 불변, 적용 후 default가 MAE-only 대비 우위 (46.9 ≤ 47.8). lift는 codebook 주도.
- **Codebook lift 보편성**: default −5.7 ~ −6.5, MAE-only에서도 −4.5 ~ −5.5 PAPE → backbone·loss 무관하게 유효.
- **K_local (Stage-1)**: K=2가 elbow. K=1은 ~1 PAPE lift 손실, K=8은 4× upload 비용 대비 sub-noise 이득 (v06 §5.5 재검증).
- **α_v0 (correction strength)**: 0.5 (MAE-neutral) ~ 2.0 (PAPE-extreme)의 monotonic Pareto (§3-3).

# 6. Conclusion

- 추가로 관련 논문[^1]에서 알 수 있듯, 개별가구의 Peak 부하는 일주일, 한달 단위의 주기성보다 클러스터, 그룹의 특성 차이에 따라 결정이 많이 되는 것을 알 수 있음
- 라이프스타일을 코드북에 매칭하면 맞춤형 수요반응(DR) 프로그램을 설계하기 좋을 것
- **RoundCB는 method-agnostic 보정 모듈**: 어떤 FL backbone에도 직교적으로 ~−6 PAPE lift를 더하며, raw representation을 전송하지 않아 privacy를 보존.
- standard FL 대비 peak 정확도 ~11% 상대 개선(best FL 52.5 → 46.8 PAPE)을 모델 규모 확대 없이 inference-time personalization으로 달성 → "큰 모델보다 task-aligned bias + 개인화".


---
# 참고
## 제출 초록
Peak-aware Federated Learning using Global Codebook Information for Individual Household Electric Load Forecasting

- 전력망 관리의 수요 측면에서 정확한 가구 부하의 피크 예측은 중요
- 변동성 크고 다종성의 소비 패턴을 가져 어려움
- 연합학습은 민감 정보 없이 중앙 집중 학습이 가능하지만, 기존 FL은 희소하고 뾰족한 Peak 지점에 underfit 해짐
- 제한적인 관측 영역에서도 특히 어려움

- Peak 중심 FL 프레임워크 제안
- Representation-Level Peak Learning with Codebook-based inference-time personalization: 표현층 수준에서 Peak 학습을 하는 코드북 기반 추론 개인화
	- 1. FL하는 동안 Peak 관련 Dynamics를 Encode 하기 위해 
	  NBEATSx로 Peak 정보가 담긴 Hidden repres. 확보
	- 2. KMeans와 Cluster 측 평균 잔차를 통해 완전 FL 코드북을 구성
- Peak 측정을 위하여 PAPE를 도입함. 
- 로컬, 중앙집중, 기본 FL, TSFM 대비 11%의 PAPE 감소량을 보임
- 종합하면 
	- task-aligned inductive bias와 추론시간 개인화로 개별 가구의 FL 효과를 낼 수 있다(큰 모델 안써도 된다)



[^1]: Jin, Ling, et al. "Investigating Underlying Drivers of Variability in Residential Energy Usage Patterns with Daily Load Shape Clustering of Smart Meter Data." _arXiv preprint arXiv:2102.11027_ (2021). https://arxiv.org/abs/2102.11027

