# RoundCB의 이론적 기반 — 사전 연구 정리 및 포지셔닝

## TL;DR
- **RoundCB의 4가지 핵심 설계요소(① post-hoc/gradient-free 코드북, ② 매 라운드 from-scratch 재구성, ③ 2-stage hierarchical federated KMeans, ④ 1-NN cluster-mean residual output correction)를 모두 동시에 갖춘 선행 연구는 검토한 ~30개 후보 범위 내에서 존재하지 않는다.** 개별 요소는 각각 강력한 선행 문헌(k-FED, FedPFT, FedProto, VQ-VAE, Prototypical Networks, Deep Nearest Centroids, Federated Residual Learning 등)에 의해 정당화되며, 이들의 *조합* — 특히 *frozen forecaster의 은닉표현에 대한 매-라운드 연합 KMeans 코드북 + residual 오프셋* — 이 RoundCB의 novelty다.
- **가장 가까운 이론적 정박점은 두 가지다**: (i) **k-FED (Dennis, Li, Smith, ICML 2021)** — 2-stage federated KMeans (client local Lloyd's → server clustering on union of centroids)가 RoundCB Stage-1/Stage-2 구조의 직접 선행임; (ii) **FedPFT (Beitollahi et al., arXiv:2402.01862, 2024; ICML 2024 "Foundation Models in the Wild" 워크숍 비-아카이벌 포스터)** — frozen foundation-model features 위에서 클라이언트가 *parametric* 통계량(GMM)을 fit하여 서버로 올리는, gradient-free/forward-decoupled 패턴의 가장 명시적 선례. RoundCB는 본질적으로 "FedPFT의 frozen-feature 통계 전송 철학 × k-FED의 2-stage centroid aggregation × Prototypical Networks의 mean-as-prototype × Federated Residual Learning의 출력-측 보정" 을 하나로 묶은 구조다.
- **인용 전략**: §Related Work에서 (a) gradient-free codebook 정당화: FedPFT, k-FED, Prototypical Networks (Snell+ 2017), Visual Recognition with Deep Nearest Centroids (Wang+ ICLR 2023 Spotlight); (b) per-round rebuild 정당화: k-FED의 one-shot 분석을 "round 내 one-shot"로 일반화; (c) 2-stage federated clustering 정당화: k-FED + FKmeansCB (Deng·Wang·Alobaedy, PLOS ONE 20(6):e0326145, 2025); (d) output-side residual correction 정당화: Federated Residual Learning (Agarwal/Langford/Wei 2020) + NBEATSx residual decomposition; (e) Contrast points (반례로 인용): FedProto (gradient-coupled), VQ-VAE (forward-coupled, commitment loss), IFCA (iterative gradient-based clustering).

## Key Findings

### 1. RoundCB의 4대 설계 결정과 각각의 이론적 정박점

| 설계 결정 | 핵심 선행 연구 | 어떤 주장을 뒷받침? |
|---|---|---|
| (A) Post-hoc / gradient-free codebook | FedPFT (Beitollahi+ 2024); k-FED (Dennis+ ICML 2021); Prototypical Networks (Snell+ NeurIPS 2017); Visual Recognition with Deep Nearest Centroids (Wang+ ICLR 2023 Spotlight) | "코드북/프로토타입을 backbone과 분리된 비파라메트릭 통계로 구성해도 잘 작동한다"는 점 |
| (B) Per-round rebuild (EMA·carry-over 없음) | k-FED의 one-shot 통신 분석; FedPFT의 single-round 통신 철학 | "한 라운드 내에서 centroid 통신만으로 충분한 군집 구조가 회수된다"는 분석 (Dennis+ 2021) |
| (C) 2-stage hierarchical federated KMeans (client K=2 → server count-weighted M=32) | k-FED (Dennis+ ICML 2021); FKmeansCB (Deng, Wang, Alobaedy, PLOS ONE 2025) | "local Lloyd's k-means → server-side KMeans on union of local centroids"는 정확히 동일한 토폴로지; FKmeansCB는 *count-weighted server-side KMeans*까지 그대로 일치 |
| (D) Output-side residual correction (ŷ + α·offset) | Federated Residual Learning (Agarwal, Langford, Wei, arXiv:2003.12880, 2020); NBEATSx doubly-residual decomposition (Olivares+ 2022); RAFT retrieval (Han, Lee, Cha, Arik, Yoon. ICML 2025, PMLR 267:21774–21797) | "글로벌 forecast + 보정 residual"의 가법적 분해는 federated/centralized 양쪽에서 검증된 패턴 |

### 2. 핵심 후보 논문 — 카드별 정리

#### 2.1 k-FED — Dennis, Li, Smith. *Heterogeneity for the Win: One-Shot Federated Clustering*. ICML 2021, PMLR 139:2611–2620. arXiv:2103.00697.
- **메커니즘**: 각 디바이스가 로컬에서 *Lloyd's k'-means*를 수행하고, **각 로컬 cluster mean만 서버로 전송**; 서버는 이 centroid들의 합집합을 거리 기반으로 군집화하여 글로벌 k-center를 만든다. 단 1라운드 통신.
- **코드북 구성 방식**: **Gradient-free** (Lloyd's iteration은 classical 통계). **Forward-pass와 완전히 분리**.
- **통신**: centroid (메시지 크기 O(d·k′)) — **raw data 비전송, gradient 비전송**.
- **Output correction**: 없음 (k-FED는 unsupervised 클러스터링 자체가 목적).
- **RoundCB와의 유사도**: **HIGH** (Stage-1/Stage-2 토폴로지에 대한 가장 직접적인 정당화).
- **뒷받침하는 주장**: "2-stage federated KMeans" + "centroid-only 통신으로 충분"이라는 정보이론적 정당화 + "heterogeneity가 분석에 *유리하게* 작용" (논문의 핵심 정리: k′ ≤ √k 영역에서 분리 조건이 완화됨).

> 직접 인용: "We develop and analyze a one-shot federated clustering scheme, k-FED, based on the widely-used Lloyd's method for k-means clustering ... each device solves a local k^(z)-means problem and then communicates its local cluster means."

#### 2.2 FedPFT — Beitollahi, Bie, Hemati, Brunswic, Li, Chen, Zhang. *Parametric Feature Transfer: One-shot Federated Learning with Foundation Models*. arXiv:2402.01862, 2024 (ICML 2024 "Foundation Models in the Wild" 비-아카이벌 워크숍 포스터).
- **메커니즘**: **frozen foundation model**의 feature 위에서 각 클라이언트가 클래스-조건부 **Gaussian Mixture Model (GMM)** 을 EM으로 fit하여 **GMM 파라미터 (mean, cov, weight) 만** 서버로 전송. 서버는 GMM에서 synthetic feature를 샘플링해 classifier head를 학습.
- **코드북 구성 방식**: **Gradient-free** (EM 기반 GMM fitting). **Forward-pass와 분리** (backbone은 frozen).
- **통신**: parametric 통계(GMM 파라미터). raw feature/raw data 비전송.
- **성능 효과**: 논문 abstract에 따르면 "FedPFT enhances the communication-accuracy frontier ... with improvements of up to 20.6%" (8개 데이터셋에서 다른 one-shot FL 방법 대비).
- **Output correction**: 없음 (서버에서 head를 학습하는 방식이지, 테스트 시 잔차 보정은 아님).
- **RoundCB와의 유사도**: **HIGH** (gradient-free/frozen-backbone/parametric-summary 패턴의 가장 명시적인 federated 선례). 단 (i) GMM ≠ KMeans, (ii) one-shot이며 per-round rebuild가 아님, (iii) hierarchical 2-stage가 아님 (서버 단계는 클러스터링이 아니라 head 학습).

> 직접 인용 (Abstract): "Transferring per-client parametric models (specifically, Gaussian mixtures) of features extracted from foundation models ... clients do not send real features."

(인용 형식 권고: FedPFT는 peer-reviewed proceedings에 게재되지 않았으므로 `arXiv:2402.01862` 로만 인용.)

#### 2.3 FedProto — Tan, Long, Liu, Zhou, Lu, Jiang, Zhang. *FedProto: Federated Prototype Learning across Heterogeneous Clients*. AAAI 2022 (Vol. 36, No. 8, pp. 8432–8440). arXiv:2105.00243.
- **메커니즘**: 각 클라이언트가 클래스별 prototype(feature mean)을 계산해 서버로 보내고, 서버는 글로벌 prototype을 집계해 클라이언트에 회신; 클라이언트는 **로컬 학습 시 손실에 L_R = MSE(local_proto, global_proto) 항을 추가**해 backbone을 학습 (총 손실 = L_S + λ·L_R, default λ=1).
- **코드북 구성 방식**: **Gradient-coupled** (prototype이 backbone 학습 손실에 들어가므로 backbone은 prototype을 향해 학습됨). **Forward-pass와는 분리**되지만 **gradient flow는 결합**됨 — 이것이 RoundCB와의 결정적 차이.
- **통신**: prototype (feature mean per class).
- **Output correction**: 없음 (regularization 기반).
- **RoundCB와의 유사도**: **MEDIUM** — 통신 단위가 centroid라는 점은 유사하지만 *gradient coupling* 때문에 사실상 정반대 철학.
- **사용법(RoundCB 논문에서)**: **Contrast point**. "FedProto는 prototype을 gradient regularizer로 쓰는 반면, RoundCB는 prototype/centroid를 gradient 경로 *바깥의* 출력-측 보정자로 쓴다"라고 차별화.

> 직접 인용: "FedProto aggregates the local prototypes ... and then sends the global prototypes back to all clients to regularize the training of local models ... minimize the classification error ... while keeping the resulting local prototypes sufficiently close to the corresponding global ones."

#### 2.4 VQ-VAE — van den Oord, Vinyals, Kavukcuoglu. *Neural Discrete Representation Learning*. NeurIPS 2017. arXiv:1711.00937.
- **메커니즘**: encoder 출력 z_e(x)를 codebook entry e_k로 nearest-neighbor 양자화하여 decoder에 입력. 손실: `L = log p(x|z_q(x)) + ||sg[z_e(x)] − e||² + β||z_e(x) − sg[e]||²` (reconstruction + codebook loss + commitment loss). 코드북은 codebook loss 또는 EMA로 업데이트, 인코더는 straight-through estimator로 학습.
- **코드북 구성 방식**: **Gradient-coupled** (codebook loss/EMA) 이며 **Forward-pass에 codebook이 들어감** (양자화 단계가 decoder 앞).
- **RoundCB와의 유사도**: **LOW (이지만 명백한 contrast)**.
- **사용법**: **Contrast point**. "RoundCB는 VQ-VAE의 codebook concept을 차용하되 (i) commitment loss를 제거, (ii) forward path에서 분리, (iii) classical KMeans로 매 라운드 새로 구성한다."

> 직접 인용: forward computation 시 nearest embedding z_q(x)가 decoder에 전달되고, backward 시 gradient는 straight-through로 encoder에 전달된다 (즉, codebook은 forward path에 포함됨).

#### 2.5 Prototypical Networks — Snell, Swersky, Zemel. NeurIPS 2017. arXiv:1703.05175.
- **메커니즘**: 임베딩 공간에서 클래스 mean을 prototype으로 두고, query를 가장 가까운 prototype으로 분류. Bregman divergence 가정 하 mean이 최적 prototype임을 클러스터링으로 정당화.
- **RoundCB 지지점**: "**mean을 prototype/centroid로 쓰는 것의 통계적 정당화**" — RoundCB의 cluster-mean residual이 곧 Bregman 의미의 최적 centroid임을 뒷받침.
- **유사도**: **MEDIUM** (federated 아님, 그러나 mean-as-codeword에 대한 이론적 정박).

#### 2.6 Deep Nearest Centroids — Wang, Han, Zhou, Liu. *Visual Recognition with Deep Nearest Centroids*. ICLR 2023 Spotlight. arXiv:2209.07383.
- **메커니즘**: 학습된 deep feature 위에서 클래스별 **sub-centroid (K-means 클러스터 중심)** 를 비파라메트릭 분류기로 사용. "deep feature + classical clustering for prediction"의 표본적 사례.
- **RoundCB 지지점**: "**deep representation을 두고 그 위에서 classical clustering (KMeans)을 통해 예측-측 구조를 만든다**"는 패턴 — RoundCB가 federated 환경에서 하는 일과 동일한 철학을 centralized vision 분야에서 입증.

> 직접 인용 (Abstract): "DNC instead conducts nonparametric, case-based reasoning; it utilizes sub-centroids of training samples to describe class distributions and clearly explains the classification as the proximity of test data and the class sub-centroids in the feature space."

- **유사도**: **MEDIUM-HIGH** (federated가 아니라는 한계만 제외하면 가장 가까운 conceptual analog).

#### 2.7 Federated Residual Learning — Agarwal, Langford, Wei. arXiv:2003.12880, 2020.
- **메커니즘**: 서버의 글로벌 모델과 클라이언트의 로컬 모델을 결합해 예측. 단순 합 ŷ = ŷ_global + ŷ_local 형태.
- **RoundCB 지지점**: **"output-side additive correction" 패턴에 대한 federated 선례**. RoundCB의 `ŷ_final = ŷ_backbone + α·offset_codebook` 는 이 가법적 분해의 특수형(personalized local model을 *codebook lookup*으로 대체)으로 해석 가능.
- **유사도**: **MEDIUM** (residual decomposition의 federated 정당화).

#### 2.8 IFCA — Ghosh, Chung, Yin, Ramchandran. *An Efficient Framework for Clustered Federated Learning*. NeurIPS 2020. arXiv:2006.04088.
- **메커니즘**: 클러스터 ID와 모델 파라미터를 alternating으로 추정 (iterative, gradient-based).
- **RoundCB와의 유사도**: **LOW** — 클러스터링이 *클라이언트 자체*를 군집화하는 것이며 gradient/iterative 라는 점에서 RoundCB와 정반대.
- **사용법**: **Contrast point**. "RoundCB는 *클라이언트*가 아니라 *은닉표현 h_g*를 군집화하며, IFCA의 iterative gradient-based 방식과 달리 매 라운드 one-shot으로 centroid만 통신한다."

#### 2.9 FedPCL — Tan et al. *Federated Learning from Pre-Trained Models: A Contrastive Learning Approach*. NeurIPS 2022. arXiv:2209.10083.
- **메커니즘**: pre-trained 백본을 frozen으로 두고 클래스 prototype을 prototype-wise contrastive loss로 학습.
- **RoundCB와의 유사도**: **MEDIUM** — frozen backbone 위에서 prototype 통신이라는 점은 RoundCB와 동일. 그러나 contrastive **gradient** 학습이 들어가며 KMeans 기반 코드북이 아니다.

#### 2.10 FKmeansCB — Deng, Wang, Alobaedy. *Federated k-means based on clusters backbone*. PLOS ONE 20(6):e0326145, 2025-06-12. DOI:10.1371/journal.pone.0326145.
- **메커니즘**: 클라이언트는 로컬 KMeans 수행 후 centroid + cluster size(count)를 서버로 업로드; **서버는 weighted KMeans 로 집계**해 글로벌 centroid 생성. (라플라스 노이즈 추가는 옵션).
- **RoundCB와의 유사도**: **HIGH** — RoundCB Stage-2 (count-weighted KMeans) 와 *정확히 동일한* 서버 측 절차. RoundCB의 "count-weighted" 가중치 부여는 FKmeansCB가 동일 형식으로 사용한 직접적 선례.
- **차이**: 데이터 공간에서 작동(원시 feature) — RoundCB는 *hidden representation* 공간에서 작동하며, 또한 federated forecasting 백본의 *post-hoc per-round* 보정에 사용된다는 점이 다르다.

#### 2.11 NBEATSx — Olivares, Challu, Marcjasz, Weron, Dubrawski. *Neural basis expansion analysis with exogenous variables*. Intl. Journal of Forecasting 2022. arXiv:2104.05522.
- **메커니즘**: doubly residual stacks. 각 stack은 직전 stack의 residual을 입력으로 받아 그 위에 새 component를 더한다. forecast = Σ stack_forecasts.
- **RoundCB 지지점**: **백본 아키텍처의 residual decomposition** — RoundCB가 "**unexplained residual dynamics**"를 codebook으로 잡는다는 표현의 직접 정당화.

#### 2.12 FedMPQ — Yang, Zhang, Zhang, Tang (Hunan University). *FedMPQ: Secure and Communication-Efficient Federated Learning with Multi-codebook Product Quantization*. arXiv:2404.13575 (IEEE 게재).
- **메커니즘**: 모델 업데이트(gradient)에 product quantization을 적용해 통신 압축. 다중 코드북을 서버 측에서 생성.
- **성능**: abstract에 "achieves 99% of the uncompressed baseline's final accuracy, while reducing the uplink communications by 90–95%."
- **RoundCB와의 유사도**: **LOW** — codebook이 *gradient/모델 업데이트 압축*에 사용되며 *예측 보정*이 아님. Contrast point.

### 3. 무엇이 *novel* 인가?

선행 문헌의 격자에 RoundCB를 올려두면 다음 셀들이 **공집합**임이 확인된다:
- (i) **federated** × (ii) **frozen backbone, post-hoc** × (iii) **KMeans 기반 codebook (GMM 아님)** × (iv) **per-round rebuild from scratch (EMA·carryover 없음)** × (v) **2-stage hierarchical (client K=2 → server count-weighted M=32)** × (vi) **1-NN lookup → cluster-mean residual added to forecast** × (vii) **backbone-agnostic plug-in (FedAvg/FedProx/FedRep/Ditto/FedProto에 동일 코드)**.

가장 가까운 단일 매치는 **k-FED ((i)+(iii)+(v) 부분 매치)** 와 **FedPFT ((i)+(ii)+(iv) 부분 매치, 단 GMM/one-shot)** 이며, 나머지는 (a) federated가 아니거나 (DNC, Prototypical Net, NBEATSx, VQ-VAE), (b) gradient-coupled이거나 (FedProto, FedPCL, IFCA, VQ-VAE), (c) output residual correction이 없거나, (d) per-round rebuild가 아니다. 따라서 **"federated post-hoc residual codebook over frozen forecaster hidden representations"** 이라는 카테고리 자체가 RoundCB의 좁은 contribution이다.

## Details

### 3.1 "Gradient-free / forward-decoupled" 주장을 뒷받침하는 인용 순서
1. **FedPFT (Beitollahi+ arXiv:2402.01862, 2024)** — "frozen foundation model에서 parametric statistic만 보내는" 패턴의 가장 명시적 선례.
2. **Prototypical Networks (Snell+ NeurIPS 2017, arXiv:1703.05175)** — class mean을 분류기로 쓰는 것의 통계적 정당화 (Bregman divergence ↔ class-mean optimality).
3. **Visual Recognition with Deep Nearest Centroids (Wang+ ICLR 2023 Spotlight, arXiv:2209.07383)** — deep feature 위에서 KMeans 서브-centroid를 비파라메트릭 분류기로 쓴 directly-analogous 사례.
4. **k-FED (Dennis+ ICML 2021)** — federated에서 centroid-only 통신의 정보이론적 충분성.

### 3.2 "2-stage hierarchical federated clustering" 주장을 뒷받침하는 인용 순서
1. **k-FED (Dennis+ ICML 2021)** — Lloyd's local → server clustering of union. 이론적 분리 조건과 수렴 보장 제공.
2. **FKmeansCB (Deng, Wang, Alobaedy, PLOS ONE 20(6):e0326145, 2025; doi:10.1371/journal.pone.0326145)** — **count-weighted server-side KMeans** 의 직접 선례 (RoundCB Stage-2의 가장 가까운 매치).
3. **Greedy centroid initialization for federated K-means (Knowl. Info. Syst. 2024)** — centroid + size를 서버에 보내는 일반 패턴.

### 3.3 "Output-side residual correction" 주장을 뒷받침하는 인용 순서
1. **Federated Residual Learning (Agarwal, Langford, Wei, arXiv:2003.12880, 2020)** — global + local residual의 가법적 분해를 federated에서 정식화.
2. **NBEATSx (Olivares+ 2022, arXiv:2104.05522)** — doubly-residual stacks가 RoundCB의 backbone aesthetic.
3. **RAFT (Han, Lee, Cha, Arik, Yoon. *Retrieval Augmented Time Series Forecasting*, ICML 2025, PMLR 267:21774–21797; arXiv:2505.04163)** — retrieval-as-correction 아이디어 (centralized이지만 *retrieval*을 사용해 forecast를 보강하는 최근 동향). RoundCB는 retrieval 결과를 *residual* 로 한정한다는 점이 차별점.
4. **TS-Memory / MEMTS** — retrieval/memory 기반 시계열 보정의 최근 동향. RoundCB의 inference-time lookup과 개념적으로 가까움.

### 3.4 명시적 Contrast Points (반례로 인용해야 할 것들)
- **FedProto (AAAI 2022)** — prototype을 *gradient regularizer* 로 쓰므로 forward/gradient가 backbone과 결합. RoundCB는 정반대.
- **VQ-VAE (NeurIPS 2017)** — codebook이 forward path에 포함되며 commitment loss로 gradient 결합. RoundCB는 양쪽 모두 끊는다.
- **IFCA (NeurIPS 2020)** — iterative + gradient-based clustering. RoundCB는 one-shot per round + classical KMeans.
- **FedMPQ (Yang+ arXiv:2404.13575)** — product quantization을 *통신 압축*에 사용 (gradient quantization, 99% 정확도 보존 / 90–95% 업링크 절감). RoundCB는 codebook이 압축이 아니라 예측 보정 도구.

## Recommendations (Master's 학위논문 §Related Work 작성 전략)

1. **단계적 권고 — 즉시 적용 가능한 인용 패키지 (5~8 인용)**
   - 첫 단락 "post-hoc, gradient-free, frozen-backbone" 정당화: **FedPFT (Beitollahi+ arXiv:2402.01862, 2024)** + **DNC (Wang+ ICLR 2023, arXiv:2209.07383)** + **Prototypical Networks (Snell+ NeurIPS 2017)**.
   - 두 번째 단락 "2-stage federated clustering" 정당화: **k-FED (Dennis+ ICML 2021)** (메인) + **FKmeansCB (Deng+ PLOS ONE 2025, doi:10.1371/journal.pone.0326145)** (count-weighted 변형 선례).
   - 세 번째 단락 "output-side residual correction" 정당화: **Federated Residual Learning (Agarwal+ arXiv:2003.12880, 2020)** + **NBEATSx (Olivares+ 2022)** + (선택) **RAFT (Han+ ICML 2025)**.
   - 네 번째 단락 contrast: **FedProto (AAAI 2022)** + **VQ-VAE (NeurIPS 2017)** + **IFCA (NeurIPS 2020)** + **FedMPQ (arXiv:2404.13575)**.

2. **방어해야 할 reviewer 질문에 대한 사전 대응**
   - *"왜 KMeans? GMM (FedPFT)이 더 표현력 있지 않나?"* → KMeans는 (i) 해석 가능한 *centroid + count* 통신 단위가 자연스럽고, (ii) k-FED 이론(heterogeneity가 분리 조건을 *완화*)이 직접 적용되며, (iii) Bregman 의미에서 mean-as-residual의 통계적 최적성(Prototypical Networks)을 그대로 상속하기 때문. GMM은 covariance 행렬까지 보내야 해 통신 비용 ↑.
   - *"왜 매 라운드 새로 짓는가? EMA로 유지하지 않는가?"* → backbone이 라운드마다 evolve하므로 hidden representation 분포가 shift; carryover된 codebook은 stale함. k-FED의 one-shot 분석이 *라운드 내* 한 번의 통신으로 충분함을 보장.
   - *"FedProto와 어떻게 다른가?"* → FedProto의 prototype은 backbone 학습 손실에 들어가 *gradient* 로 결합. RoundCB의 codebook은 inference time에만 사용되며 backbone과 forward/gradient 양쪽 모두 분리.
   - *"FedPFT와 어떻게 다른가?"* → (i) per-round rebuild vs one-shot, (ii) KMeans vs GMM, (iii) 2-stage hierarchical KMeans vs 클라이언트→서버 head 학습, (iv) test-time residual lookup vs synthetic-feature classifier.

3. **결정 기준 (체크포인트)**
   - **만약 ablation에서 EMA carryover가 from-scratch보다 좋다면** → "per-round rebuild" 주장을 후퇴시키고 EMA 변형을 main으로. 그 경우 VQ-VAE의 EMA codebook update 인용을 강화.
   - **만약 client K>2가 의미 있게 도움이 된다면** → k-FED의 k′ 분석 (k′ ≤ √k regime)을 명시적으로 인용하여 분리 조건을 논의.
   - **만약 backbone-agnostic 주장(FedAvg/FedProx/FedRep/Ditto/FedProto 모두에 plug-in 가능)을 강조하고 싶다면** → §Experiments에서 다섯 backbone 모두에 동일 코드 attach한 표를 제시하고, FedRep (Collins+ ICML 2021), Ditto (Li+ ICML 2021)의 personalization 메커니즘이 RoundCB와 *orthogonal* 함을 강조.

4. **인용 형식 권고**
   - FedPFT는 peer-reviewed proceedings 게재가 확인되지 않음 (ICML 2024 "Foundation Models in the Wild" 워크숍은 비-아카이벌). **arXiv preprint 형식**으로만 인용 (`arXiv:2402.01862`). Reviewer가 "FedPFT는 어디 게재되었나"라고 물을 가능성에 대비할 것.
   - k-FED, FedProto, VQ-VAE, IFCA, Prototypical Networks, NBEATSx, DNC (ICLR 2023 Spotlight), Federated Residual Learning, RAFT (ICML 2025), FKmeansCB (PLOS ONE 2025) 는 모두 메이저 venue 또는 peer-reviewed journal 게재이므로 안전.

## Caveats

- **FedPFT의 venue 미확정**: arXiv:2402.01862는 ICML 2024 "Foundation Models in the Wild" 워크숍(비-아카이벌)에서 발표된 것 외에 peer-reviewed proceedings 게재가 확인되지 않음. Flower baseline 구현(`flower.ai/docs/baselines/fedpft.html`)은 존재 — arXiv preprint으로만 인용 권고.
- **"RoundCB와 동일"한 단일 논문 부재**: 위 정리는 *부분 매치* 의 합성이며, 본 보고서의 "novel" 판정은 검토한 ~30개 후보 논문 범위 내에서의 결론. 시계열 forecasting 분야 federated 논문(특히 KSNRE/AFORE 같은 에너지 도메인 venue), 그리고 *federated load forecasting* / *federated electricity price forecasting* 키워드의 niche venue 에는 검색이 닿지 않은 사례가 있을 수 있음 — 최종 제출 전 해당 키워드로 추가 sweep 권고.
- **k-FED의 server-side step은 standard KMeans가 아닐 수 있음**: 원 논문은 server-side를 "distance-based clustering on union of local centers"로 기술 (Algorithm 1: 합집합에서 중심들을 거리 기반으로 선택해 initial center로 사용), RoundCB의 *count-weighted standard KMeans* 와 형식이 미세하게 다르다. 정확한 매치는 FKmeansCB(Deng+ 2025)이지만 이 논문은 federated KMeans 응용 분야 (privacy via Laplace noise) 가 RoundCB와 다르다.
- **"Backbone-agnostic"의 검증 부담**: RoundCB가 FedAvg/FedProx/FedRep/Ditto/FedProto 모두에 동일 코드로 attach된다는 claim은 *경험적*으로 보여야 한다. 이론적 정당화는 "codebook이 forward/gradient 어디에도 결합되지 않으므로 backbone 학습 알고리즘에 invariant"라는 단순 논증으로 가능. FedProto 위에 attach할 경우 prototype loss와 RoundCB의 codebook이 *동시에* 작동하므로 ablation에서 명확히 분리할 것.
- **NBEATSx residual decomposition은 *intra-model*** (stack 간 residual)이며 RoundCB의 *post-prediction codebook offset* 과는 다른 층위. NBEATSx는 "RoundCB의 백본이 자연스럽게 residual을 노출한다"는 의미에서만 인용하는 것이 안전.
- **VQ-VAE의 commitment loss / EMA 업데이트**는 RoundCB가 *명시적으로 거부* 하는 설계 — 차별화 포인트로만 활용해야 하며, "RoundCB는 VQ-VAE의 일반화" 식으로 쓰면 오류.
- **DNC와의 가까움**: DNC가 RoundCB와 가장 닮은 *centralized* 사례이므로, "DNC의 federated, residual, post-hoc 확장으로 볼 수 있다"는 한 줄 framing이 reviewer에게 명확한 mental model을 제공한다. 단, DNC는 *학습 도중* 매 epoch sub-centroid를 갱신하므로 *완전 frozen* 은 아님 — 이 nuance를 정확히 전달할 것.