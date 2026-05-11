# FL 방법론 분석 — FedSGD vs FedAvg

> v06 (round-level FL training dynamics) 설계용 reference. 채택 축은
> McMahan 2017 원조 dichotomy인 **FedSGD vs FedAvg** 두 가지. 다른 분류
> 축들 (Sync vs Async, Horizontal vs Vertical, Cross-silo vs Cross-device,
> Generic FL vs Personalised FL) 은 §6 "추후 연구 후보" 에 메모만 남기고
> v06 본 실험에서는 다루지 않는다.

---

## 1. Why this doc

v01–v05는 모두 *train → cold* protocol 위에서 cold-side 정확도만 평가했다.
즉 federated 학습이 끝까지 돈 뒤의 **단일 종점 snapshot** 만 보고 cold
personalisation 효과를 측정한 것이다. FL 자체의 **학습 동역학** —
라운드별 train-side 정확도, 통신 비용, client drift, convergence 형태 —
은 평가되지 않았다.

v06의 목표:

> **FL 통신 패턴이 *학습 trajectory* 에 미치는 효과를 라운드 단위로 측정**.

이 목표를 달성하려면 두 가지 근본적 방법론을 reference axis로 깔아야 한다 —
McMahan 2017 [1] 원조 paper에서 정의된 **FedSGD** 와 **FedAvg**. 이 둘이
"local 계산량 vs 통신 빈도"라는 trade-off의 양 끝점을 형성한다.

---

## 2. 원조 framing — McMahan 2017

McMahan et al. 2017 *"Communication-Efficient Learning of Deep Networks
from Decentralized Data"* (AISTATS) [1] 가 FL field 자체를 만든 paper.
Phase 두 가지를 같은 framework에서 정의:

- **FederatedSGD (FedSGD)**: client는 매 step gradient를 서버에 보낸다.
  서버는 weighted-mean으로 합쳐 한 번 SGD step을 진행 → broadcast.
- **FederatedAveraging (FedAvg)**: client가 *여러 epoch (E)* 의 local SGD를
  돌린 뒤 **weight** 자체를 서버에 보낸다. 서버는 데이터-비례 가중평균.

McMahan paper의 핵심 주장: **FedAvg는 FedSGD의 일반화** 이다. 세 hyperparam:

| 기호 | 의미 | FedSGD | FedAvg (논문 default) |
|---|---|---|---|
| `E` | client 한 라운드당 local epoch | 1 (정확히는 1 step) | 1, 5, 20, … |
| `B` | local mini-batch 크기 | `∞` (full local batch) | finite (10, 50, …) |
| `C` | 라운드당 sampled client 비율 | 같음 | 같음 |

`E=1, B=∞` 로 설정하면 FedAvg는 정확히 FedSGD가 된다. 즉 둘은 *같은 알고
리즘 패밀리의 두 한계점* 이지, 이질적 두 방법이 아니다.

---

## 3. FedSGD — 정의와 특성

### 3.1 알고리즘

```
Round t = 1, 2, …:
    Server broadcasts θ_t to selected clients S_t
    For each client k ∈ S_t in parallel:
        g_k ← ∇_θ L_k(θ_t)         # one full-batch gradient on client k
    Server aggregates:
        θ_{t+1} ← θ_t − η · Σ_k (n_k / n) g_k
```

### 3.2 통신 비용

- **라운드당**: gradient 한 번 → 모델 크기 `|θ|` × 32-bit float
- **총 학습**: gradient를 매 SGD step마다 보내므로, 같은 데이터-pass 수
  를 만들기 위해 라운드 수가 *FedAvg의 E배* 만큼 늘어난다.
- **Smart-meter 80 가구 case** (NBEATSxAux ≈ 100K 파라미터): 라운드당
  업로드 ≈ 0.4 MB/client × 80 = **~32 MB/round**. 200 epoch-equivalent
  까지 학습하려면 200 라운드 × 32 MB ≈ **6.4 GB**.

### 3.3 수렴 특성

- Centralised SGD와 정확히 등가 (when `B=∞`, `C=1.0`, no aggregation
  noise). 따라서 **convergence 분석이 깨끗** 하고 baseline으로 명확.
- **Client drift 없음** — 매 step 동기화되므로 non-IID에 이론적 robust.
- 단점은 순전히 **통신 비용** 과 **straggler 영향** — 가장 느린 client가
  매 step의 wall-clock을 결정.

### 3.4 v06에서의 역할

v06에서 FedSGD는 **이상적 reference (limit case)** 로 깔아둔다. 어차피
실제 system에서 FedSGD는 통신비 때문에 쓸모가 없지만, "FedAvg가
client drift 없이 돌면 어디까지 갈 수 있는가" 의 *상한* 을 알려준다.
FedAvg의 cold-side / train-side 정확도를 FedSGD baseline과 비교하면
*federation 자체가 만들어내는 성능 손실* 을 분리 측정 가능.

---

## 4. FedAvg — 정의와 특성

### 4.1 알고리즘

```
Round t = 1, 2, …:
    Server broadcasts θ_t to selected clients S_t
    For each client k ∈ S_t in parallel:
        θ_k ← θ_t
        For e = 1, …, E:
            For each mini-batch b of size B:
                θ_k ← θ_k − η · ∇ L_k(θ_k; b)
    Server aggregates:
        θ_{t+1} ← Σ_k (n_k / n) · θ_k         # weighted mean of weights
```

### 4.2 통신 비용

- **라운드당**: weight 한 번 (gradient 대신) → 모델 크기 `|θ|`
- **총 학습**: 같은 데이터-pass 수를 만드는 데 라운드 수가 `1/E` 배.
- **80 가구, E=10, |θ|≈100K, 32-bit fp**: 라운드당 ≈ 32 MB,
  20 라운드 → **~640 MB**. FedSGD 대비 **~10×** 절감 (E배 그대로).
- 실측 (McMahan 2017 Fig. 1): MNIST CNN에서 FedAvg(E=20, B=10) 가
  FedSGD 대비 **~30× 적은 라운드** 로 같은 정확도 도달.

### 4.3 수렴 특성

- **장점**: 통신 라운드 수가 압도적으로 적음. 같은 정확도 budget 내에서
  bytes-vs-accuracy frontier가 FedSGD를 dominant.
- **단점 — client drift**: 각 client의 local SGD가 자기 데이터의 local
  optimum 쪽으로 끌려가서, E가 클수록 *같은 weight space의 평균* 이
  무의미해진다. Non-IID에서 특히 심해짐 (Karimireddy et al. 2020,
  SCAFFOLD [3]; Li et al. 2020, FedProx [4]).
- **non-IID drift 보정 알고리즘들** — FedAvg의 직계 후손:
  - **FedProx** [4]: local objective에 `μ/2 · ‖θ_k − θ_t‖²` proximal
    항을 더해 local model이 global에서 너무 멀어지지 않게 함.
  - **SCAFFOLD** [3]: control variate를 client/server 양쪽에 두고
    drift를 *해석적으로* 빼낸다.
  - **MOON** [5]: contrastive loss로 local model이 global model의
    representation에서 멀어지지 않게 regularise.
- 모두 "FedAvg에 한 항 추가" 형태의 변형으로, 본 plan의 v04 baseline
  matrix에 이미 들어있는 **FedAvg / FedProx / FedRep / Ditto** 4종은
  전부 *FedAvg family* 다. 즉 v06에서 FedAvg는 단일 알고리즘이 아니라
  *family 묶음* 으로 다뤄진다.

### 4.4 v06에서의 역할

v06의 main protocol 자체가 FedAvg다 — "10 epoch local → FL comm → ..."
의 E=10이 그것. v06의 비교 축:

- **E sweep**: E ∈ {1, 5, 10, 20} 으로 FedSGD ↔ FedAvg 사이 spectrum을
  측정. E=1, B=∞ → FedSGD (limit). E↑ → 통신 절감 vs drift 증가.
- **R sweep**: 총 epoch budget을 fix (예: 200 epoch-equivalent) 한 뒤
  R = 200/E 라운드 동안 FL 통신.
- **per-round metric**: 매 라운드 직후 (i) train-client validation,
  (ii) cold-client zero-shot, (iii) FL-eval segment 의 PAPE/HR@k/MAE
  를 기록.

→ "round-vs-accuracy" 와 "bytes-vs-accuracy" frontier가 FedSGD를 한쪽
끝, FedAvg(E=20)을 다른 쪽 끝으로 한 *parametric curve* 로 그려진다.

---

## 5. 두 방법론의 통합 관점

본질은 **"local 계산을 얼마나 누적한 뒤 동기화할 것인가"** 라는 단 하나의
연속적 trade-off:

```
   FedSGD                                                       FedAvg(E=∞)
     |                                                                |
     |←————————————— E (local epochs per round) ———————————————→|
     |←————————————— communication frequency ↓ ————————————————→|
     |←————————————— client drift ↑ ——————————————————————————→|
     |←————————————— total bytes ↓ ———————————————————————————→|
     |←————————————— wall-clock vs convergence ramp shape ————→|
```

v06은 이 spectrum을 **이산적 E ∈ {1, 5, 10, 20}** 로 sample하여 곡선의
형태 (특히 cold-side metric의 "knee point") 를 측정하는 실험이다. 한
쪽 끝(FedSGD)도 무한히 다른 쪽 끝(E→∞, 사실상 centralised pooling) 도
*degenerate* 라는 것은 알려진 사실이지만, *어떤 E에서 cold PAPE-vs-bytes
frontier가 minimum 인지* 는 residential load 도메인에서 실측된 적이
없다 — v06의 가장 깨끗한 contribution candidate.

---

## 6. 추후 연구 후보 (v07+ 로 미룸)

본 doc은 v06이 다루지 *않는* 다른 분류 축들을 메모만 남긴다:

### 6.1 Synchronous vs Asynchronous FL

- 서버가 모든 client를 기다리느냐 (sync) vs 도착하는 대로 통합 (async).
- 대표: FedAsync (Xie 2019) [7], FedBuff (Nguyen 2022, Meta) [8].
- v06 환경 (UMass 80 가구, 시뮬레이션, stragglers 약함) 에서는 async의
  이득이 크지 않을 가능성. 실제 cross-device 배포로 framing을 옮길 때
  필요해지는 축.

### 6.2 Horizontal vs Vertical FL

- Yang et al. 2019 [6] 의 데이터 분할 축. UMass 80 가구는 모두 동일
  feature (load timeseries) 이므로 **Horizontal 한쪽만 해당**. 본
  project에서는 dichotomy로 쓸 수 없는 setting.

### 6.3 Cross-silo vs Cross-device

- Kairouz et al. 2021 *Advances and Open Problems in Federated Learning*
  [9] 의 federation-scale 축. 80 가구는 명목상 cross-device이지만 full
  participation + stable hardware로 시뮬되어 있어 실제로는 cross-silo에
  가깝다. 진짜 cross-device 효과(드문 참여, 배터리/네트워크 제약) 는
  v07+ 의 별도 axis.

### 6.4 Generic FL vs Personalised FL

- Personalised FL (FedRep, Ditto, FedProto, pFedMe, Per-FedAvg) 은
  v01–v05에서 이미 다룬 main thread. v06은 의도적으로 personalisation
  *없는 깨끗한 FL* 만 다룬다 (사용자 지시: "personalised 보류"). pFL을
  round-level dynamics 위에 다시 얹는 건 v07+ 후보.

---

## 7. v06 plan에 직접 반영해야 할 점

1. **Algorithm anchor**: v06 main protocol = FedAvg (E=10), reference
   limit = FedSGD (E=1, B=∞).
2. **Hyperparam sweep**: E ∈ {1, 5, 10, 20}; B fixed at 512 (v04
   convention 그대로); C=1.0 (full participation, 80 가구 모두 매 라운드).
3. **Total budget**: 총 epoch-equivalent 200으로 고정 → R = 200/E 라운드.
4. **Per-round logging**: 라운드별 train-val PAPE/HR@k/MAE, cold zero-shot
   PAPE/HR@k/MAE, FL-eval segment metrics, communication bytes
   (cumulative).
5. **Frontier outputs**: round-vs-accuracy curve, bytes-vs-accuracy
   curve — 두 plot이 v06의 main figures.
6. **Backbone**: NBEATSxAux frozen-architecture, fresh weights (no v10
   checkpoint reuse — train trajectory 자체가 종속변수).
7. **W5 codebook은 *v05 와 동일* 하게 frozen post-training fit** —
   v06은 *학습 dynamics* 만 보므로 codebook은 라운드별로 매번 fit하지
   않는다 (그것은 또 다른 종속변수가 되어 dynamics 측정을 흐림).

---

## 8. 참고문헌

[1] McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B.
    *Communication-Efficient Learning of Deep Networks from Decentralized Data*.
    AISTATS 2017. arXiv:1602.05629.
    https://arxiv.org/abs/1602.05629

[2] Konečný, J., McMahan, H. B., Yu, F. X., Richtárik, P., Suresh, A. T.,
    & Bacon, D. *Federated Learning: Strategies for Improving Communication
    Efficiency*. NeurIPS Workshop 2016. arXiv:1610.05492.
    https://arxiv.org/abs/1610.05492

[3] Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S. J., Stich, S. U.,
    & Suresh, A. T. *SCAFFOLD: Stochastic Controlled Averaging for
    Federated Learning*. ICML 2020. arXiv:1910.06378.
    https://arxiv.org/abs/1910.06378

[4] Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V.
    *Federated Optimization in Heterogeneous Networks*. MLSys 2020.
    arXiv:1812.06127.
    https://arxiv.org/abs/1812.06127

[5] Li, Q., He, B., & Song, D. *Model-Contrastive Federated Learning*.
    CVPR 2021. arXiv:2103.16257.
    https://arxiv.org/abs/2103.16257

[6] Yang, Q., Liu, Y., Chen, T., & Tong, Y. *Federated Machine Learning:
    Concept and Applications*. ACM TIST 2019. arXiv:1902.04885.
    https://arxiv.org/abs/1902.04885

[7] Xie, C., Koyejo, S., & Gupta, I. *Asynchronous Federated Optimization*.
    arXiv:1903.03934, 2019.
    https://arxiv.org/abs/1903.03934

[8] Nguyen, J., Malik, K., Zhan, H., Yousefpour, A., Rabbat, M., Esmaeili,
    M. M., & Huba, D. *Federated Learning with Buffered Asynchronous
    Aggregation*. AISTATS 2022. arXiv:2106.06639.
    https://arxiv.org/abs/2106.06639

[9] Kairouz, P. et al. *Advances and Open Problems in Federated Learning*.
    Foundations and Trends in ML, 2021. arXiv:1912.04977.
    https://arxiv.org/abs/1912.04977

---

*Last updated: 2026-05-01.*
