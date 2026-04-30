# scratch_daily_peak_periodicity — pFL을 위한 데이터 탐색

UMass Smart\* 2016 50가구를 대상으로, **peak-aware pFL** 설계 전 단계에서
"가구간 peak 정보가 진짜로 공유 가능한가?"를 검증한 일회성(scratch) 실험 묶음.

모든 스크립트는 `data/raw/Umass/2016/Apt*_2016.csv`를 읽고
출력은 `outputs/scratch_daily_peak_periodicity/`에 저장된다.

```bash
# 스크립트는 모두 단독 실행 가능 (인자는 거의 모두 --n 50, --seed 42)
uv run python experiments/scratch_daily_peak_periodicity/01_decompose_daily_peak.py
uv run python experiments/scratch_daily_peak_periodicity/02_decompose_50hh.py --n 50
uv run python experiments/scratch_daily_peak_periodicity/03_intercorr_pca_loo.py --n 50
uv run python experiments/scratch_daily_peak_periodicity/04_transfer_experiment.py --n 50 --seed 42
uv run python experiments/scratch_daily_peak_periodicity/05_k_sweep_random_control.py --n 50 --seed 42
uv run python experiments/scratch_daily_peak_periodicity/06_verify_paper_h4.py --n 50
uv run python experiments/scratch_daily_peak_periodicity/07_within_hh_peak_hour.py --n 50
```

## 연구 흐름

본 실험 시리즈는 다음 질문을 차례로 푼다.

> Q1. 한 가구의 일간 peak에 주기성이 있는가? → **01**
> Q2. 50가구로 확장해도 주기성이 보이는가? → **02**
> Q3. 가구간 peak에 공유할 정보가 있는가(상관/PCA/LOO)? → **03**
> Q4. 실제로 A→B transfer가 작동하는가(ridge MAE)? → **04**
> Q5. cluster pooling이 random pooling보다 정보적인가? → **05**
> Q6. 본 논문의 H4(amplitude-dominant heterogeneity)가 50가구에서도 성립하나? → **06**
> Q7. HR\@k의 ceiling은 어디인가(within-HH 시간 일관성)? → **07**

각 질문이 다음 질문의 전제를 깎아내리는 구조.
"주기성이 있어 보이지만 (01–03) 실제 transfer는 별로 효과가 없고 (04–05),
그건 가구가 archetype으로 분할되지 않고 (06) **모양은 같고 진폭만 다른** 분포이며,
시간 정보의 within-HH ceiling이 ~40%라서 HR\@2가 구조적으로 막혀있기 때문이다 (07)"
이 한 문장이 본 시리즈의 결론.

---

## 01_decompose_daily_peak.py — 한 가구 STL

**무엇을 보나.** Apt1의 일간 peak (분 단위 → daily max)에 STL period=7을 적용,
trend / seasonal / residual의 분산 비율과 강도(F\_t, F\_s) + ACF + FFT.

**핵심 결과.**
- trend 67% / seasonal 15% / residual 30%
- F\_s ≈ 0.21 (주간 주기성 약함)
- ACF lag 7 = 0.33 (눈에 띄는 weekly bump 없음)

**시사점.** 단일 가구에서도 주간 주기성은 약하다. peak의 day-to-day 변동은 noise + slow trend가 지배.

산출물: `outputs/scratch_daily_peak_periodicity/apt1_period7_decomp.png`.

---

## 02_decompose_50hh.py — 50가구 통계

**무엇을 보나.** 01을 50가구로 확장. 가구별 (F\_t, F\_s, ACF lag 1/7/365)을 모은 뒤 분포로 본다.

**핵심 결과.**
| 통계량 | median | IQR |
|---|---|---|
| F\_t (trend strength) | 0.40 | [0.27, 0.54] |
| F\_s (seasonal strength) | **0.05** | [0.02, 0.10] |
| ACF lag 7 | 0.33 | [0.21, 0.45] |
| ACF lag 365 | 0.15 | — |

50가구 중 강한 weekly seasonal은 **1가구**, 48가구는 약함.

**시사점.** Q1의 답이 "단일 가구에서도 약함"이고 Q2의 답이 "모집단으로도 약함"임이 확인되었으므로,
peak 예측에 단순 주간 주기 prior를 박는 접근은 효과를 기대하기 어렵다.

산출물: `summary_first50hh.csv`, `summary_first50hh.png`.

---

## 03_intercorr_pca_loo.py — 가구간 공유 구조 3종

**무엇을 보나.** T × N 일간 peak 행렬에서 다음 세 가지를 동시에 본다.
1. pairwise Pearson r 분포 (off-diagonal 1225 쌍)
2. PCA scree (각 가구를 train 통계로 표준화 후)
3. LOO 회귀: y\_i ~ a + b · mean(others), per-i R²

**핵심 결과.**
- Pearson r: median **+0.39**, |r| > 0.5 인 쌍이 33%
- PCA: PC1 **47%**, top-3 **68%** — 강한 공통 mode 1개 + sub-mode 2개
- LOO R²: median **0.38**, IQR [0.20, 0.70] — bimodal: 잘 따라가는 가구와 안 따라가는 가구가 갈린다

**시사점.** **공유 신호는 분명히 존재한다.** 다만 LOO R²가 bimodal이므로,
"평균에 잘 따라가는 가구만" 정보가 흐른다 = uniform pooling은 일부 가구를 손해 본다는 신호.

산출물: `intercorr_pca_loo_n50.png`, `loo_r2_n50.csv`.

---

## 04_transfer_experiment.py — 진짜 A→B transfer

**무엇을 보나.** 03이 "공유 신호가 있다"까지만 보였으므로, 실제로 ridge regression으로
**test MAE를 kW로** 떨어뜨리는지 검증.

각 가구마다 70/10/20 split. feature는 `[lag1, lag7, lag14, sin/cos(DoW), sin/cos(DoY)]`.
4개 전략으로 학습:
- **B-only**: 본인 train만 (개인화 상한)
- **Others-only**: 본인 제외 모두 (순수 transfer)
- **Pool**: 모두 (naive FL)
- **Cluster-pool**: 같은 archetype에 속한 가구만. clustering은 train-only feature
  `[cv, ACF lag1, ACF lag7, weekend_ratio, peak_season_cos]`에서 KMeans K=3.

**핵심 결과 (seed 42).**

| 전략 | median test MAE (kW) | Δ vs B-only |
|---|---|---|
| B-only | 0.471 | — |
| Others-only | 0.541 | +0.070 (꾸준히 손해) |
| Pool | 0.430 | **−0.046** [95%CI −0.195, −0.007], 64% win |
| Cluster-pool | 0.456 | −0.018 (cluster c2는 +0.029로 손해) |

baselines: B-mean MAE 0.625, lag1 persistence 0.501, DoW 0.520. B-only는 lag1 대비 −0.030.

**시사점.**
- **Pool > B-only**: 가구간 transfer는 작지만 실재한다.
- **Others-only ≪ B-only**: 본인 데이터를 빼면 손해 — "본인 데이터가 가장 정보적".
- **Cluster-pool ≈ Pool**: hard partitioning의 추가 이득이 작다 (cluster c2는 손해).
- 효과 크기는 **0.05 kW 수준**이고 95%CI가 0에 매우 가깝다 → "있다"고는 말할 수 있으나 큰 lever는 아니다.

산출물: `transfer_n50_seed{42,123}.csv` + `.png`.

---

## 05_k_sweep_random_control.py — random-cluster null + leakage-free K\*

**무엇을 보나.** 04에서 K=3을 미리 박았던 게 cherry-pick일까봐:
- K=2..8 sweep
- val MAE로 K\* 선택 (test 미관측)
- 같은 sizes의 **random label**을 control로 깐다

**핵심 결과.**
- K\* (val로 선택) = **2**
- silhouette 모든 K에서 < 0.2 (구조 약함)
- test Δ vs B-only: K=2에서 cluster-pool **−0.118**, random-pool **−0.124**
- 모든 K에서 cluster ≈ random — **archetype clustering이 random subsampling보다 정보적이지 않다**

**시사점 (이 시리즈 최대의 발견 중 하나).**
04의 "Cluster-pool ≈ Pool"이 단순한 비효율이 아니라 **archetype 자체가 information을 갖지 않음**을 의미한다.
즉, 본 데이터에서 hard partitioning을 통해 가구 그룹을 만드는 접근(=hard pFL clustering)은
random partitioning과 구분되지 않는다. **"가구가 K개 archetype으로 갈린다"는 가설은 깨졌다.**

> 단, 이 결과는 hard cluster + 데이터 분할만 봤다. 본 논문의 codebook은
> shared backbone + per-cluster **residual offset (W5)** 구조라 다르다 — 06에서 그 지점으로 들어간다.

산출물: `k_sweep_n50_seed42.csv` + `.png`.

---

## 06_verify_paper_h4.py — paper §4.6 H4 50가구 재현

**무엇을 보나.** 본 논문(`papers/pfl_unified/paper.md` §4.6)의 H4:
"이질성은 amplitude 축이 지배. 시간 모양은 거의 같다."
50가구 train portion에서 hour-of-day cosine + W1 amplitude로 재현.

**핵심 결과.**

| 통계량 | 본 실험 (50hh) | paper 보고치 |
|---|---|---|
| hour-of-day cosine mean | **0.967** | 0.970 |
| hour-of-day cosine min | **0.885** | 0.811 |
| W1 amplitude mean (kW) | **0.409** | 0.379 |
| W1 amplitude max (kW) | **1.439** | 1.439 |
| per-HH peak amplitude | min 0.31 / max 2.36 | (미보고) |
| → spread max/min | **7.6×** | — |

shape similarity (peak-normalized cosine)는 여전히 mean 0.99 이상. **모양은 거의 같다.**

**시사점.**
H4가 50가구에서 **거의 그대로 재현**된다. 즉:
- 가구간 차이의 본질은 "다른 archetype"이 아니라 "**동일 모양의 다른 진폭**"
- 04/05에서 hard clustering이 작동하지 않은 이유 = 분할할 archetype이 없기 때문
- 본 논문의 codebook이 사실상 **amplitude quantizer**로 동작한다는 가설로 자연스럽게 이어짐

산출물: `h4_verify_n50.png`.

---

## 07_within_hh_peak_hour.py — HR\@k ceiling

**무엇을 보나.** HR\@k = "예측 peak hour가 정답 ±k 안에 들어가는 일수 비율".
한 가구의 day-to-day peak hour 자체가 얼마나 흔들리는가? 가장 똑똑한 모델조차 가구의 modal hour를 맞히는 게 한계라면,
HR\@k의 천장은 **within-HH consistency**다.

per-day peak hour를 hourly-resampled train series에서 뽑아서:
- 원형(circular) std h
- modal hour
- modal-hour baseline에 대한 oracle HR\@1 / HR\@2 (= "modal hour를 항상 외치면 얼마나 맞나")

**핵심 결과 (50가구 train portion).**
| 지표 | median | IQR |
|---|---|---|
| within-HH circular std (h) | **5.83 h** | [4.21, 7.04] |
| oracle HR\@1 (modal-hour ceiling, %) | **26.9** | [21.4, 33.3] |
| oracle HR\@2 ceiling (%) | **39.8** | [32.5, 46.7] |

본 논문의 proposed 보고치: HR\@1 = 26.4%, HR\@2 = 38.0%.

| 비교 | proposed (paper) | 50가구 ceiling | headroom |
|---|---|---|---|
| HR\@1 | 26.4% | 26.9% | **+0.5 pp** |
| HR\@2 | 38.0% | 39.8% | **+1.8 pp** |

modal hour 분포는 bimodal: 7–9시 (아침) + 19–22시 (저녁).

**시사점 (가장 강한 결론).**
- **proposed는 modal-hour baseline에서 1.8 pp 이내**에 있다 = 사실상 "각 가구의 평균 peak 시간을 외친다"는 단순 baseline 수준
- HR\@2가 막힌 진짜 이유는 모델 부족이 아니라 **데이터 자체의 within-HH circular std가 5.83 h**
- 즉 **HR\@k 축에서는 더 이상 짜낼 게 거의 없다**. 시간 정보를 더 보고 싶으면 **외부 covariate (날씨, 캘린더)** 도입이 필수
- "constant modal-hour ceiling"은 oracle이라 부르기는 좀 강하지만 ('진짜 천장'이 아니라 'modal-only 가설의 천장'),
  proposed가 그것의 1.8 pp 안에 있다는 사실은 **architecture 개선의 한계 신호**로 강하다

산출물: `within_hh_hour_n50.csv`, `within_hh_hour_n50.png`.

---

## 종합

| # | 발견 | pFL 설계에 주는 함의 |
|---|---|---|
| 01 | 단일 가구도 weekly 주기성 약함 | 단순 주기 prior 무력 |
| 02 | 50가구 모집단도 F\_s ~0.05 | 동일 |
| 03 | Pearson r ≈ 0.4, PC1 47%, LOO R² bimodal | 공유 신호 있으나 일부 가구만 수혜 |
| 04 | Pool − B-only ≈ −0.05 kW (작지만 실재) | 가구간 transfer는 lever가 작다 |
| 05 | cluster-pool ≈ random-pool **모든 K**에서 | **hard archetype clustering은 죽었다** |
| 06 | shape cosine 0.97 / amplitude spread 7.6× | 이질성의 축은 진폭, codebook = amplitude quantizer 후보 |
| 07 | within-HH circular std 5.83 h, modal-hour ceiling 39.8% | **HR\@k는 천장에 거의 도달**, 시간축 개선엔 외부 covariate 필요 |

**한 줄 요약.** 50가구 UMass에서 "peak는 하나의 모양, 진폭은 7배 분산, 시간은 5.83 h 진동".
이 데이터 분포가 본 논문의 모든 실험 결과를 사전적으로 설명한다 — codebook의 역할, FL의 약한 효과,
HR\@k의 천장, FedProto·FedRep 간의 ≤1 pp 차이.

## 다음 단계 (제안)

1. **codebook = amplitude quantizer 검증**
   `src/models/vq_kmeans.py` 의 KMeans(M=32)를 명시적 amplitude quantile bin (예: 8 bins)으로 대체해 PAPE 비교.
   같으면 codebook의 본질은 진폭 라우팅이라는 H4 강화. (1주차)
2. **1-parameter scaling이 FedRep/Ditto/FedProto와 맞먹는가**
   가구별 학습 가능 scalar α만 두고 backbone은 frozen-shared. (2–3주차)
3. **HR ceiling 돌파**
   NOAA 날씨 + 캘린더 covariate를 NBEATSx exogenous에 합쳐 within-HH std 자체를 줄이려는 시도. (4–6주차)
4. **iterative codebook refit (보류)**
   round-wise refit이 single-shot 대비 의미 있는 이득을 주는지. TAR 위험 검토 후 v04 이후로 이연.
