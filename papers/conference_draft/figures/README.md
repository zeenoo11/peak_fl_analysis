# Presentation Figures — 설명 & 해석

`presentation_final.md`에 쓰이는 발표용 그림 모음.

**공통 조건** (모든 그림 동일):
- 데이터: UMass Smart\* 2016 hourly, **114 가구**, per-client 70/10/20 (chronological), per-apt z-norm(train only).
- 프로토콜: **MAE-only (λ_aux = 0)** — codebook 보정의 순수 효과를 분리하기 위함. R = 20 rounds, 5 FL 알고리즘.
- 집계: seeds {42, 123, 7}, **mean ± std across seeds** (fig7 제외, fig7은 FedAvg/seed42 단일 run).
- 지표: **PAPE (Peak Absolute Percentage Error, %)** — 하루 최대 부하의 진폭 오차. 낮을수록 좋음.
- 생성 스크립트: fig1–fig6·fig8 = `experiments/v09_round_vq_codebook/12_make_presentation_figures.py`,
  fig7 = `.../13_make_presentation_tsne.py`.

> 제목에 `MAE-only`·`aux=0`·`v09` 같은 내부 기호는 의도적으로 넣지 않음(발표 관점).

---

## fig1 — `fig1_fl_baseline_pape.png`
**Federated Baseline: Peak Forecasting Error per Round**

- **무엇**: codebook 보정 *전*, 5개 FL 알고리즘(FedAvg/FedProx/FedRep/Ditto/FedProto)의 라운드별 test PAPE.
- **해석**: R1의 near-random(~70) 상태에서 빠르게 하강해 **~52.5 PAPE로 수렴**. 5개 알고리즘이 **~1 PAPE 내로 겹침** →
  이 스케일(114가구·R20)에서 **알고리즘 선택은 peak 정확도의 결정 변수가 아니다**. 실질적 선택 기준은 통신 비용 등 효율.

## fig2 — `fig2_codebook_corrected_pape.png`
**With Global Codebook Correction: Peak Error per Round**

- **무엇**: codebook 보정 *후*의 라운드별 test PAPE(평균선만). 빨간 점선 = 보정 전 수렴 수준(~53).
- **해석**: 모든 알고리즘이 **baseline 아래(~47.7~48.0)로 하강**하고 그 상태를 유지 →
  codebook 보정이 backbone·알고리즘과 무관하게 **일관되게 peak 오차를 낮춘다**.

## fig2b(=fig8) — `fig8_baseline_vs_corrected.png`
**Global Codebook Correction Shifts Every Algorithm Down** *(fig1 + fig2 합본)*

- **무엇**: 한 축에 baseline(**실선**, ~52.5)과 보정 후(**점선**, ~47.8)를 함께. 색 = 알고리즘.
- **해석**: 실선 cluster와 점선 cluster의 **세로 간격이 곧 codebook lift(−4.5~5.5 PAPE)**.
  보정이 전체 묶음을 통째로 끌어내림을 한눈에 보여줌.

## fig3 — `fig3_codebook_lift.png`
**Codebook Correction Lift on Peak Error**

- **무엇**: 최종 라운드 PAPE를 알고리즘별 막대로. 연한 막대 = baseline, 진한 막대 = 보정 후, 위 라벨 = Δ.
- **해석**: 전 backbone에서 **Δ −4.5 ~ −5.5 PAPE**. best standard FL 기준 **52.48 → 47.68 ≈ 9% 상대 감소**(모델 규모 확대 없이).
  errorbar(seed std)가 lift보다 훨씬 작아 효과가 안정적.

## fig4 — `fig4_peak_gain_vs_mae_cost.png`
**Large Peak Gain, Negligible Average Cost** *(최종 라운드, 알고리즘별)*

- **무엇**: 이중축 막대. 파랑(좌축) = peak error 변화(Δpp), 주황(우축) = 평균 MAE 변화(상대 %).
- **해석**: peak는 **−4.5~5.5pp 하락**하는데 평균 MAE는 **+0.7~1.4%만 상승** →
  "peak를 크게 개선하면서 평균 예측은 거의 해치지 않는다"는 trade-off의 비대칭성을 정량화.

## fig5 — `fig5_codebook_effect_per_round.png`
**Codebook Correction per Round (5개 알고리즘 평균)**

- **무엇**: fig4의 라운드별 버전. 파랑(좌축) ΔPAPE, 주황(우축) MAE 상대 %, 회색 = ±2% band.
- **해석**: 초반(R1) backbone이 부정확할 땐 codebook이 PAPE·MAE 둘 다 크게 보정하지만,
  **수렴 후엔 PAPE −5pp 유지 / MAE는 ±2% band 안**에 머무름 → 정상 구간에서 평균 비용이 무시할 수준임을 시계열로 확인.
  *주의*: 우축이 R1 transient(backbone near-random) 때문에 −60%까지 늘어남 — 메시지는 수렴 구간 기준으로 읽을 것.

## fig6 — `fig6_codebook_effect_vs_offset.png`
**Correction Settles as Offsets Shrink and Stabilize** *(5개 알고리즘 평균)*

- **무엇**: 주황(좌축) ΔPAPE, 갈색(우축) = 평균 CMO offset L2 norm(보정 벡터의 크기).
- **해석**: 라운드가 진행되며 **offset 크기가 줄고 안정화**될수록 ΔPAPE가 **−5 부근으로 수렴** →
  backbone이 좋아질수록 필요한 보정량은 작아지지만 lift는 유지됨. codebook이 라운드와 함께 "자리 잡는" 과정을 보여줌.

## fig7 — `fig7_latent_codebook_tsne.png`
**Federated Residual Codebook in Latent Space (t-SNE)** *(FedAvg, seed 42)*

- **무엇**: backbone hidden `h_g`(64-d)의 t-SNE 3-panel.
  (좌) 가구별 local prototype(★, Stage-1) · (중) 서버에서 merge된 global codebook(✕, Stage-2) · (우) test latent의 최근접 codebook routing(색 = 배정 cluster).
- **해석**: R/A/C 파이프라인의 시각적 요약. local prototype들이 latent space를 덮고, 서버 codebook이 그 분포를 대표하며,
  test latent이 cluster별로 깔끔히 라우팅됨(utilization 1.0, perplexity ≈ 24.8) → federated codebook이 pooled 구성과 동등한 품질.
  *주의*: t-SNE는 run별 투영이라 알고리즘 평균이 무의미 → 단일 run 예시.

---

## Global-baseline 비교 (fig9–fig11)

세 그림 모두 **동일 데이터·동일 서사**의 다른 시각화 — RoundCB(우리)가 비-federated 참조점들과 어디에 서는가.
포함 baseline: **Chronos-Bolt-small**(zero-shot, 전체 최고)·**TimesFM**(zero-shot)·**DLinear**(centralized)·
**NBEATSx**(centralized, 동일 backbone) + 우리 **FL baseline**·**RoundCB**.

- 수치 출처: foundation/centralized NF는 **v09 per-client 프로토콜**(114가구, test split)에서 직접 측정.
- 우리 FL baseline/RoundCB는 5 algo × 3 seed의 **최종 라운드 pooled mean ± std** (baseline ≈ 52.7, RoundCB ≈ 47.8).
- **† Centralized NBEATSx (49.4)** 는 v06(E=40) 수치 — v09 R=20과 학습예산이 달라 각주로 명시(공통).

**핵심 메시지**: RoundCB(47.8)는 데이터를 모으지 않고도 **동일 backbone의 centralized 상한(49.4)을 추월**하고,
훨씬 큰 **최고 foundation model(Chronos 46.6)에 근접**.

### fig9 — `fig9_global_baseline_comparison.png`
**Where RoundCB Stands vs Global Baselines** *(수평 막대, 랭킹)*

- **무엇**: PAPE 오름차순 수평 막대. 색 = 분류(회색 foundation / 주황 centralized / 파랑 ours), RoundCB 막대 굵은 테두리 + 47.8 점선.
- **해석**: 순위가 한눈에 — Chronos(46.6) < DLinear(46.7) < **RoundCB(47.8)** < NBEATSx centralized(49.4) ≈ TimesFM(49.5) < FL baseline(52.7).
  *주의*: x축을 44부터 잘라(truncated) 차이를 키운 막대 → 막대 **길이의 절대비교**는 지양.

### fig10 — `fig10_trajectory_vs_baselines.png`
**Every Codebook-Corrected FL Algorithm Crosses the Centralized Bound** *(라운드 궤적 + 기준선)*

- **무엇**: 5개 FL 알고리즘의 **codebook-corrected(보정 후) 라운드별 곡선** + 두 anchor 수평 기준선
  (Chronos-Bolt-small 46.6, NBEATSx centralized 49.4 — 굵은 점선).
- **해석**: **모든 FL 알고리즘이 보정 후 NBEATSx centralized 선 아래(~47.8)로 내려와** Chronos 수준에 근접 →
  "어떤 FL backbone이든 codebook이 centralized 상한을 넘긴다"는 backbone-agnostic 주장을 동역학으로 보여줌. 서사가 가장 강함.

### fig11 — `fig11_global_baseline_lollipop.png`
**Where RoundCB Stands vs Global Baselines** *(lollipop dot plot)*

- **무엇**: fig9와 같은 정렬·색이지만 막대 대신 stem + dot(errorbar 포함).
- **해석**: 메시지는 fig9와 동일하되 **truncated-bar 길이 왜곡 없이** 값 위치만 정직하게 비교. 요약 슬라이드용.

---

## fig12 — `fig12_communication_cost.png`
**Codebook Adds Negligible Communication on Top of Any FL Algorithm** *(통신비용, FL 관점)*

- **무엇**: per-client/round upload(KB, **linear**). 5개 FL 알고리즘의 가중치 업로드 막대 + codebook add-on 막대(단일 패널).
- **수치 산출**: `src/fl/round_aux.py`의 `comm_stats` 의미를 v09 실제 모델 state_dict로 계산 (해석적, 측정 로그 아님).
  - FedAvg / FedProx / Ditto = **274.5 KB** (full weights) · **FedRep = 225.6 KB** (encoder-only, head는 local) ·
    **FedProto = 282.5 KB** (full + prototype) · **Codebook(RoundCB) = 3.6 KB** (Stage-1 centroids+counts + Stage-3 잔차합+counts)
  - Codebook은 한 번의 weight update의 **≈1.3%** — linear 막대에서 거의 안 보이는 sliver.
- **해석**: codebook은 어떤 FL backbone 위에도 얹히는 **거의 공짜 add-on** — raw latent·부하 대신 **cluster 요약만** 전송하여
  통신을 ~1.3%만 늘리고 −5 PAPE lift. privacy 보존 + 경량성을 동시에 보여줌.
  *생성*: `12_make_presentation_figures.py` (round_aux.py 통신 회계: FedRep encoder-only, FedProto +prototype 반영).

---

### 발표 시 권장 흐름
1. **fig1** (baseline는 알고리즘 무관) → 2. **fig2 / fig2b** (codebook이 전부 끌어내림) →
3. **fig3** (lift 크기 정량화) → 4. **fig4** (peak↑ vs MAE 비용 무시 가능) →
5. **fig10** (global baseline 대비 위치 — centralized 추월) →
보조: **fig5/fig6** (라운드 동역학), **fig7** (메커니즘 직관), **fig9/fig11** (순위 요약),
**fig12** (통신 경량성 — "거의 공짜 보정" + privacy).
