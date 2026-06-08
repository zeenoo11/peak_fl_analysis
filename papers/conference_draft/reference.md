# References

FL_Peak_Project 참고문헌. **★ = conference draft(`presentation_final.md`)에서 핵심으로 직접 인용되는 항목.**

## A. 데이터셋 · 문제 동기

[1] ★ S. Barker, A. Mishra, D. Irwin, E. Cecchet, P. Shenoy, and J. Albrecht. *Smart\*: An Open Data Set and Tools for Enabling Research in Sustainable Homes.* SustKDD, 2012. — UMass Smart\* 2016 데이터셋 출처 (§2-1).

[2] ★ L. Jin, C. A. Spurlock, et al. *Investigating Underlying Drivers of Variability in Residential Energy Usage Patterns with Daily Load Shape Clustering of Smart Meter Data.* arXiv:2102.11027, 2021. — who-vs-when 동기: peak는 가구 간 클러스터 차이가 지배 (§2-2, §7).

[3] Y. Peng, et al. *Short-term Load Forecasting at Different Aggregation Levels with Predictability Analysis.* arXiv:1903.10679, 2019. — 가구 단위 예측의 stochasticity 상한.

[4] P. Emami, et al. *BuildingsBench: A Large-Scale Dataset of 900K Buildings for Foundation Models in Short-Term Load Forecasting.* NeurIPS Datasets and Benchmarks, 2023. — cold-start 부하 예측 벤치마크.

## B. 예측 Backbone · Neural Forecasting Baseline

[5] ★ K. G. Olivares, C. Challu, G. Marcjasz, R. Weron, and A. Dubrawski. *Neural Basis Expansion Analysis with Exogenous Variables: Forecasting Electricity Prices with NBEATSx.* International Journal of Forecasting 39(2), 2023. — backbone `MinimalNBEATSx`의 출처 (§4-0a).

[6] B. N. Oreshkin, D. Carpov, N. Chapados, and Y. Bengio. *N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting.* ICLR, 2020. — doubly-residual stacking의 원전.

[7] A. Zeng, M. Chen, L. Zhang, and Q. Xu. *Are Transformers Effective for Time Series Forecasting?* (DLinear). AAAI, 2023. — neural forecasting baseline (v04).

[8] C. Challu, K. G. Olivares, et al. *NHITS: Neural Hierarchical Interpolation for Time Series Forecasting.* AAAI, 2023. — neural baseline (v04).

[9] Y. Zhang and J. Yan. *Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting.* ICLR, 2023. — transformer baseline (v04).

## C. Foundation Model · Zero-Shot Baseline

[10] ★ A. Das, W. Kong, R. Sen, and Y. Zhou. *A Decoder-Only Foundation Model for Time-Series Forecasting* (TimesFM). ICML, 2024. — TSFM zero-shot baseline (§6).

[11] A. F. Ansari, et al. *Chronos: Learning the Language of Time Series.* arXiv:2403.07815, 2024. — foundation model baseline (v04).

## D. Federated Learning 알고리즘

[12] ★ H. B. McMahan, et al. *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg). AISTATS, 2017. — 핵심 FL 집계 (§4-0b, §5-1).

[13] ★ T. Li, A. K. Sahu, M. Zaheer, et al. *Federated Optimization in Heterogeneous Networks* (FedProx). MLSys, 2020. — FL backbone (§5-1).

[14] ★ L. Collins, H. Hassani, A. Mokhtari, and S. Shakkottai. *Exploiting Shared Representations for Personalized Federated Learning* (FedRep). ICML, 2021. — FL backbone, encoder/head 분리 (§5-1).

[15] ★ T. Li, S. Hu, A. Beirami, and V. Smith. *Ditto: Fair and Robust Federated Learning Through Personalization.* ICML, 2021. — FL backbone (§5-1).

[16] ★ Y. Tan, et al. *FedProto: Federated Prototype Learning across Heterogeneous Clients.* AAAI, 2022. — FL backbone 및 count-weighted prototype 집계 근거 (§4-1).

[17] M. Stallmann and A. Wilbik. *Towards Federated Clustering: A Federated Fuzzy c-Means Algorithm (FFCM).* arXiv:2201.07316, 2022. — federated clustering 선행.

[18] J. Tang, et al. *FedHiP: Heterogeneity-Invariant Personalized Federated Learning Through Closed-Form Solutions.* arXiv:2508.04470, 2025. — frozen-backbone + analytic head pFL framing (v02).

[19] P. P. Liang, et al. *Think Locally, Act Globally: Federated Learning with Local and Global Representations* (LG-FedAvg). arXiv:2001.01523, 2020. — 파라미터 분리 pFL.

[20] C. T. Dinh, N. Tran, and J. Nguyen. *Personalized Federated Learning with Moreau Envelopes* (pFedMe). NeurIPS, 2020. — pFL 비교군.

[21] A. Fallah, A. Mokhtari, and A. Ozdaglar. *Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach* (Per-FedAvg). NeurIPS, 2020. — meta-learning pFL.

[22] R. Dai, et al. *FedNH: Tackling Both Data Heterogeneity and Class Imbalance in Federated Learning via Class Prototypes.* AAAI, 2023. — prototype 기반 pFL.

[23] A. Ghosh, J. Chung, D. Yin, and K. Ramchandran. *An Efficient Framework for Clustered Federated Learning* (IFCA). NeurIPS, 2020. — clustered FL.

[24] F. Sattler, K.-R. Müller, and W. Samek. *Clustered Federated Learning.* IEEE Trans. on Neural Networks and Learning Systems, 2021. — clustered FL.

## E. Vector Quantization · Codebook · Clustering

[25] ★ A. van den Oord, O. Vinyals, and K. Kavukcuoglu. *Neural Discrete Representation Learning* (VQ-VAE). NeurIPS, 2017 (arXiv:1711.00937). — codebook/벡터 양자화 개념; in-forward VQ 대비 post-hoc 대조 (§4-1, §6-3).

[26] ★ D. Arthur and S. Vassilvitskii. *k-means++: The Advantages of Careful Seeding.* SODA, 2007. — 2-stage federated KMeans++ (A축) 알고리즘 (§4-1).

[27] H. Gui, X. Li, and X. Chen. *Vector Quantization Pretraining for EEG Time Series with Random Projection and Phase Alignment* (VQ-MTM). ICML, 2024. — 시계열에서의 VQ 활용 선행.

[28] ★ L. van der Maaten and G. Hinton. *Visualizing Data using t-SNE.* JMLR 9, 2008. — latent codebook 시각화 (Fig 7).

## F. Peak-aux Head · 적응

[29] X. Zhang, et al. *Seq2Peak: A Sequence-to-Peak Auxiliary Forecasting Framework.* CIKM, 2023. — peak-aux head 동기 (§5-3).

[30] E. J. Hu, et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022. — K-shot 개인화 adapter (v03).

## G. Privacy · Security

[31] ★ *Breaking Privacy in Federated Clustering: Perfect Input Reconstruction via Temporal Correlations* (Trajectory-Aware Reconstruction, TAR). arXiv:2511.07073, 2025. — 반복 centroid 공개 위험 → 1-shot/round-wise codebook 근거 (§4-1).
