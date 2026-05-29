# Bootstrapped Exploration with Causal Reasoning: A Training Paradigm for Adaptive Forecasting Agent (BECRA)

BECRA is an agent training paradigm for adaptive time-series forecasting. Instead of hand-tuning a fixed pipeline for every dataset, BECRA learns **symbolic strategy lessons** through contrast-aware exploration and agent-level causal reasoning, without human-annotated supervision, and reuses them for **zero-shot, lesson-guided planning** on unseen datasets.

Real-world series are heterogeneous: they differ in volatility, seasonality, missingness, anomaly patterns, and cross-variate structure. No single forecasting model or preprocessing chain is universally optimal. BECRA treats forecasting as an **agent decision problem**: given dataset **meta-features**, the agent composes a multi-stage **toolchain** (imputation, anomaly handling, transformation, decomposition, normalization, and forecasting) and accumulates transferable knowledge that can be verified, stored, and retrieved at deployment time.

<p align="center">
  <img src="asset/framework_overview.jpg" alt="BECRA framework overview" width="90%">
  <br>
  <em>Overview of the four-stage BECRA cycle.</em>
</p>

---


## Citation
