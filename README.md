# WEANet: Wavelet Inductive Bias meets Polyphonic Convolutional Core for Robust Time Series Analysis

This repository is the official implementation of the paper "WEANet: Wavelet Inductive Bias meets Polyphonic Convolutional Core for Robust Time Series Analysis"

Overview

Time series data is inherently characterized by strong temporal correlations, non-stationarity, and multi-scale dynamics. Conventional general-purpose initialization strategies (e.g., Xavier, Kaiming) assume i.i.d. features, leading to unstructured spectral responses that force the network to learn basic signal properties from scratch. 

**WEANet** introduces a principled architecture that embeds a rigorous mathematical structural prior into the network:
1. **Wavelet-Initialized Front-End (WFE):** Replaces random initialization with multi-wavelet coefficients constrained by the discrete admissibility condition, ensuring the network starts in a highly favorable optimization basin.
2. **Polyphonic Convolutional Core (PCC):** A lightweight bottleneck architecture ($K=2$) that operates on the decomposed frequency bands with local temporal receptive fields ($k=3$), perfectly decoupling rigorous frequency decomposition from flexible spatial-temporal modeling.
3. **Dual-Objective Reconstruction Loss ($L_{recon}$):** A learnable Inverse Discrete Wavelet Transform (IDWT) path exclusively used during training to quantify and preserve the measurable inductive bias, preventing feature degradation.

> **Note on Deployability:** The IDWT reconstruction branch is strictly confined to the training phase. During inference, WEANet operates as a blazing-fast, purely feed-forward fully-convolutional network.

---


Due to space and data sharing constraints, only standardized example data for reproduction is provided in data/.
Please download the full datasets (ETTh1, ETTm1, HeartBeat, EthanolConcentration, etc.) from standard open sources.

Put the datasets into data/ and provide the path to main.py. Standard data normalization and random seeds are handled by data_provider/data_loader.py to ensure completely fair comparisons with baselines.

