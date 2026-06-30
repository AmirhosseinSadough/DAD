# DAD: Real-Time Decorrelation-Based Anomaly Detection for Multivariate Time Series

Official implementation of **DAD**, an unsupervised, real-time anomaly detector for multivariate time series.

📄 **Paper:** [Real-Time Decorrelation-Based Anomaly Detection for Multivariate Time Series](https://arxiv.org/abs/2507.07559) (arXiv:2507.07559)

You can cite this work as follows:

```bibtex
@misc{sadough2025realtimedecorrelationbasedanomalydetection,
      title={Real-Time Decorrelation-Based Anomaly Detection for Multivariate Time Series}, 
      author={Amirhossein Sadough and Mahyar Shahsavari and Mark Wijtvliet and Marcel van Gerven},
      year={2025},
      eprint={2507.07559},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.07559}, 
}
```

## Overview

Real-time anomaly detection on multivariate sensor streams, common in the (Industrial) Internet of Things, must run under strict compute and memory budgets, process each observation as it arrives, and stay effective as the number of channels grows. DAD meets these requirements by continuously learning a **decorrelation matrix** that captures the evolving correlation structure of the stream. It updates sample-wise without storing past observations and flags anomalies from the residual correlations that emerge when the learned structure no longer explains the incoming data, while adapting to nominal distributional drift through a single interpretable learning-rate parameter.

To remove manual configuration, we introduce **SearchLR**, a label-free self-initialization procedure that estimates the learning rate and initial decorrelation matrix from a short warm-up prefix of each stream, so DAD deploys with no offline training, no labeled data, and no manual hyperparameter tuning.

We validate DAD on synthetic, real-world tabular, and real-world streaming (TSB-AD-M) benchmarks, where it achieves a leading accuracy-efficiency trade-off, ranking among the top detectors while operating at up to several orders of magnitude fewer computations and parameters than competing methods.

## Repository structure

- **[`Implementation/`](Implementation)** — `DAD.py`, the core algorithm, including the online decorrelation update and the SearchLR initialization.
- **[`Benchmarks/Synthetic&Tabular/`](Benchmarks/Synthetic%26Tabular)** — reproduces the synthetic and tabular results, built on an extended outlier-detection framework. Setup and run commands are in its [README](Benchmarks/Synthetic%26Tabular).
- **[`Benchmarks/TSB-AD-M/`](Benchmarks/TSB-AD-M)** — reproduces the streaming results by integrating DAD into the [TSB-AD](https://github.com/TheDatumOrg/TSB-AD) benchmark. Integration steps and the figure/table notebook are in its [README](Benchmarks/TSB-AD-M).

## Reproducing the paper

The two benchmark folders cover the three evaluation settings from the paper:

| Setting | Folder |
|---|---|
| Synthetic | [`Benchmarks/Synthetic&Tabular/`](Benchmarks/Synthetic%26Tabular) |
| Real-world tabular | [`Benchmarks/Synthetic&Tabular/`](Benchmarks/Synthetic%26Tabular) |
| Real-world streaming (TSB-AD-M) | [`Benchmarks/TSB-AD-M/`](Benchmarks/TSB-AD-M) |

Follow the README in each folder for environment setup, evaluation commands, and figure/table reproduction.