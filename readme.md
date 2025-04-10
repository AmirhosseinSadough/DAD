# DECODE: Real-Time Decorrelation-Based Anomaly Detection

This repository extends the benchmarking framework proposed in [Outlier Detection: Benchmarking Methods and Evaluation Criteria](https://jmlr.org/papers/v25/23-0570.html) and its corresponding GitHub repository: [outlierdetection](https://github.com/RoelBouman/outlierdetection). Our extension introduces evaluations based on a new benchmarking approach, **Hyperparameter Tuning-based (HPT-based)** benchmarking, along with default hyperparameter settings and peak performance evaluations.

Additionally, we have incorporated synthetic datasets and an IIoT dataset (DAMADICS) into the benchmark. The reference framework has been modified to accommodate these new additions.

You can cite this work as follows:

```bibtex
@article{DECODE_draft,
  author  = {Amirhossein Sadough and Mahyar Shahsavari and Mark Wijtvliet, Marcel van Gerven},
  title   = {DECODE: Real-Time Decorrelation-Based Anomaly Detection Method for Multivariate Time Series},
  journal = {},
  year    = {},
  volume  = {},
  number  = {},
  pages   = {},
  url     = {}
}

# Installation
To run the various benchmarking methods, you first need to install the required dependencies. For compatibility and isolation of dependencies, we recommend creating a new Anaconda environment using the provided environment.yml file:

