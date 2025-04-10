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
```

# Installation

To run the various benchmarking methods, you first need to install the required dependencies. For compatibility and isolation of dependencies, we recommend creating a new Anaconda environment using the provided environment.yml file:

1. Create the environment:
```
conda env create -f environment.yml
```

2. Activate the environment:
```
conda activate UADbenchmark
```

# Running the benchmark
Running Anomaly Detection Methods on the Benchmark
You can run the benchmarking evaluations using the following commands:

HPT-based Benchmarking (Hyperparameter Tuning-based):
```
python main.py benchmark hpt
```

Default Hyperparameter Benchmarking:
```
python main.py benchmark default
```
Peak and Average Performance Evaluation (over a set of hyperparameters per method):
```
python main.py benchmark max_mean
```
Test on Synthetic Dataset:
```
python main.py synthetic
```
Test on DAMADICS Dataset (Industrial IoT dataset):
```
python main.py damadics
```

# Generating Figures and Tables

To produce performance figures and tables as described in our paper, use the following commands:

HPT-based Performance:
```
python results.py benchmark hpt
```

Average Performance:
```
python results.py benchmark average
```

Peak Performance:
```
python results.py benchmark maximum
```

Default Hyperparameter Performance:
```
python results.py benchmark default
```