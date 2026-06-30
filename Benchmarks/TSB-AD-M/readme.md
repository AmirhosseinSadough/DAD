## Reproducing results on TSB-AD-M

Our method integrates into the [TSB-AD](https://github.com/TheDatumOrg/TSB-AD) codebase with three steps:

1. **Add the model.** Copy `DAD.py` into `TSB_AD/models/`.
2. **Register it.** Replace `TSB_AD/model_wrapper.py` and `TSB_AD/HP_list.py` with the versions provided in this repository. These register DAD and DAD_Auto with the benchmark's dispatcher and hyperparameter list.
3. **Run the benchmark.** Follow the standard TSB-AD-M instructions to evaluate.

**Note on DAD_Auto.** Unlike the manually tuned DAD variant, DAD_Auto requires no hyperparameter tuning. It self-initializes via SearchLR from a short warm-up prefix of each series and is therefore run directly on the 180-series evaluation set, skipping the benchmark's tuning protocol.

For full details on the benchmark, its datasets, and evaluation measures, see the TSB-AD paper, Liu and Paparrizos, *The Elephant in the Room: Towards a Reliable Time-Series Anomaly Detection Benchmark*, NeurIPS 2024 ([paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3f3c6907262e5d6823856c1 d6fc324-Abstract-Datasets_and_Benchmarks_Track.html)), and the [TSB-AD repository](https://github.com/TheDatumOrg/TSB-AD).