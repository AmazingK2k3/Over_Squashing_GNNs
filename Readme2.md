
# Rewiring Experiments for Graph Neural Networks

This repository is a fork of Errica et al.'s ICLR 2020 codebase, [A Fair Comparison of Graph Neural Networks for Graph Classification](https://openreview.net/pdf?id=HygDF6NFPB). The original project provides datasets, model implementations, and experiment scripts for a controlled comparison of graph neural network architectures on graph classification benchmarks.

This fork extends the original codebase with graph rewiring experiments aimed at studying over-squashing in GNNs. In particular, it adds support for storing both the original graph edges and a rewired edge set, then using the rewired edges at selected layers of supported models.

## Original Citation

If you use the original benchmark code, please cite:

```bibtex
@inproceedings{errica_fair_2020,
    title = {A fair comparison of graph neural networks for graph classification},
    booktitle = {Proceedings of the 8th {International} {Conference} on {Learning} {Representations} ({ICLR})},
    author = {Errica, Federico and Podda, Marco and Bacciu, Davide and Micheli, Alessio},
    year = {2020}
}
```

## What This Fork Adds

- Rewiring-aware preprocessing in `PrepareDatasets.py`.
- Complement-graph rewiring utilities in `rewire_functions.py`.
- Support for `rewired_edge_index` in processed PyTorch Geometric `Data` objects.
- Model changes so selected architectures can switch from `edge_index` to `rewired_edge_index` at a chosen layer.
- Fixed experiment configs for rewiring runs, including:
  - `config_fixed_gin.yml`,
  - `config_fixed_DGCNN.yml`,
  - `config_DiffPool_fixed.yml`,
  - `config_GraphSAGE_fixed.yml`.
- A `skip_model_selection` shortcut for faster fixed-config runs.
- Automated experiment launchers:
  - `autolaunch.py` for running several experiments sequentially,
  - `iterative_rewire.py` for generating datasets with varying rewired-edge percentages,
  - `iterative_runs.py` for running experiments over those percentages.
- `summarize_results.py` for collecting nested-CV results into a CSV file.

## Supported Models

The rewiring changes are currently integrated into:

- GIN
- GraphSAGE
- DGCNN
- DiffPool

The model configs expose two important rewiring options:

- `rewire_for_all_layers` or `rewire_all_layers`: set to `1` to use rewired edges in all supported layers, or `0` to rewire only one selected layer.
- `rewired_layer`: chooses which layer should use rewired edges when all-layer rewiring is disabled. A value of `1` means the final layer.

Note: DiffPool contains an in-code note that penultimate-layer rewiring is not fully supported yet.

## Skipping Model Selection

This fork adds a `skip_model_selection` path for experiments where the config should be used directly instead of running the full inner model-selection loop.

The shortcut is implemented in:

- `evaluation/model_selection/K_Fold_Selection.py`
- `evaluation/model_selection/HoldOutSelector.py`

When enabled, the selector returns the first generated model configuration directly, with validation and training selection scores set to `None`. The K-fold assessment path in `evaluation/risk_assessment/K_Fold_Assessment.py` currently calls model selection with `skip_model_selection=True`, so fixed-config rewiring experiments can avoid the expensive inner cross-validation step.

Some fixed config files also include:

```yaml
skip_model_selection:
  - 1
```

This is used as part of the fixed rewiring experiment setup and documents that the run is intended to use the supplied hyperparameters directly.

## Installation

The original repository provides two installation scripts.

Clone the repository and enter it:

```bash
git clone <your-fork-url>
cd Over_Squashing_GNNs-1
```

If you want to recreate the original environment used in the ICLR 2020 paper, run:

```bash
source install_original.sh [<your_cuda_version>]
```

where `<your_cuda_version>` can be `cpu`, `cu92`, `cu100`, or `cu101`.

For a newer environment, run:

```bash
source install.sh [<your_cuda_version>] [<your_pytorch_version>]
```

where `<your_pytorch_version>` should be `>= 2.0.1`, and `<your_cuda_version>` can be `cpu`, `cu116`, `cu117`, or `cu118`. If no arguments are supplied, the script defaults to PyTorch `2.0.1` and CPU.

Make sure your Python, PyTorch, PyTorch Geometric, and CUDA versions are mutually compatible.

## Dataset Preparation

The original workflow preprocesses TU datasets into the format expected by the experiment runner.

Chemical datasets:

```bash
python PrepareDatasets.py DATA/CHEMICAL --dataset-name <DATASET> --outer-k 10
```

Social datasets with constant node features:

```bash
python PrepareDatasets.py DATA/SOCIAL_1 --dataset-name <DATASET> --use-one --outer-k 10
```

Social datasets with degree node features:

```bash
python PrepareDatasets.py DATA/SOCIAL_DEGREE --dataset-name <DATASET> --use-degree --outer-k 10
```

Expected dataset organization:

```text
DATA/CHEMICAL:
    NCI1
    DD
    ENZYMES
    PROTEINS

DATA/SOCIAL_1 or DATA/SOCIAL_DEGREE:
    IMDB-BINARY
    IMDB-MULTI
    REDDIT-BINARY
    REDDIT-MULTI-5K
    COLLAB
```

The repository also includes predefined split files under `data_splits/`. To reproduce the original benchmark protocol, replace the generated split files with the corresponding files from `data_splits/`.

## Preparing Rewired Datasets

To preprocess a dataset and attach a complement-graph rewired edge set, use `--use-rewired --rewiring-strategy complement`.

Example using complement-graph rewiring:

```bash
python PrepareDatasets.py DATA/CHEMICAL --dataset-name PROTEINS --outer-k 10 --use-rewired --rewiring-strategy complement
```

Example for a social dataset:

```bash
python PrepareDatasets.py DATA/SOCIAL_1 --dataset-name REDDIT-BINARY --use-one --outer-k 10 --use-rewired --rewiring-strategy complement
```

The preprocessing script keeps the original graph in `edge_index` and stores the rewired graph in `rewired_edge_index`.

## Running Experiments

After preprocessing, copy or place the processed dataset where `Launch_Experiments.py` expects it. The original README uses the following pattern:

```bash
cp -r DATA/[CHEMICAL|SOCIAL_1|SOCIAL_DEGREE]/<DATASET> DATA
python Launch_Experiments.py --config-file <CONFIG> --dataset-name <DATASET> --result-folder <RESULT_FOLDER> --debug
```

Examples:

```bash
python Launch_Experiments.py --config-file config_fixed_gin.yml --dataset-name NCI1 --result-folder results --debug
```

```bash
python Launch_Experiments.py --config-file config_DiffPool_fixed.yml --dataset-name ENZYMES --result-folder results --debug
```

CUDA is supported with the `--debug` option. Parallel multi-GPU execution is not provided by the original codebase.

## Running Multiple Experiments

`autolaunch.py` runs several experiment configurations sequentially and writes a JSON log to `experiment_log.json`.

Use a JSON config file:

```bash
python autolaunch.py --config experiments.json
```

Or edit the experiment list directly in `autolaunch.py` and run:

```bash
python autolaunch.py
```

## Edge-Percentage Experiments

`iterative_rewire.py` generates multiple rewired datasets by sampling different percentages of complement edges.

Full range from 30% to 100% in 5% increments:

```bash
python iterative_rewire.py --data-dir DATA/CHEMICAL --dataset-name NCI1 --seed 42
```

Single percentage:

```bash
python iterative_rewire.py --data-dir DATA/CHEMICAL --dataset-name ENZYMES --single-p 0.50 --seed 42
```

Custom range:

```bash
python iterative_rewire.py --data-dir DATA/CHEMICAL --dataset-name NCI1 --start-p 0.40 --end-p 0.80 --p-increment 0.10
```

The generated files are saved under:

```text
DATA/CHEMICAL/<DATASET>_rewired_seed<SEED>/
```

`iterative_runs.py` then copies each generated `.pt` file into the location expected by `Launch_Experiments.py`, runs training, stores the results, and can plot accuracy versus edge percentage.

Example:

```bash
python iterative_runs.py --dataset NCI1 --config-file config_fixed_gin.yml --debug
```

Run a single percentage:

```bash
python iterative_runs.py --dataset NCI1 --config-file config_fixed_gin.yml --percentage 0.50 --debug
```

Plot from an existing results JSON:

```bash
python iterative_runs.py --dataset NCI1 --config-file config_fixed_gin.yml --plot-only RESULTS_EDGE_EXP/<results-file>.json
```

## Summarizing Results

Use `summarize_results.py` to collect nested cross-validation results into a CSV file:

```bash
python summarize_results.py <RESULTS_DIR> --out_file summary.csv
```

The summary includes model name, dataset, rewiring metadata inferred from folder names, selected hyperparameters, and train/test scores.

## Troubleshooting

If PyTorch spawns too many CPU threads, set:

```bash
export OMP_NUM_THREADS=1
```

For installation issues, check that the PyTorch, PyTorch Geometric, CUDA, and Python versions match. The original installation scripts are sensitive to these versions.

## License

This project inherits the license of the original repository. See `LICENSE` for details.
