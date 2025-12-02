**UA-PINN: Uncertainty-Aware PINN & DeepAR for Prognostics**

This repository implements and compares probabilistic time-series regression models (e.g., DeepAR) and a physics-constrained PINN-style model (`UA_PINN`). It also includes several classical machine learning baselines (see `base.py`). The project uses battery (NASA) and `SUN` datasets as examples, and provides data preprocessing, training, prediction, and evaluation pipelines.

Highlights:
- Uncertainty estimation: supports Gaussian (`g`) and Negative Binomial (`nb`) likelihoods for probabilistic prediction and sampling.
- PINN constraints: `UA_PINN` integrates dynamical constraints into training to improve robustness.
- Modular data loading: `dataset/datasets.py` supports concatenating multiple files, preprocessing, and a few-shot mode.

Main dependencies (examples):
- Python 3.8+
- `torch` (choose install according to GPU/CUDA availability)
- `numpy`, `pandas`, `scikit-learn`, `tqdm`, `matplotlib`

Installation example (Windows PowerShell):

```powershell
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn tqdm matplotlib
pip install torch           # choose the torch package suitable for your CUDA/CPU
```

You may also create a `requirements.txt` and run `pip install -r requirements.txt`.

Data organization:
- The `data/` folder contains example datasets under `NASA data/` and `SUN data/`. Data loading and preprocessing logic are implemented in `dataset/datasets.py`. To add a new dataset, follow the CSV format and extend `DataList`.

Quick start (training example):
- Train or evaluate a model with `main.py`:

```powershell
python main.py --dataset NASA --model UA_PINN --num_epochs 200 --n_samples 200 --train_batch_size 1024 --few_shot
```

- Batch experiment script: `run_main.py` calls `main.py` in subprocesses and can be adapted to run different parameter combinations.

Key scripts and modules:
- `main.py`: Training pipeline that supports `UA_PINN`, `DeepAR`, `MLP`, `CNN`; handles training, testing, saving results, and plotting.
- `run_main.py`: Simple example script to run experiments in batch using `subprocess`.
- `dataset/datasets.py`: Dataset loading, preprocessing, and `Dataset` / `PINNDataset` definitions.
- `model/model.py`: Implementations of `DeepAR` and `UA_PINN` (likelihoods, training and test methods).
- `model/CNN.py`, `model/MLP.py`: Simple CNN / MLP implementations for comparison.
- `utils/args.py`: Command-line argument parsing and saving (`args.json` written to `args.save_path`).
- `utils/util.py`, `utils/result.py`: Utility functions, result saving and plotting (see files for details).

Outputs and results:
- During training an `args.json` is generated (with run parameters), models are saved to `args.save_path/result/model.pth`, and plots and evaluation files are saved under `args.save_path`.

Common CLI arguments (`--help`):
- `--dataset`: dataset name (default `NASA`)
- `--model`: model name (`UA_PINN`, `DeepAR`, `CNN`, `MLP`)
- `--num_epochs`: number of training epochs
- `--train_batch_size`, `--test_batch_size`
- `--n_samples`: number of samples during testing for probabilistic models
- `--few_shot`: enable few-shot training (predefined few-shot file lists)

Extending the project:
- New model: add a model file under `model/`, implement `Train`, `test` or `predict` interfaces compatible with `main.py`.
- New dataset: add `get_<YourDataset>_list` in `dataset/DataList` and place CSV files under `data/`.

