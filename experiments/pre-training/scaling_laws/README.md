# Scaling Laws Experiments

This folder contains a self-contained scaling-law study for small decoder-only Transformers (about 1M to 50M parameters) trained on a subset of FineWeb. The main workflow lives in `scaling_laws.ipynb` and is designed for Google Colab, with minimal changes required for local runs.

## Contents

- `scaling_laws.ipynb`: End-to-end notebook including model definition, data processing, training loop, experiment grid, and plots.
- `experiments/`: TensorBoard event logs for each run.
- `resources/`: Generated plots used in the notebook.

## Experiment Grid

Runs are organized by compute budget and model width. The notebook uses:

- FLOP budgets: `3e14, 6e14, 1e15, 3e15, 6e15, 1e16, 3e16`
- Model sizes (`n_embedding`): `32, 64, 96, ... , 352` (step 32)

Each run is stored under:

- `experiments/experiment_{F}e14_{N}/logs/`

Where `{F}` is `flops/1e14` and `{N}` is `n_embedding`.

## Outputs

- TensorBoard event logs are stored under each experiment folder.
- Summary figures live in `resources/`: `isoflop_curve.png`, `isoflop_curve_points_only.png`, `minima_params_vs_flops_observed.png`, `minima_tokens_vs_flops_observed.png`, `minima_tokens_vs_params_observed.png`.

## Notes from the Notebook

- Dynamic batch sizes and gradient accumulation are used to maximize compute usage.
- Mixed precision training with `autocast` and `GradScaler` (disabled if `bf16`).
- AdamW optimizer with warmup (2% of steps) and cosine decay to 10% of peak LR.
- Runs limited to token/parameter ratios in `[1, 200]` to expose clear U-shaped curves.
