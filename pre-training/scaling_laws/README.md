# Scaling Laws

This folder analyzes scaling behavior across compute, parameters, and tokens.

## What Is Implemented
- Loading cached experiment results from `data/` binaries.
- Plotting reference scaling curves for comparison.
- Interpreting scaling trends to guide model sizing.

## Notebooks
- `scaling_laws.ipynb`: produces scaling plots and summarizes observations.

## Figures
![Isoflop curve](resources/isoflop_curve.png)
![Isoflop curve points](resources/isoflop_curve_points_only.png)
![Minima params vs FLOPs](resources/minima_params_vs_flops_observed.png)
![Minima tokens vs params](resources/minima_tokens_vs_params_observed.png)
![Minima tokens vs FLOPs](resources/minima_tokens_vs_flops_observed.png)
