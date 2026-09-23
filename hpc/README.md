# Thesis production scripts

These are the scripts that produced the results in the thesis: training on
the $32\times8$ lattice and the full analysis/plotting pipeline. They predate
the `SADflow` package and carry their own copies of the lattice, model and
loss code. They are kept here as they were run, for reproducibility; for new
work, use the package and `scripts/train.jl`.

| File | Purpose |
|---|---|
| `train_thesis.jl` | Standalone training; writes per-epoch models to `./models_free1/` and training metrics to `./data_free1/` |
| `plot_thesis.jl` | Loads those checkpoints and metrics and produces the thesis figures (training metrics, reweighted correlators, effective masses, signal-to-noise) |
| `grid_search.sh` | Submits one SLURM training job per point of the (κ, λ, learning rate, modes) grid |
| `plot_grid.sh` | Submits the matching plotting jobs |

`grid_search.sh` and `plot_grid.sh` submit `train.job` and `plot.job`. These
are cluster-specific SLURM wrappers (partition, modules, time limits) and are
not part of the repository; each simply runs the corresponding Julia script
with the arguments it receives.

All scripts are run from the repository root with `julia --project=.`. The
large ensembles they read (`priors/`, `priors_plot/`) are not included.
