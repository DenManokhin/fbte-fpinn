# fPINN for the fractional Bloch–Torrey equation

Code for the paper on solving the 1D fractional Bloch–Torrey equation (FBTE) with a fractional
physics-informed neural network (fPINN): the forward problem, the inverse problem of recovering the
fractional orders α and β, and the computational cost experiments.

## Setup

The pinned versions in `requirements.txt` need Python 3.9–3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch is installed from PyPI. For a CUDA build, install `torch` from the PyTorch index first
(see https://pytorch.org/get-started/locally/) and then the rest of the requirements.

## Running the experiments

To run all experiments of the paper and collect the tables:

```bash
python run_all.py                  # seeds 42 1 2 3 4 for the inverse problem
python run_all.py --only 04 05     # selected scripts
python run_all.py --tables-only    # rebuild the tables from existing results
```

Logs go to `results/logs/`, Tables 1–2 and the mean ± std of the recovered α and β to
`results/tables.md` and `results/table_inverse.csv`. See `python run_all.py --help` for all options.

Every script is run from the repository root and writes its figures and tables into `results/`
(created on the first run).

| Script | Paper | What it does |
|---|---|---|
| `01_fig3_manufactured.py` | Table 1, Fig. 7 | Forward problem against a manufactured solution |
| `02_fig4_forward_vs_fdm.py` | Table 2, Fig. 8 | Forward problem against the implicit FDM solution |
| `03_fig5_grid_density.py` | Fig. 9 | fPINN error when the collocation grid is coarsened by k |
| `04_fig6_7_inverse.py` | Figs. 10–11 | Recovery of α and β, single-stage and multi-stage training |
| `05_fig8_9_inverse_noise.py` | Figs. 12–13 | Noisy data example and recovery of α and β from noisy data, SNR = 50, 20, 10, 5 |
| `06_complexity_grid.py` | Figs. 3–4 | Training time (CPU and GPU) and GPU memory vs grid size |
| `07_complexity_architecture.py` | Fig. 5 | Training time and memory vs network size |
| `08_complexity_optimizer.py` | – | Adam vs Adam + L-BFGS |
| `09_complexity_hardware.py` | – | CPU vs GPU training and inference time |
| `10_complexity_fdm_compare.py` | Fig. 6 | FDM vs fPINN computation time |
| `11_inverse_fdm_baseline.py` | – | Baseline for the inverse problem: FDM + nonlinear least squares, on the data of script 05 (`--fpinn TAG` for a non-default setting) and over 50 noise realisations per SNR (`--realisations N`) |
| `sweep_noise.py` | – | Runs script 05 with several training settings and seeds in parallel |

Command line options:

* `--cpu` (or `FORCE_CPU=1`) runs on the CPU even when a GPU is available.
* `--seed N` (scripts 04 and 05) sets the random seed, default 42. The parameter histories are saved
  to `results/04_inverse_{single,multi}_seed{N}.npz` and `results/05_inverse_noise_seed{N}.npz`,
  so results over several seeds can be aggregated, e.g.
  `for s in 1 2 3 4 5; do python 05_fig8_9_inverse_noise.py --seed $s; done`.

On Apple Silicon (MPS) the scripts run in float32, elsewhere in float64.

### Noise robustness of the inverse problem (script 05 options)

With no options script 05 runs the paper's setting. The options below change the training to study
why the fPINN fails at SNR 10 and 5 (`python 05_fig8_9_inverse_noise.py --help` lists them all):
`--w-data`, `--w-pde`, `--w-pde-start` (PDE weight ramp), `--stage1`, `--stage2`, `--lbfgs N`
(L-BFGS after stage 2), `--width`, `--depth`, `--l2`, `--ic-bc` (loss on the known IC/BC),
`--val-frac` (early stopping on held-out points) and `--snrs`. A non-default setting writes to
`results/05_inverse_noise_<tag>_seed{N}.npz` (`--tag`, or a tag built from the options), so the
paper's files are not overwritten. The noise is the same as in the paper's runs for the same seed.

`sweep_noise.py` runs several settings in parallel, and `11_inverse_fdm_baseline.py --fpinn <tag>`
fits FDM + least squares to the same data and writes the comparison to `results/11_same_data_<tag>.md`:

```bash
python sweep_noise.py --jobs 3 --seeds 1 2 3 4 5 --snrs 50 20 10 5 inf --config wd1_lbfgs="--w-data 1 --lbfgs 5000"
python 11_inverse_fdm_baseline.py --fpinn wd1_lbfgs
```

Why the paper's setting fails at low SNR: when the network reproduces the noise, the PDE residual
evaluated on it is smallest for the lowest orders (alpha -> 0, beta -> 1, the bounds of the sigmoid),
because they amplify the noise least. With data weight 100 and PDE weight 1 the network does fit the
noise. With equal weights (`--w-data 1`) the PDE keeps the fit smooth, but Adam then converges slowly
(beta about 1.49 on clean data after 15000 iterations), which an L-BFGS stage fixes (`--lbfgs 5000`;
it runs on the CPU in float64, because in float32 the line search stalls).

Recovered alpha, beta (mean ± std over seeds 1–5, on the 16 × 25 grid, true alpha = 0.8, beta = 1.6);
FDM-LS is fitted to the same noisy data as the fPINN:

| Case | fPINN, paper setting¹ | fPINN `--w-data 1 --lbfgs 5000` | FDM + least squares |
|---|---|---|---|
| clean data | 0.798 ± 0.001, 1.591 ± 0.004 | 0.799 ± 0.001, 1.597 ± 0.002 | 0.800, 1.600 |
| SNR 50 | 0.795 ± 0.012, 1.584 ± 0.015 | 0.800 ± 0.007, 1.597 ± 0.016 | 0.802 ± 0.006, 1.594 ± 0.012 |
| SNR 20 | 0.777 ± 0.022, 1.521 ± 0.056 | 0.804 ± 0.024, 1.606 ± 0.039 | 0.810 ± 0.015, 1.611 ± 0.021 |
| SNR 10 | 0.008 ± 0.008, 1.001 ± 0.000 | 0.803 ± 0.056, 1.629 ± 0.127 | 0.808 ± 0.046, 1.642 ± 0.102 |
| SNR 5 | 0.001 ± 0.000, 1.001 ± 0.000 | 0.754 ± 0.096, 1.501 ± 0.138 | 0.792 ± 0.071, 1.587 ± 0.096 |

¹ seeds 42, 1–4 (`results/table_inverse.csv`); the noise for a seed is the same in all settings.

## Conventions

* The equation is eq. (2) of the paper, split into real and imaginary parts (C = γG, R = 1/T₂):

  ```
  D_t^α u =  C x v − R u + D ∂^β u/∂|x|^β
  D_t^α v = −C x u − R v + D ∂^β v/∂|x|^β
  ```

  The residual of this system is implemented once, in `src/physics.py::fbte_residual`, and used by
  the PINN models, the manufactured-solution test and the inverse problem. The FDM solver
  (`src/fdm_solver.py`) discretizes the same system.
* `get_riesz_matrix_np` and `torch_riesz_matrix` return the matrix of the Riesz derivative itself,
  eq. (13), which is negative definite: for β = 2 it is the usual `[1, −2, 1]/h²` Laplacian.
* The PDE residual is not enforced at t = 0 and at the Dirichlet nodes x = ±1, where the solution is
  fixed by the initial and boundary conditions (`src/physics.py::interior_mask`).
* The inverse problem (scripts 04, 05) uses the 16 × 25 grid described in the paper
  (`INVERSE_GRID` in `src/physics.py`); the forward problem uses 64 × 100.
* Noise is added as in eq. (28): SNR = max|M| / σ, with the same σ for the real and imaginary parts.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests
```

The tests check the L1 and Riesz matrices against known derivatives, the agreement of the NumPy and
PyTorch operators, and that the FDM solution has zero PINN residual at the interior nodes.
