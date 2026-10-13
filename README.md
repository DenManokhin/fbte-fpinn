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

Every script is run from the repository root and writes its figures and tables into `results/`
(created on the first run).

| Script | Paper | What it does |
|---|---|---|
| `01_fig3_manufactured.py` | Table 1, Fig. 7 | Forward problem against a manufactured solution |
| `02_fig4_forward_vs_fdm.py` | Table 2, Fig. 8 | Forward problem against the implicit FDM solution |
| `03_fig5_grid_density.py` | Fig. 9 | fPINN error when the collocation grid is coarsened by k |
| `04_fig6_7_inverse.py` | Figs. 10–11 | Recovery of α and β, single-stage and multi-stage training |
| `05_fig8_9_inverse_noise.py` | Fig. 13 | Recovery of α and β from noisy data, SNR = 50, 20, 10, 5 |
| `06_complexity_grid.py` | Figs. 3–4 | Training time (CPU and GPU) and GPU memory vs grid size |
| `07_complexity_architecture.py` | Fig. 5 | Training time and memory vs network size |
| `08_complexity_optimizer.py` | – | Adam vs Adam + L-BFGS |
| `09_complexity_hardware.py` | – | CPU vs GPU training and inference time |
| `10_complexity_fdm_compare.py` | Fig. 6 | FDM vs fPINN computation time |

Command line options:

* `--cpu` (or `FORCE_CPU=1`) runs on the CPU even when a GPU is available.
* `--seed N` (scripts 04 and 05) sets the random seed, default 42. The parameter histories are saved
  to `results/04_inverse_{single,multi}_seed{N}.npz` and `results/05_inverse_noise_seed{N}.npz`,
  so results over several seeds can be aggregated, e.g.
  `for s in 1 2 3 4 5; do python 05_fig8_9_inverse_noise.py --seed $s; done`.

On Apple Silicon (MPS) the scripts run in float32, elsewhere in float64.

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
