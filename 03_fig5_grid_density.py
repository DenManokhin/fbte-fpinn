import os
import sys
import numpy as np
import torch
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from src.physics import PARAMS
from src.fdm_solver import FDMSolver
from src.models import ForwardGridData
from src.utils import select_device, set_reproducible, configure_precision, ensure_results_dir

device = select_device()
configure_precision(device)
ensure_results_dir()

# k divides the number of points in each dimension: 64x100, 32x50, 16x25, 8x12, 4x6
K_VALUES = [1, 2, 4, 8, 16]
# On very coarse grids the result depends on the network initialisation, so every k is
# trained with several seeds. "--seeds N" changes how many (default 5: seeds 42..46).
N_SEEDS = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 5
SEEDS = [42 + i for i in range(N_SEEDS)]


def grid_label(k):
    return f"{PARAMS['N_SPACE'] // k}x{PARAMS['N_TIME'] // k}"


def train_fpinn(p, X_flat_dense, seed):
    set_reproducible(seed)
    x_vals = np.linspace(p["X_RANGE"][0], p["X_RANGE"][1], p["N_SPACE"])
    t_vals = np.linspace(p["T_RANGE"][0], p["T_RANGE"][1], p["N_TIME"])
    XX_k, TT_k = np.meshgrid(x_vals, t_vals)
    X_flat_k = np.hstack((XX_k.flatten()[:, None], TT_k.flatten()[:, None])).astype(dde.config.real(np))

    data = ForwardGridData(torch.tensor(X_flat_k, dtype=torch.get_default_dtype(), device=device), p, device)
    net = dde.nn.FNN([2] + [64]*4 + [2], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3, loss_weights=[100, 100, 1])
    model.train(iterations=10000, display_every=5000)

    model.compile("adam", lr=1e-4, loss_weights=[100, 100, 1])
    model.train(iterations=5000, display_every=5000)

    y_pred = model.predict(X_flat_dense)
    u_pred = y_pred[:, 0].reshape(PARAMS["N_TIME"], PARAMS["N_SPACE"])
    v_pred = y_pred[:, 1].reshape(PARAMS["N_TIME"], PARAMS["N_SPACE"])
    return u_pred, v_pred


def run_experiment():
    print(f"Using compute device: {device}, seeds: {SEEDS}")

    # Ground truth on the dense grid
    x_g, t_g, u_fdm, v_fdm = FDMSolver(PARAMS).solve()
    XX, TT = np.meshgrid(x_g, t_g)
    X_flat_dense = np.hstack((XX.flatten()[:, None], TT.flatten()[:, None])).astype(dde.config.real(np))
    pts = np.column_stack([TT.ravel(), XX.ravel()])

    n_k, n_s = len(K_VALUES), len(SEEDS)
    mse_u = np.zeros((n_k, n_s))
    mse_v = np.zeros((n_k, n_s))
    # Baseline: the same FDM scheme solved on the coarse grid and linearly interpolated
    # onto the dense grid, i.e. what a mesh-based method gives at the same resolution
    fdm_mse_u = np.zeros(n_k)
    fdm_mse_v = np.zeros(n_k)

    for i, k in enumerate(K_VALUES):
        p = PARAMS.copy()
        p["N_SPACE"] = p["N_SPACE"] // k
        p["N_TIME"] = p["N_TIME"] // k
        print(f"\n--- k={k}: collocation grid {grid_label(k)} ---")

        x_c, t_c, u_c, v_c = FDMSolver(p).solve()
        u_c_dense = RegularGridInterpolator((t_c, x_c), u_c)(pts).reshape(u_fdm.shape)
        v_c_dense = RegularGridInterpolator((t_c, x_c), v_c)(pts).reshape(v_fdm.shape)
        fdm_mse_u[i] = np.mean((u_fdm - u_c_dense)**2)
        fdm_mse_v[i] = np.mean((v_fdm - v_c_dense)**2)

        for j, seed in enumerate(SEEDS):
            u_pred, v_pred = train_fpinn(p, X_flat_dense, seed)
            mse_u[i, j] = np.mean((u_fdm - u_pred)**2)
            mse_v[i, j] = np.mean((v_fdm - v_pred)**2)
            print(f"k={k}, seed={seed} -> fPINN MSE(u): {mse_u[i, j]:.2e}, MSE(v): {mse_v[i, j]:.2e}")

        print(f"k={k} -> coarse FDM MSE(u): {fdm_mse_u[i]:.2e}, MSE(v): {fdm_mse_v[i]:.2e}")

    np.savez("results/03_fig5_grid_density.npz", k=np.array(K_VALUES), seeds=np.array(SEEDS),
             fpinn_mse_u=mse_u, fpinn_mse_v=mse_v, fdm_mse_u=fdm_mse_u, fdm_mse_v=fdm_mse_v)

    print("\nfPINN: median [min, max] over seeds; FDM: coarse solution interpolated to 64x100")
    print("   k   grid    fPINN u                          fPINN v                          FDM u      FDM v")
    for i, k in enumerate(K_VALUES):
        cu = f"{np.median(mse_u[i]):.2e} [{mse_u[i].min():.2e}, {mse_u[i].max():.2e}]"
        cv = f"{np.median(mse_v[i]):.2e} [{mse_v[i].min():.2e}, {mse_v[i].max():.2e}]"
        print(f"{k:4d}   {grid_label(k):6s}  {cu}   {cv}   {fdm_mse_u[i]:.2e}   {fdm_mse_v[i]:.2e}")

    plt.figure(figsize=(8, 6))
    for mse, fdm, color, marker, name in [(mse_u, fdm_mse_u, 'C0', 'o', 'u (real)'),
                                          (mse_v, fdm_mse_v, 'C1', 's', 'v (imag)')]:
        plt.plot(K_VALUES, np.median(mse, axis=1), marker=marker, color=color, label=f'fPINN, {name}')
        if n_s > 1:
            plt.fill_between(K_VALUES, mse.min(axis=1), mse.max(axis=1), color=color, alpha=0.2)
        # k=1 is the reference grid itself, so the baseline error there is exactly zero
        plt.plot(K_VALUES[1:], fdm[1:], marker=marker, color=color, linestyle='--', mfc='none',
                 label=f'coarse FDM + interpolation, {name}')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xticks(K_VALUES, [f"{k}\n{grid_label(k)}" for k in K_VALUES])
    plt.xlabel('Grid reduction factor k (collocation grid)')
    plt.ylabel('MSE vs dense 64x100 FDM solution')
    title = 'MSE vs Grid Density Reduction'
    if n_s > 1:
        title += f' (median and range over {n_s} seeds)'
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', alpha=0.4)
    plt.savefig("results/03_fig5_grid_density.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    run_experiment()
