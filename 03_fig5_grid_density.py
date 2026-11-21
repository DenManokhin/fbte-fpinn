import os
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
set_reproducible(42)
ensure_results_dir()

def run_experiment():
    print(f"Using compute device: {device}")
    
    # Ground truth
    fdm = FDMSolver(PARAMS)
    x_g, t_g, u_fdm, v_fdm = fdm.solve()
    XX, TT = np.meshgrid(x_g, t_g)
    X_flat_dense = np.hstack((XX.flatten()[:, None], TT.flatten()[:, None])).astype(dde.config.real(np))

    # k divides the number of points in each dimension: 64x100, 32x50, 16x25, 8x12, 4x6
    k_values = [1, 2, 4, 8, 16]
    mses_u = []
    mses_v = []
    # Baseline: the same FDM scheme solved on the coarse grid and linearly interpolated
    # onto the dense grid, i.e. what a mesh-based method gives at the same resolution
    fdm_mses_u = []
    fdm_mses_v = []

    for k in k_values:
        p = PARAMS.copy()
        p["N_SPACE"] = p["N_SPACE"] // k
        p["N_TIME"] = p["N_TIME"] // k
        print(f"\n--- k={k}: collocation grid {p['N_SPACE']}x{p['N_TIME']} ---")

        x_c, t_c, u_c, v_c = FDMSolver(p).solve()
        pts = np.column_stack([TT.ravel(), XX.ravel()])
        u_c_dense = RegularGridInterpolator((t_c, x_c), u_c)(pts).reshape(u_fdm.shape)
        v_c_dense = RegularGridInterpolator((t_c, x_c), v_c)(pts).reshape(v_fdm.shape)
        fdm_mses_u.append(np.mean((u_fdm - u_c_dense)**2))
        fdm_mses_v.append(np.mean((v_fdm - v_c_dense)**2))

        x_vals = np.linspace(p["X_RANGE"][0], p["X_RANGE"][1], p["N_SPACE"])
        t_vals = np.linspace(p["T_RANGE"][0], p["T_RANGE"][1], p["N_TIME"])
        XX_k, TT_k = np.meshgrid(x_vals, t_vals)
        X_flat_k = np.hstack((XX_k.flatten()[:, None], TT_k.flatten()[:, None])).astype(dde.config.real(np))

        data = ForwardGridData(torch.tensor(X_flat_k, dtype=torch.get_default_dtype(), device=device), p, device)
        net = dde.nn.FNN([2] + [64]*4 + [2], "tanh", "Glorot normal")
        model = dde.Model(data, net)

        model.compile("adam", lr=1e-3, loss_weights=[100, 100, 1])
        model.train(iterations=10000)

        model.compile("adam", lr=1e-4, loss_weights=[100, 100, 1])
        model.train(iterations=5000)

        y_pred = model.predict(X_flat_dense)
        u_pred = y_pred[:, 0].reshape(PARAMS["N_TIME"], PARAMS["N_SPACE"])
        v_pred = y_pred[:, 1].reshape(PARAMS["N_TIME"], PARAMS["N_SPACE"])

        mse_u = np.mean((u_fdm - u_pred)**2)
        mse_v = np.mean((v_fdm - v_pred)**2)
        
        mses_u.append(mse_u)
        mses_v.append(mse_v)
        
        print(f"k={k} -> fPINN MSE(u): {mse_u:.2e}, MSE(v): {mse_v:.2e} | "
              f"coarse FDM MSE(u): {fdm_mses_u[-1]:.2e}, MSE(v): {fdm_mses_v[-1]:.2e}")

    np.savez("results/03_fig5_grid_density.npz", k=np.array(k_values),
             fpinn_mse_u=np.array(mses_u), fpinn_mse_v=np.array(mses_v),
             fdm_mse_u=np.array(fdm_mses_u), fdm_mse_v=np.array(fdm_mses_v))
    print("\n   k   grid      fPINN u    fPINN v    FDM u      FDM v")
    for i, k in enumerate(k_values):
        grid = f"{PARAMS['N_SPACE'] // k}x{PARAMS['N_TIME'] // k}"
        print(f"{k:4d}   {grid:8s}  {mses_u[i]:.2e}   {mses_v[i]:.2e}   "
              f"{fdm_mses_u[i]:.2e}   {fdm_mses_v[i]:.2e}")

    plt.figure(figsize=(8, 6))
    plt.plot(k_values, mses_u, marker='o', color='C0', label='fPINN, u (real)')
    plt.plot(k_values, mses_v, marker='s', color='C1', label='fPINN, v (imag)')
    plt.plot(k_values, fdm_mses_u, marker='o', color='C0', linestyle='--', label='coarse FDM + interpolation, u')
    plt.plot(k_values, fdm_mses_v, marker='s', color='C1', linestyle='--', label='coarse FDM + interpolation, v')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xticks(k_values, [f"{k}\n{PARAMS['N_SPACE'] // k}x{PARAMS['N_TIME'] // k}" for k in k_values])
    plt.xlabel('Grid reduction factor k (collocation grid)')
    plt.ylabel('MSE vs dense 64x100 FDM solution')
    plt.title('MSE vs Grid Density Reduction')
    plt.legend()
    plt.grid(True, which='both', alpha=0.4)
    plt.savefig("results/03_fig5_grid_density.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    run_experiment()
