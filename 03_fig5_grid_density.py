import os
import numpy as np
import torch
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import matplotlib.pyplot as plt

from src.physics import PARAMS
from src.fdm_solver import FDMSolver
from src.models import ForwardGridData
from src.utils import select_device, set_reproducible

device = select_device()
set_reproducible(42)

def run_experiment():
    print(f"Using compute device: {device}")
    
    # Ground truth
    fdm = FDMSolver(PARAMS)
    x_g, t_g, u_fdm, v_fdm = fdm.solve()
    XX, TT = np.meshgrid(x_g, t_g)
    X_flat_dense = np.hstack((XX.flatten()[:, None], TT.flatten()[:, None])).astype(np.float32)

    k_values = [1, 2, 4, 8]
    mses_u = []
    mses_v = []

    for k in k_values:
        print(f"\n--- Training fPINN with grid downsampled by k={k} ---")
        p = PARAMS.copy()
        p["N_SPACE"] = p["N_SPACE"] // k
        p["N_TIME"] = p["N_TIME"] // k

        x_vals = np.linspace(p["X_RANGE"][0], p["X_RANGE"][1], p["N_SPACE"])
        t_vals = np.linspace(p["T_RANGE"][0], p["T_RANGE"][1], p["N_TIME"])
        XX_k, TT_k = np.meshgrid(x_vals, t_vals)
        X_flat_k = np.hstack((XX_k.flatten()[:, None], TT_k.flatten()[:, None])).astype(np.float32)

        data = ForwardGridData(torch.tensor(X_flat_k, dtype=torch.float32, device=device), p, device)
        net = dde.nn.FNN([2] + [60]*4 + [2], "tanh", "Glorot normal")
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
        
        print(f"k={k} -> MSE(u): {mse_u:.2e}, MSE(v): {mse_v:.2e}")

    plt.figure(figsize=(8, 6))
    plt.plot(k_values, mses_u, marker='o', label='MSE (Real part u)')
    plt.plot(k_values, mses_v, marker='s', label='MSE (Imag part v)')
    plt.xlabel('Grid density reduction factor (k)')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs Grid Density Reduction')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_experiment()
