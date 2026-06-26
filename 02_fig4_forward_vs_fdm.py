import os
import numpy as np
import torch
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import matplotlib.pyplot as plt

from src.physics import PARAMS
from src.fdm_solver import FDMSolver
from src.models import ForwardGridData
from src.utils import select_device, set_reproducible, configure_precision

device = select_device()
configure_precision(device)
set_reproducible(42)

def run_experiment():
    print(f"Using compute device: {device}")
    
    # 1. Get FDM Solution
    fdm = FDMSolver(PARAMS)
    x_g, t_g, u_fdm, v_fdm = fdm.solve()

    XX, TT = np.meshgrid(x_g, t_g)
    X_flat = np.hstack((XX.flatten()[:, None], TT.flatten()[:, None])).astype(dde.config.real(np))

    # 2. Train Forward fPINN
    print("\n[fPINN] Training Forward Model...")
    data = ForwardGridData(torch.tensor(X_flat, dtype=torch.get_default_dtype(), device=device), PARAMS, device)
    net = dde.nn.FNN([2] + [60]*4 + [2], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    # Weight the IC and BC higher
    model.compile("adam", lr=1e-3, loss_weights=[100, 100, 1])
    model.train(iterations=10000, display_every=1000)

    # Fine-tuning
    model.compile("adam", lr=1e-4, loss_weights=[100, 100, 1])
    model.train(iterations=5000, display_every=1000)

    # 3. Predict & Compare
    y_pred = model.predict(X_flat)
    u_pred = y_pred[:, 0].reshape(PARAMS["N_TIME"], PARAMS["N_SPACE"])
    v_pred = y_pred[:, 1].reshape(PARAMS["N_TIME"], PARAMS["N_SPACE"])

    # 4. Visualization
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    mid_idx = PARAMS["N_TIME"] // 2

    # Compare Real Field (u)
    axs[0, 0].plot(x_g, u_fdm[mid_idx, :], 'k-', linewidth=2, label='FDM')
    axs[0, 0].plot(x_g, u_pred[mid_idx, :], 'r--', linewidth=2, label='fPINN')
    axs[0, 0].set_title(f"Real Magnetization (u) at t={t_g[mid_idx]:.2f}")
    axs[0, 0].legend()

    # Compare Imaginary Field (v)
    axs[0, 1].plot(x_g, v_fdm[mid_idx, :], 'k-', linewidth=2, label='FDM')
    axs[0, 1].plot(x_g, v_pred[mid_idx, :], 'b--', linewidth=2, label='fPINN')
    axs[0, 1].set_title(f"Imaginary Magnetization (v) at t={t_g[mid_idx]:.2f}")
    axs[0, 1].legend()

    # Real Error Heatmap
    err_u = np.abs(u_fdm - u_pred)
    im1 = axs[1, 0].imshow(err_u, aspect='auto', extent=[-1, 1, 1, 0], cmap='inferno')
    plt.colorbar(im1, ax=axs[1, 0], label='Abs Error')
    axs[1, 0].set_title("Absolute Error Map (Real Part)")

    # Real Ground Truth Heatmap
    im2 = axs[1, 1].imshow(u_fdm, aspect='auto', extent=[-1, 1, 1, 0], cmap='viridis')
    plt.colorbar(im2, ax=axs[1, 1], label='Magnitude')
    axs[1, 1].set_title("Ground Truth FDM (Real Part)")

    plt.tight_layout()
    plt.show()

    print(f"\nFinal Mean Squared Error (u): {np.mean((u_fdm - u_pred)**2):.2e}")
    print(f"Final Mean Squared Error (v): {np.mean((v_fdm - v_pred)**2):.2e}")

if __name__ == "__main__":
    run_experiment()
