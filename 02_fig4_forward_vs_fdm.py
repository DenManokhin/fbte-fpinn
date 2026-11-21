import os
import numpy as np
import torch
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import matplotlib.pyplot as plt

from src.physics import PARAMS
from src.fdm_solver import FDMSolver
from src.models import ForwardGridData
from src.utils import select_device, set_reproducible, configure_precision, calculate_metrics, save_metrics_to_csv, ensure_results_dir

device = select_device()
configure_precision(device)
set_reproducible(42)
ensure_results_dir()

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
    net = dde.nn.FNN([2] + [64]*4 + [2], "tanh", "Glorot normal")
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
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0, 0].imshow(u_fdm, aspect='auto', extent=[-1, 1, 1, 0], cmap='jet')
    axs[0, 0].set_title("FDM Real (u)")

    axs[0, 1].imshow(u_pred, aspect='auto', extent=[-1, 1, 1, 0], cmap='jet')
    axs[0, 1].set_title("Predicted Real (u)")

    u_err = np.abs(u_fdm - u_pred)
    im = axs[0, 2].imshow(u_err, aspect='auto', extent=[-1, 1, 1, 0], cmap='inferno')
    plt.colorbar(im, ax=axs[0, 2])
    axs[0, 2].set_title(f"Absolute Error (Max: {np.max(u_err):.4f})")

    axs[1, 0].imshow(v_fdm, aspect='auto', extent=[-1, 1, 1, 0], cmap='jet')
    axs[1, 0].set_title("FDM Imaginary (v)")

    axs[1, 1].imshow(v_pred, aspect='auto', extent=[-1, 1, 1, 0], cmap='jet')
    axs[1, 1].set_title("Predicted Imaginary (v)")

    v_err = np.abs(v_fdm - v_pred)
    im = axs[1, 2].imshow(v_err, aspect='auto', extent=[-1, 1, 1, 0], cmap='inferno')
    plt.colorbar(im, ax=axs[1, 2])
    axs[1, 2].set_title(f"Absolute Error (Max: {np.max(v_err):.4f})")

    plt.tight_layout()
    plt.savefig("results/02_fig4_forward_vs_fdm.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n--- Evaluation Metrics (vs FDM) ---")
    u_metrics = calculate_metrics(u_fdm, u_pred)
    print("Real Component (u) Metrics:")
    for name, val in u_metrics.items():
        print(f"  {name}: {val:.4e}")

    v_metrics = calculate_metrics(v_fdm, v_pred)
    print("Imaginary Component (v) Metrics:")
    for name, val in v_metrics.items():
        print(f"  {name}: {val:.4e}")
    print("-----------------------------------\n")

    save_metrics_to_csv(u_metrics, v_metrics, "results/metrics_02_forward_vs_fdm.csv")
    print("Metrics saved to results/metrics_02_forward_vs_fdm.csv")

if __name__ == "__main__":
    run_experiment()
