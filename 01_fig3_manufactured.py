import os
import math
import numpy as np
import torch
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import matplotlib.pyplot as plt
from src.physics import PARAMS, fbte_residual, interior_mask
from src.matrices import torch_riesz_matrix, torch_l1_matrix
from src.models import MMSData
from src.utils import select_device, set_reproducible, calculate_metrics, save_metrics_to_csv, configure_precision, ensure_results_dir

device = select_device()
configure_precision(device)
set_reproducible(42)
ensure_results_dir()

def run_experiment():
    print("--- Starting MMS Verification ---")
    
    # Grid Setup
    x_vals = np.linspace(PARAMS["X_RANGE"][0], PARAMS["X_RANGE"][1], PARAMS["N_SPACE"])
    t_vals = np.linspace(PARAMS["T_RANGE"][0], PARAMS["T_RANGE"][1], PARAMS["N_TIME"])
    X, T = np.meshgrid(x_vals, t_vals)
    grid_np = np.hstack((X.flatten()[:, None], T.flatten()[:, None])).astype(dde.config.real(np))

    x_grid = torch.tensor(X, dtype=torch.get_default_dtype(), device=device)
    t_grid = torch.tensor(T, dtype=torch.get_default_dtype(), device=device)

    # 1. Exact Functions
    u_exact = (t_grid ** 2) * torch.sin(math.pi * x_grid)
    v_exact = (t_grid ** 3) * torch.sin(math.pi * x_grid)

    # 2. Discrete Derivatives
    dt = (PARAMS["T_RANGE"][1] - PARAMS["T_RANGE"][0]) / (PARAMS["N_TIME"] - 1)
    domain_len = PARAMS["X_RANGE"][1] - PARAMS["X_RANGE"][0]
    
    M_space = torch_riesz_matrix(PARAMS["N_SPACE"], PARAMS["BETA"], domain_len, device)
    M_time = torch_l1_matrix(PARAMS["N_TIME"], PARAMS["ALPHA"], dt, device)

    # 3. Assemble Forcing Terms: the residual of the exact solution under the same
    # discrete FBTE operator that is used in all other experiments
    f_real, f_imag = fbte_residual(u_exact, v_exact, x_grid, M_time, M_space, PARAMS)

    def pde_loss_fn(x, y):
        u = y[:, 0:1].view(PARAMS["N_TIME"], PARAMS["N_SPACE"])
        v = y[:, 1:2].view(PARAMS["N_TIME"], PARAMS["N_SPACE"])
        x_loc = x[:, 0:1].view(PARAMS["N_TIME"], PARAMS["N_SPACE"])

        res_u, res_v = fbte_residual(u, v, x_loc, M_time, M_space, PARAMS, f_real, f_imag)

        mask = interior_mask(u)
        return torch.cat([(res_u * mask).view(-1, 1), (res_v * mask).view(-1, 1)], dim=1)

    data = MMSData(grid_np, pde_loss_fn, u_exact, v_exact, PARAMS)

    net = dde.nn.FNN([2] + [64]*4 + [2], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)

    model.train(iterations=15000, display_every=1000)

    # Predict
    y_pred = model.predict(grid_np)
    u_pred = y_pred[:, 0].reshape(PARAMS["N_TIME"], PARAMS["N_SPACE"])
    v_pred = y_pred[:, 1].reshape(PARAMS["N_TIME"], PARAMS["N_SPACE"])

    # Plotting
    u_true_np = u_exact.cpu().numpy()
    v_true_np = v_exact.cpu().numpy()

    print("\n--- Evaluation Metrics ---")
    u_metrics = calculate_metrics(u_true_np, u_pred)
    print("Real Component (u) Metrics:")
    for name, val in u_metrics.items():
        print(f"  {name}: {val:.4e}")

    v_metrics = calculate_metrics(v_true_np, v_pred)
    print("Imaginary Component (v) Metrics:")
    for name, val in v_metrics.items():
        print(f"  {name}: {val:.4e}")
    print("--------------------------\n")

    save_metrics_to_csv(u_metrics, v_metrics, "results/metrics_01_manufactured.csv")
    print("Metrics saved to results/metrics_01_manufactured.csv")

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0, 0].imshow(u_true_np, aspect='auto', extent=[-1, 1, 1, 0], cmap='jet')
    axs[0, 0].set_title("Exact Real (u)")

    axs[0, 1].imshow(u_pred, aspect='auto', extent=[-1, 1, 1, 0], cmap='jet')
    axs[0, 1].set_title("Predicted Real (u)")

    u_err = np.abs(u_true_np - u_pred)
    im = axs[0, 2].imshow(u_err, aspect='auto', extent=[-1, 1, 1, 0], cmap='inferno')
    plt.colorbar(im, ax=axs[0, 2])
    axs[0, 2].set_title(f"Absolute Error (Max: {np.max(u_err):.4f})")

    axs[1, 0].imshow(v_true_np, aspect='auto', extent=[-1, 1, 1, 0], cmap='jet')
    axs[1, 0].set_title("Exact Imaginary (v)")

    axs[1, 1].imshow(v_pred, aspect='auto', extent=[-1, 1, 1, 0], cmap='jet')
    axs[1, 1].set_title("Predicted Imaginary (v)")

    v_err = np.abs(v_true_np - v_pred)
    im = axs[1, 2].imshow(v_err, aspect='auto', extent=[-1, 1, 1, 0], cmap='inferno')
    plt.colorbar(im, ax=axs[1, 2])
    axs[1, 2].set_title(f"Absolute Error (Max: {np.max(v_err):.4f})")

    plt.tight_layout()
    plt.savefig("results/01_fig3_manufactured.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    run_experiment()
