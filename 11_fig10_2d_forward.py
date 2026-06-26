import os
import math
import numpy as np
import torch
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import matplotlib.pyplot as plt
from src.physics import PARAMS
from src.matrices import torch_riesz_matrix, torch_l1_matrix
from src.models import MMSData2D
from src.utils import select_device, set_reproducible

device = select_device()
set_reproducible(42)

def run_experiment():
    print("--- Starting 2D MMS Verification ---")
    
    # Grid Setup
    x_vals = np.linspace(PARAMS["X_RANGE"][0], PARAMS["X_RANGE"][1], PARAMS["N_SPACE_X"])
    y_vals = np.linspace(PARAMS["Y_RANGE"][0], PARAMS["Y_RANGE"][1], PARAMS["N_SPACE_Y"])
    t_vals = np.linspace(PARAMS["T_RANGE"][0], PARAMS["T_RANGE"][1], PARAMS["N_TIME_2D"])
    
    # Create 3D meshgrid: indexing='ij' gives shape (N_TIME_2D, N_SPACE_X, N_SPACE_Y)
    T, X, Y = np.meshgrid(t_vals, x_vals, y_vals, indexing='ij')
    
    grid_np = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1), T.reshape(-1, 1))).astype(np.float32)

    x_grid = torch.tensor(X, dtype=torch.float32, device=device)
    y_grid = torch.tensor(Y, dtype=torch.float32, device=device)
    t_grid = torch.tensor(T, dtype=torch.float32, device=device)

    # 1. Exact Functions (2D Spatial)
    u_exact = (t_grid ** 2) * torch.sin(math.pi * x_grid) * torch.sin(math.pi * y_grid)
    v_exact = (t_grid ** 3) * torch.sin(math.pi * x_grid) * torch.sin(math.pi * y_grid)

    # 2. Discrete Derivatives
    dt = (PARAMS["T_RANGE"][1] - PARAMS["T_RANGE"][0]) / (PARAMS["N_TIME_2D"] - 1)
    len_x = PARAMS["X_RANGE"][1] - PARAMS["X_RANGE"][0]
    len_y = PARAMS["Y_RANGE"][1] - PARAMS["Y_RANGE"][0]
    
    M_time = torch_l1_matrix(PARAMS["N_TIME_2D"], PARAMS["ALPHA"], dt, device)
    M_space_x = torch_riesz_matrix(PARAMS["N_SPACE_X"], PARAMS["BETA"], len_x, device)
    M_space_y = torch_riesz_matrix(PARAMS["N_SPACE_Y"], PARAMS["BETA"], len_y, device)

    # Time derivatives
    u_t_exact = torch.einsum('ij, jkl -> ikl', M_time, u_exact)
    v_t_exact = torch.einsum('ij, jkl -> ikl', M_time, v_exact)

    # Spatial derivatives
    u_xx_exact = torch.einsum('ij, kjl -> kil', M_space_x, u_exact)
    u_yy_exact = torch.einsum('ij, klj -> kli', M_space_y, u_exact)
    
    v_xx_exact = torch.einsum('ij, kjl -> kil', M_space_x, v_exact)
    v_yy_exact = torch.einsum('ij, klj -> kli', M_space_y, v_exact)

    # 3. Assemble Forcing Terms
    c_real = PARAMS["COUPLING"] * (x_grid + y_grid) * v_exact
    c_imag = -PARAMS["COUPLING"] * (x_grid + y_grid) * u_exact

    f_real = u_t_exact - PARAMS["DIFFUSION"] * (u_xx_exact + u_yy_exact) + c_real + PARAMS["RELAX"] * u_exact
    f_imag = v_t_exact - PARAMS["DIFFUSION"] * (v_xx_exact + v_yy_exact) + c_imag + PARAMS["RELAX"] * v_exact

    def pde_loss_fn(inputs, outputs):
        u = outputs[:, 0:1].view(PARAMS["N_TIME_2D"], PARAMS["N_SPACE_X"], PARAMS["N_SPACE_Y"])
        v = outputs[:, 1:2].view(PARAMS["N_TIME_2D"], PARAMS["N_SPACE_X"], PARAMS["N_SPACE_Y"])
        x_loc = inputs[:, 0:1].view(PARAMS["N_TIME_2D"], PARAMS["N_SPACE_X"], PARAMS["N_SPACE_Y"])
        y_loc = inputs[:, 1:2].view(PARAMS["N_TIME_2D"], PARAMS["N_SPACE_X"], PARAMS["N_SPACE_Y"])

        u_t = torch.einsum('ij, jkl -> ikl', M_time, u)
        v_t = torch.einsum('ij, jkl -> ikl', M_time, v)
        
        u_xx = torch.einsum('ij, kjl -> kil', M_space_x, u)
        u_yy = torch.einsum('ij, klj -> kli', M_space_y, u)
        v_xx = torch.einsum('ij, kjl -> kil', M_space_x, v)
        v_yy = torch.einsum('ij, klj -> kli', M_space_y, v)

        c_real_net = PARAMS["COUPLING"] * (x_loc + y_loc) * v
        c_imag_net = -PARAMS["COUPLING"] * (x_loc + y_loc) * u

        res_u = u_t - PARAMS["DIFFUSION"]*(u_xx + u_yy) + c_real_net + PARAMS["RELAX"]*u - f_real
        res_v = v_t - PARAMS["DIFFUSION"]*(v_xx + v_yy) + c_imag_net + PARAMS["RELAX"]*v - f_imag

        mask = torch.ones_like(u); mask[0, :, :] = 0
        return torch.cat([(res_u * mask).view(-1, 1), (res_v * mask).view(-1, 1)], dim=1)

    data = MMSData2D(grid_np, pde_loss_fn, u_exact, v_exact, PARAMS)

    net = dde.nn.FNN([3] + [64]*4 + [2], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)

    model.train(iterations=10000, display_every=1000)

    # Predict
    y_pred = model.predict(grid_np)
    u_pred = y_pred[:, 0].reshape(PARAMS["N_TIME_2D"], PARAMS["N_SPACE_X"], PARAMS["N_SPACE_Y"])
    v_pred = y_pred[:, 1].reshape(PARAMS["N_TIME_2D"], PARAMS["N_SPACE_X"], PARAMS["N_SPACE_Y"])

    # Error calculation
    u_true_np = u_exact.cpu().numpy()
    v_true_np = v_exact.cpu().numpy()
    
    l2_err_u = np.linalg.norm(u_pred - u_true_np) / np.linalg.norm(u_true_np)
    l2_err_v = np.linalg.norm(v_pred - v_true_np) / np.linalg.norm(v_true_np)
    print(f"Final Relative L2 Error (u): {l2_err_u:.4e}")
    print(f"Final Relative L2 Error (v): {l2_err_v:.4e}")

    # Plotting
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Exact u at t={t_vals[-1]:.2f}")
    plt.contourf(x_vals, y_vals, u_true_np[-1, :, :].T, levels=50, cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title(f"Predicted u at t={t_vals[-1]:.2f}")
    plt.contourf(x_vals, y_vals, u_pred[-1, :, :].T, levels=50, cmap="viridis")
    plt.colorbar()

    plt.savefig("fig10_2d_forward.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig10_2d_forward.png")

if __name__ == "__main__":
    run_experiment()
