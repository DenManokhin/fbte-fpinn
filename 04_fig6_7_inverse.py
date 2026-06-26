import os
import numpy as np
import torch
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import matplotlib.pyplot as plt

from src.physics import PARAMS
from src.fdm_solver import FDMSolver
from src.models import InverseGridData
from src.matrices import torch_riesz_matrix, torch_l1_matrix
from src.utils import select_device, set_reproducible, configure_precision

device = select_device()
configure_precision(device)
set_reproducible(42)

def run_experiment(multi_stage=True):
    print(f"--- Inverse Problem (Multi-stage={multi_stage}) ---")
    
    # 1. Generate Synthetic Patient Scan using FDM as ground truth
    fdm = FDMSolver(PARAMS)
    x_g, t_g, u_fdm, v_fdm = fdm.solve()
    
    XX, TT = np.meshgrid(x_g, t_g)
    grid_np = np.hstack((XX.flatten()[:, None], TT.flatten()[:, None])).astype(dde.config.real(np))
    y_patient = np.hstack((u_fdm.flatten()[:, None], v_fdm.flatten()[:, None])).astype(dde.config.real(np))

    # 2. Setup Inverse Model
    alpha_raw = dde.Variable(0.0)
    beta_raw = dde.Variable(0.0)
    alpha_history = []
    beta_history = []

    def pde_inverse(x, y):
        curr_alpha = torch.sigmoid(alpha_raw)
        curr_beta = 1.0 + torch.sigmoid(beta_raw)

        dt = (PARAMS["T_RANGE"][1] - PARAMS["T_RANGE"][0]) / (PARAMS["N_TIME"] - 1)
        domain_len = PARAMS["X_RANGE"][1] - PARAMS["X_RANGE"][0]

        M_t = torch_l1_matrix(PARAMS["N_TIME"], curr_alpha, dt, device)
        M_s = torch_riesz_matrix(PARAMS["N_SPACE"], curr_beta, domain_len, device)

        u = y[:, 0:1].view(PARAMS["N_TIME"], PARAMS["N_SPACE"])
        v = y[:, 1:2].view(PARAMS["N_TIME"], PARAMS["N_SPACE"])
        x_loc = x[:, 0:1].view(PARAMS["N_TIME"], PARAMS["N_SPACE"])

        u_t = torch.matmul(M_t, u)
        u_xx = torch.matmul(u, M_s.T)
        v_t = torch.matmul(M_t, v)
        v_xx = torch.matmul(v, M_s.T)

        c_real = PARAMS["COUPLING_CONST"] * x_loc * v
        c_imag = -PARAMS["COUPLING_CONST"] * x_loc * u

        res_real = u_t + PARAMS["DIFFUSION"] * u_xx - c_real + PARAMS["RELAX"] * u
        res_imag = v_t + PARAMS["DIFFUSION"] * v_xx - c_imag + PARAMS["RELAX"] * v

        mask = torch.ones_like(u); mask[0, :] = 0
        return torch.cat([(res_real*mask).view(-1,1), (res_imag*mask).view(-1,1)], dim=1)

    data_inv = InverseGridData(grid_np, y_patient, pde_inverse)
    model_inv = dde.Model(data_inv, dde.nn.FNN([2] + [40]*3 + [2], "tanh", "Glorot normal"))

    def record_variables(model):
        a = torch.sigmoid(alpha_raw).item()
        b = 1.0 + torch.sigmoid(beta_raw).item()
        alpha_history.append(a)
        beta_history.append(b)
        print(f"Iter: {model.train_state.step} | Alpha: {a:.4f} | Beta: {b:.4f}")

    class RecordVars(dde.callbacks.Callback):
        def on_epoch_end(self):
            if self.model.train_state.step % 1000 == 0:
                record_variables(self.model)

    if multi_stage:
        print("\n--- Stage 1: Warmup (LR 1e-3, 5k iters) ---")
        model_inv.compile("adam", lr=1e-3, loss_weights=[100, 0], external_trainable_variables=[alpha_raw, beta_raw])
        model_inv.train(iterations=5000, callbacks=[RecordVars()], batch_size=len(grid_np))

        print("\n--- Stage 2: Acceleration (LR 1e-3, 15k iters) ---")
        model_inv.compile("adam", lr=1e-3, loss_weights=[100, 1], external_trainable_variables=[alpha_raw, beta_raw])
        model_inv.train(iterations=15000, callbacks=[RecordVars()], batch_size=len(grid_np))
    else:
        print("\n--- Single Stage Training (20k iters) ---")
        model_inv.compile("adam", lr=1e-3, loss_weights=[100, 1], external_trainable_variables=[alpha_raw, beta_raw])
        model_inv.train(iterations=20000, callbacks=[RecordVars()], batch_size=len(grid_np))

    # Plotting
    print(f"\nFINAL RECOVERY:\nAlpha: {alpha_history[-1]:.4f} (True: {PARAMS['ALPHA']})\nBeta:  {beta_history[-1]:.4f} (True: {PARAMS['BETA']})")

    plt.figure(figsize=(10, 5))
    plt.plot(alpha_history, label='Alpha (Pred)', linewidth=2)
    plt.axhline(PARAMS['ALPHA'], color='k', linestyle='--', label='Alpha (True)')
    plt.plot(beta_history, label='Beta (Pred)', linewidth=2)
    plt.axhline(PARAMS['BETA'], color='r', linestyle='--', label='Beta (True)')
    plt.title(f"Parameter Convergence History (Multi-stage={multi_stage})")
    plt.xlabel("Epochs (x1000)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_experiment(multi_stage=False) # Fig 6
    run_experiment(multi_stage=True)  # Fig 7
