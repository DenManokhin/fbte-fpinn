import os
import numpy as np
import torch
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import matplotlib.pyplot as plt

from src.physics import PARAMS, INVERSE_GRID, fbte_residual, interior_mask
from src.fdm_solver import FDMSolver
from src.models import InverseGridData
from src.matrices import torch_riesz_matrix, torch_l1_matrix
from src.utils import select_device, set_reproducible, configure_precision, ensure_results_dir, parse_seed

device = select_device()
configure_precision(device)
SEED = parse_seed()
set_reproducible(SEED)
ensure_results_dir()

# Inverse problem is solved on the 16x25 grid described in the paper
P = {**PARAMS, **INVERSE_GRID}

def run_experiment(multi_stage=True):
    print(f"--- Inverse Problem (Multi-stage={multi_stage}, grid {P['N_SPACE']}x{P['N_TIME']}, seed={SEED}) ---")

    # 1. Generate Synthetic Patient Scan using FDM as ground truth
    fdm = FDMSolver(P)
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

        dt = (P["T_RANGE"][1] - P["T_RANGE"][0]) / (P["N_TIME"] - 1)
        domain_len = P["X_RANGE"][1] - P["X_RANGE"][0]

        M_t = torch_l1_matrix(P["N_TIME"], curr_alpha, dt, device)
        M_s = torch_riesz_matrix(P["N_SPACE"], curr_beta, domain_len, device)

        u = y[:, 0:1].view(P["N_TIME"], P["N_SPACE"])
        v = y[:, 1:2].view(P["N_TIME"], P["N_SPACE"])
        x_loc = x[:, 0:1].view(P["N_TIME"], P["N_SPACE"])

        res_real, res_imag = fbte_residual(u, v, x_loc, M_t, M_s, P)

        mask = interior_mask(u)
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

    stage = "multi" if multi_stage else "single"
    np.savez(f"results/04_inverse_{stage}_seed{SEED}.npz",
             alpha=np.array(alpha_history), beta=np.array(beta_history),
             alpha_true=P["ALPHA"], beta_true=P["BETA"])

    # Plotting
    print(f"\nFINAL RECOVERY:\nAlpha: {alpha_history[-1]:.4f} (True: {P['ALPHA']})\nBeta:  {beta_history[-1]:.4f} (True: {P['BETA']})")

    plt.figure(figsize=(10, 5))
    plt.plot(alpha_history, label='Alpha (Pred)', linewidth=2)
    plt.axhline(P['ALPHA'], color='k', linestyle='--', label='Alpha (True)')
    plt.plot(beta_history, label='Beta (Pred)', linewidth=2)
    plt.axhline(P['BETA'], color='r', linestyle='--', label='Beta (True)')
    plt.title(f"Parameter Convergence History (Multi-stage={multi_stage})")
    plt.xlabel("Epochs (x1000)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/04_inverse_{stage}_seed{SEED}.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    run_experiment(multi_stage=False) # Single-stage training
    run_experiment(multi_stage=True)  # Multi-stage training
