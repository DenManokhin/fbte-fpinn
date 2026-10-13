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
from src.utils import add_complex_gaussian_noise, select_device, set_reproducible, configure_precision, ensure_results_dir, parse_seed

device = select_device()
configure_precision(device)
SEED = parse_seed()
set_reproducible(SEED)
ensure_results_dir()

# Inverse problem is solved on the 16x25 grid described in the paper
P = {**PARAMS, **INVERSE_GRID}

def run_experiment():
    print(f"--- Inverse Problem with Noise (grid {P['N_SPACE']}x{P['N_TIME']}, seed={SEED}) ---")

    # 1. Generate Synthetic Patient Scan using FDM
    fdm = FDMSolver(P)
    x_g, t_g, u_fdm, v_fdm = fdm.solve()

    XX, TT = np.meshgrid(x_g, t_g)
    grid_np = np.hstack((XX.flatten()[:, None], TT.flatten()[:, None])).astype(dde.config.real(np))

    snrs = [50, 20, 10, 5]
    all_alpha_history = {}
    all_beta_history = {}

    for snr in snrs:
        print(f"\n--- Testing with SNR = {snr} ---")

        # Add noise with SNR = max|M| / sigma, eq. (28)
        u_noisy, v_noisy = add_complex_gaussian_noise(u_fdm, v_fdm, snr)
        y_patient = np.hstack((u_noisy.flatten()[:, None], v_noisy.flatten()[:, None])).astype(dde.config.real(np))

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

        class RecordVars(dde.callbacks.Callback):
            def on_epoch_end(self):
                if self.model.train_state.step % 1000 == 0:
                    record_variables(self.model)

        # Multi-stage training
        model_inv.compile("adam", lr=1e-3, loss_weights=[100, 0], external_trainable_variables=[alpha_raw, beta_raw])
        model_inv.train(iterations=5000, callbacks=[RecordVars()], batch_size=len(grid_np))

        model_inv.compile("adam", lr=1e-3, loss_weights=[100, 1], external_trainable_variables=[alpha_raw, beta_raw])
        model_inv.train(iterations=15000, callbacks=[RecordVars()], batch_size=len(grid_np))

        all_alpha_history[snr] = alpha_history
        all_beta_history[snr] = beta_history
        print(f"Final Alpha: {alpha_history[-1]:.4f}, Final Beta: {beta_history[-1]:.4f}")

    np.savez(f"results/05_inverse_noise_seed{SEED}.npz",
             snrs=np.array(snrs),
             alpha=np.array([all_alpha_history[s] for s in snrs]),
             beta=np.array([all_beta_history[s] for s in snrs]),
             alpha_true=P["ALPHA"], beta_true=P["BETA"])

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for snr in snrs:
        axs[0].plot(all_alpha_history[snr], label=f'Alpha (SNR={snr})')
        axs[1].plot(all_beta_history[snr], label=f'Beta (SNR={snr})')

    axs[0].axhline(P['ALPHA'], color='k', linestyle='--', label='True Alpha')
    axs[0].set_title("Alpha Convergence under Noise")
    axs[0].set_xlabel("Epochs (x1000)")
    axs[0].set_ylabel("Value")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].axhline(P['BETA'], color='r', linestyle='--', label='True Beta')
    axs[1].set_title("Beta Convergence under Noise")
    axs[1].set_xlabel("Epochs (x1000)")
    axs[1].set_ylabel("Value")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"results/05_fig8_9_inverse_noise_seed{SEED}.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    run_experiment()
