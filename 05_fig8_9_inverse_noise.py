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
from src.utils import add_gaussian_noise, select_device, set_reproducible, configure_precision

device = select_device()
configure_precision(device)
set_reproducible(42)

def run_experiment():
    print(f"--- Inverse Problem with Noise ---")
    
    # 1. Generate Synthetic Patient Scan using FDM
    fdm = FDMSolver(PARAMS)
    x_g, t_g, u_fdm, v_fdm = fdm.solve()
    
    XX, TT = np.meshgrid(x_g, t_g)
    grid_np = np.hstack((XX.flatten()[:, None], TT.flatten()[:, None])).astype(dde.config.real(np))
    
    snrs = [50, 20, 10, 5]
    all_alpha_history = {}
    all_beta_history = {}

    for snr in snrs:
        print(f"\n--- Testing with SNR = {snr} ---")
        
        # Add noise
        u_noisy = add_gaussian_noise(u_fdm, snr)
        v_noisy = add_gaussian_noise(v_fdm, snr)
        y_patient = np.hstack((u_noisy.flatten()[:, None], v_noisy.flatten()[:, None])).astype(dde.config.real(np))

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

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for snr in snrs:
        axs[0].plot(all_alpha_history[snr], label=f'Alpha (SNR={snr})')
        axs[1].plot(all_beta_history[snr], label=f'Beta (SNR={snr})')

    axs[0].axhline(PARAMS['ALPHA'], color='k', linestyle='--', label='True Alpha')
    axs[0].set_title("Alpha Convergence under Noise")
    axs[0].set_xlabel("Epochs (x1000)")
    axs[0].set_ylabel("Value")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].axhline(PARAMS['BETA'], color='r', linestyle='--', label='True Beta')
    axs[1].set_title("Beta Convergence under Noise")
    axs[1].set_xlabel("Epochs (x1000)")
    axs[1].set_ylabel("Value")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()
