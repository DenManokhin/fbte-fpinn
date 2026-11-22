"""Inverse problem from noisy data (Figs. 12-13): recovery of alpha and beta at SNR = 50, 20, 10, 5.

With no options this runs the paper's setting. The options below change the stage-2 training to
study robustness to noise; any non-default option writes to results/05_inverse_noise_<tag>_seed{N}.npz
(tag from --tag, or built from the options), so the paper's files are never overwritten.

  --w-data W        data weight in both stages (paper: 100)
  --w-pde W         PDE weight in stage 2 (paper: 1)
  --w-pde-start W   ramp the PDE weight geometrically from W to --w-pde over stage 2 (default: off)
  --stage1 N        stage-1 iterations, data only, alpha and beta fixed (paper: 5000)
  --stage2 N        stage-2 iterations, data + PDE (paper: 15000)
  --lbfgs N         after stage 2, up to N L-BFGS iterations with the stage-2 weights (default: 0, off)
  --width N, --depth N   hidden layers of the network (paper: 40, 3)
  --l2 W            weight of an L2 penalty on the network weights (not alpha, beta) (default: 0)
  --ic-bc W         weight of a loss on the known IC u(x,0)=exp(-10x^2), v(x,0)=0 and the zero
                    Dirichlet BC, the same prior knowledge the FDM + least-squares baseline uses (default: 0)
  --val-frac F      hold out a fraction F of the grid points from the data loss and report alpha, beta
                    at the stage-2 step with the lowest held-out data error (early stopping; with --lbfgs
                    the L-BFGS result is only kept in the history) (default: 0)
  --snrs S [S ...]  noise levels to run, "inf" for clean data (default: 50 20 10 5 inf)
  --tag NAME        name of the output files for a non-default setting
"""
import argparse
import json
import os
import numpy as np
import torch
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import matplotlib.pyplot as plt

from src.physics import PARAMS, INVERSE_GRID, fbte_residual, interior_mask, gaussian_ic
from src.fdm_solver import FDMSolver
from src.models import InverseGridData
from src.matrices import torch_riesz_matrix, torch_l1_matrix
from src.utils import add_complex_gaussian_noise, select_device, set_reproducible, configure_precision, ensure_results_dir, parse_seed

PAPER = dict(w_data=100.0, w_pde=1.0, w_pde_start=None, stage1=5000, stage2=15000, lbfgs=0, width=40, depth=3,
             l2=0.0, ic_bc=0.0, val_frac=0.0)
ALL_SNRS = [50, 20, 10, 5, np.inf]


def parse_options():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--w-data", type=float, default=PAPER["w_data"])
    ap.add_argument("--w-pde", type=float, default=PAPER["w_pde"])
    ap.add_argument("--w-pde-start", type=float, default=PAPER["w_pde_start"])
    ap.add_argument("--stage1", type=int, default=PAPER["stage1"])
    ap.add_argument("--stage2", type=int, default=PAPER["stage2"])
    ap.add_argument("--lbfgs", type=int, default=PAPER["lbfgs"])
    ap.add_argument("--width", type=int, default=PAPER["width"])
    ap.add_argument("--depth", type=int, default=PAPER["depth"])
    ap.add_argument("--l2", type=float, default=PAPER["l2"])
    ap.add_argument("--ic-bc", type=float, default=PAPER["ic_bc"])
    ap.add_argument("--val-frac", type=float, default=PAPER["val_frac"])
    ap.add_argument("--snrs", nargs="+", type=float, default=ALL_SNRS)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    cfg = {k: getattr(args, k) for k in PAPER}
    changed = {k: v for k, v in cfg.items() if v != PAPER[k]}
    if args.tag:
        tag = args.tag
    elif changed:
        tag = "_".join(f"{k.replace('_', '')}{v:g}" for k, v in changed.items())
    else:
        tag = ""
    return args, cfg, tag


ARGS, CFG, TAG = parse_options()
SUFFIX = f"_{TAG}" if TAG else ""

device = select_device()
configure_precision(device)
SEED = parse_seed()
set_reproducible(SEED)
ensure_results_dir()

# Inverse problem is solved on the 16x25 grid described in the paper
P = {**PARAMS, **INVERSE_GRID}

def label(snr):
    return f"SNR={snr}" if np.isfinite(snr) else "clean data"


def run_experiment():
    print(f"--- Inverse Problem with Noise (grid {P['N_SPACE']}x{P['N_TIME']}, seed={SEED}) ---")
    changed = {k: v for k, v in CFG.items() if v != PAPER[k]}
    print("Setting:", "paper" if not changed else changed)

    # 1. Generate Synthetic Patient Scan using FDM
    fdm = FDMSolver(P)
    x_g, t_g, u_fdm, v_fdm = fdm.solve()

    XX, TT = np.meshgrid(x_g, t_g)
    grid_np = np.hstack((XX.flatten()[:, None], TT.flatten()[:, None])).astype(dde.config.real(np))

    # The noise for every SNR is drawn in the paper's order (50, 20, 10, 5) before any training.
    # Training does not draw from NumPy's generator, so this is the same noise the paper's runs used,
    # and a run with a subset of --snrs sees the same data as the full run.
    noise = {}
    for snr in ALL_SNRS:
        # Add noise with SNR = max|M| / sigma, eq. (28)
        noise[snr] = add_complex_gaussian_noise(u_fdm, v_fdm, snr) if np.isfinite(snr) else (u_fdm, v_fdm)

    # np.inf is the clean-data reference curve
    snrs = [s for s in ALL_SNRS if any(s == float(r) for r in ARGS.snrs)]
    noisy_snrs = [s for s in snrs if np.isfinite(s)]
    all_alpha_history = {}
    all_beta_history = {}
    final = {}
    noisy_u, noisy_v = {}, {}

    # Points held out of the data loss for early stopping (same split for every SNR)
    val_mask = np.zeros(len(grid_np), dtype=bool)
    if CFG["val_frac"] > 0:
        rng = np.random.RandomState(SEED + 12345)  # own generator, does not touch the noise stream
        val_mask[rng.choice(len(grid_np), int(round(CFG["val_frac"] * len(grid_np))), replace=False)] = True

    for snr in snrs:
        print(f"\n--- Testing with {label(snr)} ---")
        u_noisy, v_noisy = noise[snr]
        noisy_u[snr], noisy_v[snr] = u_noisy, v_noisy
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

            # device and dtype of the inputs, so the same function runs in the float64 L-BFGS stage on the CPU
            M_t = torch_l1_matrix(P["N_TIME"], curr_alpha, dt, x.device, dtype=x.dtype)
            M_s = torch_riesz_matrix(P["N_SPACE"], curr_beta, domain_len, x.device, dtype=x.dtype)

            u = y[:, 0:1].view(P["N_TIME"], P["N_SPACE"])
            v = y[:, 1:2].view(P["N_TIME"], P["N_SPACE"])
            x_loc = x[:, 0:1].view(P["N_TIME"], P["N_SPACE"])

            res_real, res_imag = fbte_residual(u, v, x_loc, M_t, M_s, P)

            mask = interior_mask(u)
            return torch.cat([(res_real*mask).view(-1,1), (res_imag*mask).view(-1,1)], dim=1)

        extra_losses, extra_weights = [], []
        if CFG["ic_bc"] > 0:
            x0 = torch.as_tensor(x_g, dtype=torch.get_default_dtype(), device=device)
            u0 = gaussian_ic(x0)
            u0[0] = u0[-1] = 0  # as in the FDM solver

            def ic_bc_loss(inputs, outputs, model):
                u = outputs[:, 0].view(P["N_TIME"], P["N_SPACE"])
                v = outputs[:, 1].view(P["N_TIME"], P["N_SPACE"])
                ic = torch.mean((u[0] - u0.to(u.device).to(u.dtype)) ** 2) + torch.mean(v[0] ** 2)
                bc = torch.mean(u[:, [0, -1]] ** 2) + torch.mean(v[:, [0, -1]] ** 2)
                return ic + bc
            extra_losses.append(ic_bc_loss)
            extra_weights.append(CFG["ic_bc"])
        if CFG["l2"] > 0:
            def l2_loss(inputs, outputs, model):
                return sum(torch.sum(p ** 2) for n, p in model.net.named_parameters() if "weight" in n)
            extra_losses.append(l2_loss)
            extra_weights.append(CFG["l2"])

        data_inv = InverseGridData(grid_np, y_patient, pde_inverse,
                                   data_mask=~val_mask if val_mask.any() else None, extra_losses=extra_losses)
        net = dde.nn.FNN([2] + [CFG["width"]] * CFG["depth"] + [2], "tanh", "Glorot normal")
        model_inv = dde.Model(data_inv, net)

        def record_variables(model):
            a = torch.sigmoid(alpha_raw).item()
            b = 1.0 + torch.sigmoid(beta_raw).item()
            alpha_history.append(a)
            beta_history.append(b)

        class RecordVars(dde.callbacks.Callback):
            def on_epoch_end(self):
                if self.model.train_state.step % 1000 == 0:
                    record_variables(self.model)

        best = {"val": np.inf, "step": None}
        grid_t = torch.as_tensor(grid_np, device=device)
        y_t = torch.as_tensor(y_patient, device=device)
        val_t = torch.as_tensor(val_mask, device=device)

        class EarlyStopping(dde.callbacks.Callback):
            """Keeps alpha, beta at the step with the lowest data error on the held-out points."""
            def on_epoch_end(self):
                step = self.model.train_state.step
                if step % 100:
                    return
                with torch.no_grad():
                    err = torch.mean((self.model.net(grid_t)[val_t] - y_t[val_t]) ** 2).item()
                if err < best["val"]:
                    best.update(val=err, step=step, alpha=torch.sigmoid(alpha_raw).item(),
                                beta=1.0 + torch.sigmoid(beta_raw).item())

        class PdeRamp(dde.callbacks.Callback):
            """Ramps the PDE weight geometrically from w_pde_start to w_pde over stage 2."""
            def __init__(self, start_step):
                super().__init__()
                self.start_step = start_step

            def on_epoch_begin(self):
                frac = min(1.0, (self.model.train_state.step - self.start_step) / max(1, CFG["stage2"]))
                w = CFG["w_pde_start"] * (CFG["w_pde"] / CFG["w_pde_start"]) ** frac
                self.model.loss_weights[1] = w

        # Multi-stage training
        model_inv.compile("adam", lr=1e-3, loss_weights=[CFG["w_data"], 0] + extra_weights,
                          external_trainable_variables=[alpha_raw, beta_raw])
        model_inv.train(iterations=CFG["stage1"], callbacks=[RecordVars()], batch_size=len(grid_np))

        stage2_callbacks = [RecordVars()]
        if val_mask.any():
            stage2_callbacks.append(EarlyStopping())
        if CFG["w_pde_start"] is not None:
            stage2_callbacks.append(PdeRamp(model_inv.train_state.step))
        w_pde0 = CFG["w_pde"] if CFG["w_pde_start"] is None else CFG["w_pde_start"]
        model_inv.compile("adam", lr=1e-3, loss_weights=[CFG["w_data"], w_pde0] + extra_weights,
                          external_trainable_variables=[alpha_raw, beta_raw])
        model_inv.train(iterations=CFG["stage2"], callbacks=stage2_callbacks, batch_size=len(grid_np))

        if CFG["lbfgs"] > 0:
            # torch L-BFGS with a strong-Wolfe line search. DeepXDE's L-BFGS has no line search and in
            # float32 often stops after a few iterations, so the network and alpha, beta are optimised here.
            # It runs on the CPU in float64: in float32 (MPS) the gradient near the end of Adam is too
            # inaccurate for the line search, which then finds no decrease and stops after a few iterations.
            # If it still stops early, it is restarted with an empty history, until --lbfgs iterations
            # or 3 short attempts in a row.
            cpu64 = dict(device="cpu", dtype=torch.float64)
            model_inv.net.to("cpu").to(torch.float64)  # in two steps: MPS tensors cannot be cast to float64
            for var in (alpha_raw, beta_raw):
                var.data = var.data.to("cpu").to(torch.float64)
            grid64 = torch.as_tensor(grid_np, **cpu64)
            y64 = torch.as_tensor(y_patient, **cpu64)
            weights = torch.as_tensor([CFG["w_data"], CFG["w_pde"]] + extra_weights, **cpu64)
            params = list(model_inv.net.parameters()) + [alpha_raw, beta_raw]

            def weighted_losses():
                return torch.stack(data_inv.losses(y64, model_inv.net(grid64), None, grid64, model_inv)) * weights

            total, attempts, short, n_bad = 0, 0, 0, [0]
            while total < CFG["lbfgs"] and short < 3:
                opt = torch.optim.LBFGS(params, lr=1.0, max_iter=CFG["lbfgs"] - total, history_size=50,
                                        line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=0.0)

                def closure():
                    opt.zero_grad()
                    loss = torch.sum(weighted_losses())
                    loss.backward()
                    if not torch.isfinite(loss):
                        # A trial step of the line search can reach alpha = 1 or beta = 2 in float32, where
                        # the loss or its gradient is not finite; a large loss makes the line search step back.
                        n_bad[0] += 1
                        for p in params:
                            p.grad.zero_()
                        return torch.tensor(1e10, **cpu64)
                    for p in params:
                        if not torch.isfinite(p.grad).all():
                            n_bad[0] += 1
                            p.grad.nan_to_num_(0.0, 0.0, 0.0)
                    return loss

                opt.step(closure)
                n = opt.state_dict()["state"][0]["n_iter"]
                total += n
                attempts += 1
                short = short + 1 if n < 10 else 0
            with torch.no_grad():
                losses = weighted_losses()
            print(f"L-BFGS: {total} iterations in {attempts} attempts, {n_bad[0]} non-finite evaluations, "
                  f"weighted losses {[f'{v:.3e}' for v in losses.tolist()]}")
            record_variables(model_inv)

        if best["step"] is not None:
            final[snr] = (best["alpha"], best["beta"])
            print(f"Early stopping: lowest held-out error {best['val']:.3e} at step {best['step']}")
        else:
            final[snr] = (alpha_history[-1], beta_history[-1])

        all_alpha_history[snr] = alpha_history
        all_beta_history[snr] = beta_history
        print(f"Final Alpha: {final[snr][0]:.4f}, Final Beta: {final[snr][1]:.4f}")

    np.savez(f"results/05_inverse_noise{SUFFIX}_seed{SEED}.npz",
             snrs=np.array(snrs),
             alpha=np.array([all_alpha_history[s] for s in snrs]),
             beta=np.array([all_beta_history[s] for s in snrs]),
             alpha_true=P["ALPHA"], beta_true=P["BETA"],
             # the reported estimates: the last values, or the early-stopping values with --val-frac
             alpha_final=np.array([final[s][0] for s in snrs]),
             beta_final=np.array([final[s][1] for s in snrs]),
             config=json.dumps(CFG),
             # the exact data each run was trained on, so 11_inverse_fdm_baseline.py can fit the same data
             u_data=np.array([noisy_u[s] for s in snrs]),
             v_data=np.array([noisy_v[s] for s in snrs]))

    # Example of the noisy data (Fig. 12): signal magnitude |M| over time at the node closest to x = 0
    ix = int(np.argmin(np.abs(x_g)))
    plt.figure(figsize=(10, 5))
    plt.plot(t_g, np.hypot(u_fdm[:, ix], v_fdm[:, ix]), 'k-', linewidth=2, label='Clean Magnitude')
    for snr in noisy_snrs:
        plt.plot(t_g, np.hypot(noisy_u[snr][:, ix], noisy_v[snr][:, ix]), '--', label=f'Noisy Magnitude (SNR={snr})')
    plt.title(f"Simulated MRI Signal Magnitude at x={x_g[ix]:.2f}")
    plt.xlabel("Time (t)")
    plt.ylabel("Transverse Magnetization (|M|)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/05_fig12_noisy_data{SUFFIX}_seed{SEED}.png", dpi=300, bbox_inches="tight")

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for snr in snrs:
        axs[0].plot(all_alpha_history[snr], label=f'Alpha ({label(snr)})')
        axs[1].plot(all_beta_history[snr], label=f'Beta ({label(snr)})')

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
    plt.savefig(f"results/05_fig8_9_inverse_noise{SUFFIX}_seed{SEED}.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    run_experiment()
