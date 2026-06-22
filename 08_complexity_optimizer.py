import os
import time
import torch
import numpy as np
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import matplotlib.pyplot as plt

from src.physics import PARAMS
from src.models import ForwardGridData
from src.utils import select_device, set_reproducible

device = select_device()

class TimerCallback(dde.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.times = []
        self.losses = []
        self.start_time = None

    def on_train_begin(self):
        self.start_time = time.time()

    def on_epoch_end(self):
        # We might not want to append every epoch if it's too frequent,
        # but for a few thousand iterations it's fine.
        self.times.append(time.time() - self.start_time)
        # self.model.train_state.loss_train is a list of losses for each term
        # we sum them to get the total training loss
        total_loss = sum(self.model.train_state.loss_train)
        self.losses.append(total_loss)

def create_model():
    run_params = PARAMS.copy()
    x_vals = np.linspace(run_params["X_RANGE"][0], run_params["X_RANGE"][1], run_params["N_SPACE"])
    t_vals = np.linspace(run_params["T_RANGE"][0], run_params["T_RANGE"][1], run_params["N_TIME"])
    X, T = np.meshgrid(x_vals, t_vals)
    grid_np = np.hstack((X.flatten()[:, None], T.flatten()[:, None])).astype(np.float32)
    x_grid = torch.tensor(grid_np, dtype=torch.float32, device=device)

    data = ForwardGridData(x_grid, run_params, device)
    net = dde.nn.FNN([2] + [64]*4 + [2], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    return model

def run_optimizer_experiment():
    print(f"--- Starting Optimizer Complexity Experiment on {device} ---")
    
    adam_iters = 3000
    lbfgs_iters = 3000
    
    # 1. Adam Only
    print("\n--- Training with Adam only ---")
    set_reproducible(42)
    model_adam = create_model()
    model_adam.compile("adam", lr=1e-3)
    
    timer_adam = TimerCallback()
    model_adam.train(iterations=adam_iters, display_every=1000, callbacks=[timer_adam])
    
    # 2. Adam + L-BFGS
    print("\n--- Training with Adam + L-BFGS ---")
    set_reproducible(42)
    model_hybrid = create_model()
    model_hybrid.compile("adam", lr=1e-3)
    
    timer_hybrid = TimerCallback()
    
    # Train with Adam first
    model_hybrid.train(iterations=adam_iters // 2, display_every=1000, callbacks=[timer_hybrid])
    
    # Switch to L-BFGS
    dde.optimizers.config.set_LBFGS_options(maxiter=lbfgs_iters)
    model_hybrid.compile("L-BFGS")
    
    # Note: L-BFGS in DeepXDE often evaluates the loss many times per epoch/step.
    # The callback will capture the epochs as DeepXDE defines them.
    # Adjust start time so the timer continues seamlessly
    timer_hybrid.start_time = time.time() - timer_hybrid.times[-1] if len(timer_hybrid.times) > 0 else time.time()
    model_hybrid.train(display_every=1000, callbacks=[timer_hybrid])
    
    # Plotting Results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_adam = np.arange(1, len(timer_adam.losses) + 1)
    epochs_hybrid = np.arange(1, len(timer_hybrid.losses) + 1)
    
    # Loss vs Epochs
    ax = axes[0]
    ax.plot(epochs_adam, timer_adam.losses, label='Adam', color='blue')
    ax.plot(epochs_hybrid, timer_hybrid.losses, label='Adam + L-BFGS', color='red')
    ax.set_yscale('log')
    ax.set_xlabel('Epochs / Optimizer Steps')
    ax.set_ylabel('Total Training Loss')
    ax.set_title('Convergence: Loss vs Epochs')
    ax.legend()
    ax.grid(True)
    
    # Loss vs Wall-clock Time
    ax = axes[1]
    ax.plot(timer_adam.times, timer_adam.losses, label='Adam', color='blue')
    ax.plot(timer_hybrid.times, timer_hybrid.losses, label='Adam + L-BFGS', color='red')
    ax.set_yscale('log')
    ax.set_xlabel('Wall-clock Time (seconds)')
    ax.set_ylabel('Total Training Loss')
    ax.set_title('Time-to-Solution: Loss vs Time')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('complexity_optimizer.png')
    print("Saved plot to complexity_optimizer.png")
    plt.show()

if __name__ == "__main__":
    run_optimizer_experiment()
