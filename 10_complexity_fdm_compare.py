import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde

from src.physics import PARAMS
from src.fdm_solver import FDMSolver
from src.models import ForwardGridData
from src.utils import select_device, set_reproducible, configure_precision

dde.config.set_default_float("float32")
device = select_device()
configure_precision(device)

def main():
    set_reproducible(42)

    print(f"Running on device: {device}")
    
    # We will benchmark symmetric grids where N_SPACE = N_TIME
    grid_sizes = [32, 64, 128, 256]
    
    fdm_times = []
    pinn_train_times = []
    pinn_infer_times = []
    
    for N in grid_sizes:
        print(f"\n--- Benchmarking Grid Size N_x = N_t = {N} ---")
        PARAMS["N_SPACE"] = N
        PARAMS["N_TIME"] = N
        
        # ---------------------------------------------------------
        # 1. FDM Benchmark (Time-to-Solution from scratch)
        # ---------------------------------------------------------
        fdm = FDMSolver(PARAMS)
        start_t = time.time()
        fdm.solve()
        fdm_time = time.time() - start_t
        fdm_times.append(fdm_time)
        print(f"FDM Solver Time: {fdm_time:.4f}s")
        
        # ---------------------------------------------------------
        # 2. fPINN Benchmark
        # ---------------------------------------------------------
        # Create 1D Grid
        x_grid = np.linspace(PARAMS["X_RANGE"][0], PARAMS["X_RANGE"][1], PARAMS["N_SPACE"], dtype=dde.config.real(np))
        t_grid = np.linspace(PARAMS["T_RANGE"][0], PARAMS["T_RANGE"][1], PARAMS["N_TIME"], dtype=dde.config.real(np))
        X, T = np.meshgrid(x_grid, t_grid)
        X_flat = np.vstack((X.flatten(), T.flatten())).T
        
        data = ForwardGridData(X_flat, PARAMS, device)
        net = dde.nn.FNN([2] + [64] * 4 + [2], "tanh", "Glorot normal")
        model = dde.Model(data, net)
        model.compile("adam", lr=1e-3)
        
        # A. Training Time (Time-to-Solution overhead)
        # We run 500 epochs just to measure the time per epoch scaling accurately
        start_t = time.time()
        model.train(iterations=500, display_every=1000)
        train_time = time.time() - start_t
        pinn_train_times.append(train_time)
        print(f"fPINN Training Time (500 epochs): {train_time:.4f}s")
        
        # B. Inference Time (Surrogate evaluation)
        # We measure inference multiple times to get a stable average, 
        # as a single forward pass on GPU is extremely fast.
        num_inference_runs = 10
        start_t = time.time()
        for _ in range(num_inference_runs):
            _ = model.predict(X_flat)
        infer_time = (time.time() - start_t) / float(num_inference_runs)
        pinn_infer_times.append(infer_time)
        print(f"fPINN Inference Time (avg): {infer_time:.4f}s")
        
    # ---------------------------------------------------------
    # Plotting the Results
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(grid_sizes, fdm_times, 'o-', label="FDM Time (from scratch)", linewidth=2)
    plt.plot(grid_sizes, pinn_train_times, 's-', label="fPINN Training (500 epochs)", linewidth=2)
    plt.plot(grid_sizes, pinn_infer_times, '^-', label="fPINN Inference (Surrogate Eval)", linewidth=2)
    
    # Log-Log scale is standard for these types of complexity comparisons
    plt.xscale('log', base=2)
    plt.yscale('log')
    
    # Force x-axis ticks to match our grid sizes
    plt.xticks(grid_sizes, [str(g) for g in grid_sizes])
    
    plt.xlabel('Grid Size ($N_x = N_t$)')
    plt.ylabel('Wall-clock Time (seconds)')
    plt.title('Computation Time Comparison: FDM vs. fPINN\n(Log-Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("complexity_fdm_compare.png", dpi=300)
    plt.close()
    print("\nSaved plot to complexity_fdm_compare.png")

if __name__ == "__main__":
    main()
