import os
import time
import torch
import numpy as np
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import matplotlib.pyplot as plt

from src.physics import PARAMS
from src.matrices import torch_riesz_matrix, torch_l1_matrix
from src.models import ForwardGridData
from src.utils import select_device, set_reproducible

device = select_device()

def get_peak_memory():
    """Returns peak memory allocated on GPU in MB, or 0 if CPU."""
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    elif device.type == 'mps':
        # MPS doesn't have max_memory_allocated, we approximate with current.
        # This will only be accurate if called right after the peak operation,
        # but since Colab (CUDA) is the target, this is mostly a fallback.
        try:
            return torch.mps.current_allocated_memory() / (1024 * 1024)
        except:
            return 0.0
    return 0.0

def run_grid_complexity_experiment():
    print(f"--- Starting Grid Complexity Experiment on {device} ---")
    
    n_space_list = [32, 64, 128]
    n_time_list = [50, 100, 200]
    
    iterations = 500
    
    results = {}

    for nx in n_space_list:
        for nt in n_time_list:
            print(f"\nEvaluating Nx={nx}, Nt={nt}")
            set_reproducible(42)
            
            # Reset memory stats
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            elif device.type == 'mps':
                torch.mps.empty_cache()

            # Prepare params for this run
            run_params = PARAMS.copy()
            run_params["N_SPACE"] = nx
            run_params["N_TIME"] = nt
            
            x_vals = np.linspace(run_params["X_RANGE"][0], run_params["X_RANGE"][1], nx)
            t_vals = np.linspace(run_params["T_RANGE"][0], run_params["T_RANGE"][1], nt)
            X, T = np.meshgrid(x_vals, t_vals)
            grid_np = np.hstack((X.flatten()[:, None], T.flatten()[:, None])).astype(np.float32)
            x_grid = torch.tensor(grid_np, dtype=torch.float32, device=device)

            data = ForwardGridData(x_grid, run_params, device)
            net = dde.nn.FNN([2] + [64]*4 + [2], "tanh", "Glorot normal")
            model = dde.Model(data, net)
            model.compile("adam", lr=1e-3)
            
            # Warmup
            model.train(iterations=10, display_every=1000)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            
            model.train(iterations=iterations, display_every=1000)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            total_time = end_time - start_time
            time_per_100 = (total_time / iterations) * 100
            
            peak_mem = get_peak_memory()
            
            results[(nx, nt)] = {
                'time_per_100': time_per_100,
                'peak_mem_mb': peak_mem
            }
            
            print(f"Time for 100 epochs: {time_per_100:.3f} s")
            if peak_mem > 0:
                print(f"Peak VRAM: {peak_mem:.2f} MB")

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Time vs Nt for different Nx
    ax = axes[0]
    for nx in n_space_list:
        times = [results[(nx, nt)]['time_per_100'] for nt in n_time_list]
        ax.plot(n_time_list, times, marker='o', label=f'Nx={nx}')
    ax.set_xlabel('Number of Time Steps ($N_t$)')
    ax.set_ylabel('Training Time per 100 epochs (s)')
    ax.set_title('Computational Time vs Grid Size')
    ax.legend()
    ax.grid(True)
    
    # 2. Memory vs Nt for different Nx
    ax = axes[1]
    has_mem_data = False
    for nx in n_space_list:
        mems = [results[(nx, nt)]['peak_mem_mb'] for nt in n_time_list]
        if any(m > 0 for m in mems):
            has_mem_data = True
        ax.plot(n_time_list, mems, marker='s', linestyle='--', label=f'Nx={nx}')
    
    if has_mem_data:
        ax.set_xlabel('Number of Time Steps ($N_t$)')
        ax.set_ylabel('Peak VRAM Allocated (MB)')
        ax.set_title('Memory Footprint vs Grid Size')
        ax.legend()
        ax.grid(True)
    else:
        ax.set_title('Memory Footprint (N/A on CPU/MPS)')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('complexity_grid.png')
    print("Saved plot to complexity_grid.png")
    plt.show()

if __name__ == "__main__":
    run_grid_complexity_experiment()
