import os
import time
import torch
import numpy as np
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import matplotlib.pyplot as plt

from src.physics import PARAMS
from src.models import ForwardGridData
from src.utils import select_device, set_reproducible, configure_precision

device = select_device()
configure_precision(device)

def get_peak_memory():
    """Returns peak memory allocated on GPU in MB, or 0 if CPU."""
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    elif device.type == 'mps':
        try:
            return torch.mps.current_allocated_memory() / (1024 * 1024)
        except:
            return 0.0
    return 0.0

def count_parameters(model):
    return sum(p.numel() for p in model.net.parameters() if p.requires_grad)

def run_arch_complexity_experiment():
    print(f"--- Starting Architecture Complexity Experiment on {device} ---")
    
    architectures = {
        "Baseline (4x64)": [2] + [64]*4 + [2],
        "Shallow/Narrow (2x32)": [2] + [32]*2 + [2],
        "Shallow/Wide (2x128)": [2] + [128]*2 + [2],
        "Deep/Narrow (8x32)": [2] + [32]*8 + [2],
        "Deep/Wide (8x128)": [2] + [128]*8 + [2],
    }
    
    iterations = 500
    results = {}

    run_params = PARAMS.copy()
    
    x_vals = np.linspace(run_params["X_RANGE"][0], run_params["X_RANGE"][1], run_params["N_SPACE"])
    t_vals = np.linspace(run_params["T_RANGE"][0], run_params["T_RANGE"][1], run_params["N_TIME"])
    X, T = np.meshgrid(x_vals, t_vals)
    grid_np = np.hstack((X.flatten()[:, None], T.flatten()[:, None])).astype(dde.config.real(np))
    x_grid = torch.tensor(grid_np, dtype=torch.get_default_dtype(), device=device)

    for arch_name, arch_shape in architectures.items():
        print(f"\nEvaluating Architecture: {arch_name} -> {arch_shape}")
        set_reproducible(42)
        
        # Reset memory stats
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        elif device.type == 'mps':
            torch.mps.empty_cache()

        data = ForwardGridData(x_grid, run_params, device)
        net = dde.nn.FNN(arch_shape, "tanh", "Glorot normal")
        model = dde.Model(data, net)
        model.compile("adam", lr=1e-3)
        
        params_count = count_parameters(model)
        
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
        
        results[arch_name] = {
            'params': params_count,
            'time_per_100': time_per_100,
            'peak_mem_mb': peak_mem
        }
        
        print(f"Parameters: {params_count}")
        print(f"Time for 100 epochs: {time_per_100:.3f} s")
        if peak_mem > 0:
            print(f"Peak VRAM: {peak_mem:.2f} MB")

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    names = list(results.keys())
    times = [results[n]['time_per_100'] for n in names]
    params = [results[n]['params'] for n in names]
    mems = [results[n]['peak_mem_mb'] for n in names]
    
    # 1. Time vs Parameters
    ax = axes[0]
    scatter = ax.scatter(params, times, s=100, c='blue')
    for i, name in enumerate(names):
        ax.annotate(name, (params[i], times[i]), xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Training Time per 100 epochs (s)')
    ax.set_title('Computational Time vs Network Size')
    ax.grid(True)
    ax.set_xscale('log')
    
    # 2. Memory vs Parameters
    ax = axes[1]
    has_mem_data = any(m > 0 for m in mems)
    if has_mem_data:
        ax.scatter(params, mems, s=100, c='red')
        for i, name in enumerate(names):
            ax.annotate(name, (params[i], mems[i]), xytext=(5, 5), textcoords='offset points')
        ax.set_xlabel('Number of Parameters')
        ax.set_ylabel('Peak VRAM Allocated (MB)')
        ax.set_title('Memory Footprint vs Network Size')
        ax.grid(True)
        ax.set_xscale('log')
    else:
        ax.set_title('Memory Footprint (N/A on CPU/MPS)')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("results/complexity_architecture.png")
    print("Saved plot to complexity_architecture.png")
    plt.show()

if __name__ == "__main__":
    run_arch_complexity_experiment()
