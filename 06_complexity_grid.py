import os
import sys
import time
import subprocess
import torch
import numpy as np

# Only import DeepXDE if we are running the actual benchmark (to avoid context/memory leaks in the main plot wrapper)
if "--run-benchmark-cpu" in sys.argv or "--run-benchmark-gpu" in sys.argv:
    os.environ["DDE_BACKEND"] = "pytorch"
    import deepxde as dde
    from src.physics import PARAMS
    from src.models import ForwardGridData
    from src.utils import select_device, set_reproducible, configure_precision

def get_peak_memory(device):
    """Returns peak memory allocated on GPU in MB, or 0 if CPU."""
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    elif device.type == 'mps':
        try:
            return torch.mps.current_allocated_memory() / (1024 * 1024)
        except:
            return 0.0
    return 0.0

def run_single_benchmark():
    device = select_device()
configure_precision(device)
    print(f"--- Running inner benchmark on {device} ---")
    
    n_space_list = [32, 64, 128]
    n_time_list = [50, 100, 200]
    iterations = 500
    
    for nx in n_space_list:
        for nt in n_time_list:
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
            grid_np = np.hstack((X.flatten()[:, None], T.flatten()[:, None])).astype(dde.config.real(np))
            x_grid = torch.tensor(grid_np, dtype=torch.get_default_dtype(), device=device)

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
            peak_mem = get_peak_memory(device)
            
            # Output structured strings to be parsed by the wrapper
            print(f"RES_TIME:{nx}:{nt}:{time_per_100}")
            print(f"RES_MEM:{nx}:{nt}:{peak_mem}")


def run_grid_complexity_experiment():
    import matplotlib.pyplot as plt
    print("--- Starting Grid Complexity Experiment (CPU vs GPU) ---")
    
    script_path = os.path.abspath(__file__)
    python_exe = sys.executable
    
    results_cpu = {}
    results_gpu = {}
    
    # 1. Run CPU Benchmark
    print("Gathering CPU timings...")
    cmd_cpu = [python_exe, script_path, "--run-benchmark-cpu", "--cpu"]
    res_cpu = subprocess.run(cmd_cpu, capture_output=True, text=True)
    for line in res_cpu.stdout.split('\n'):
        if line.startswith("RES_TIME:"):
            _, nx, nt, t = line.split(":")
            results_cpu[(int(nx), int(nt))] = float(t)
            
    # 2. Run GPU Benchmark
    print("Gathering GPU timings and VRAM...")
    cmd_gpu = [python_exe, script_path, "--run-benchmark-gpu"]
    res_gpu = subprocess.run(cmd_gpu, capture_output=True, text=True)
    
    mem_gpu = {}
    for line in res_gpu.stdout.split('\n'):
        if line.startswith("RES_TIME:"):
            _, nx, nt, t = line.split(":")
            results_gpu[(int(nx), int(nt))] = float(t)
        elif line.startswith("RES_MEM:"):
            _, nx, nt, mem = line.split(":")
            mem_gpu[(int(nx), int(nt))] = float(mem)
            
    n_space_list = [32, 64, 128]
    n_time_list = [50, 100, 200]
    
    # Check if CPU and GPU actually ran successfully
    if not results_cpu or not results_gpu:
        print("Error: Subprocesses failed to return data.")
        print("CPU Error:", res_cpu.stderr)
        print("GPU Error:", res_gpu.stderr)
        return

    # Plot results - Figure 1: Computation Time (CPU vs GPU)
    fig_time, axes_time = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    # Plot 1a: CPU Time vs Nt
    ax = axes_time[0]
    for nx in n_space_list:
        times = [results_cpu.get((nx, nt), 0) for nt in n_time_list]
        ax.plot(n_time_list, times, marker='o', label=f'Nx={nx}')
    ax.set_xlabel('Number of Time Steps ($N_t$)')
    ax.set_ylabel('Training Time per 100 epochs (s)')
    ax.set_title('CPU Computational Time vs Grid Size')
    ax.legend()
    ax.grid(True)

    # Plot 1b: GPU Time vs Nt
    ax = axes_time[1]
    for nx in n_space_list:
        times = [results_gpu.get((nx, nt), 0) for nt in n_time_list]
        ax.plot(n_time_list, times, marker='^', label=f'Nx={nx}')
    ax.set_xlabel('Number of Time Steps ($N_t$)')
    ax.set_title('GPU Computational Time vs Grid Size')
    ax.legend()
    ax.grid(True)
    
    fig_time.tight_layout()
    fig_time.savefig('complexity_grid_time.png')
    print("\nSaved time plots to complexity_grid_time.png")
    
    # Plot results - Figure 2: GPU Memory
    fig_mem, ax_mem = plt.subplots(1, 1, figsize=(6, 5))
    has_mem_data = False
    for nx in n_space_list:
        mems = [mem_gpu.get((nx, nt), 0) for nt in n_time_list]
        if any(m > 0 for m in mems):
            has_mem_data = True
        ax_mem.plot(n_time_list, mems, marker='s', linestyle='--', label=f'Nx={nx}')
    
    if has_mem_data:
        ax_mem.set_xlabel('Number of Time Steps ($N_t$)')
        ax_mem.set_ylabel('Peak VRAM Allocated (MB)')
        ax_mem.set_title('GPU Memory Footprint vs Grid Size')
        ax_mem.legend()
        ax_mem.grid(True)
    else:
        ax_mem.set_title('GPU Memory Footprint (N/A or CPU fallback)')
        ax_mem.axis('off')

    fig_mem.tight_layout()
    fig_mem.savefig('complexity_grid_vram.png')
    print("Saved VRAM plot to complexity_grid_vram.png")
    plt.savefig("results/06_complexity_grid.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    if "--run-benchmark-cpu" in sys.argv or "--run-benchmark-gpu" in sys.argv:
        run_single_benchmark()
    else:
        run_grid_complexity_experiment()
