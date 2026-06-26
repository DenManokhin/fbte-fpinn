import os
import sys
import time
import subprocess
import torch
import numpy as np

# We only set this if we are running the actual benchmark, not the wrapper
if "--run-benchmark" in sys.argv:
    os.environ["DDE_BACKEND"] = "pytorch"
    import deepxde as dde
    from src.physics import PARAMS
    from src.models import ForwardGridData
    from src.utils import select_device, set_reproducible, configure_precision

def run_single_benchmark():
    device = select_device()
configure_precision(device)
    set_reproducible(42)
    
    run_params = PARAMS.copy()
    run_params["N_SPACE"] = 128
    run_params["N_TIME"] = 100
    
    x_vals = np.linspace(run_params["X_RANGE"][0], run_params["X_RANGE"][1], run_params["N_SPACE"])
    t_vals = np.linspace(run_params["T_RANGE"][0], run_params["T_RANGE"][1], run_params["N_TIME"])
    X, T = np.meshgrid(x_vals, t_vals)
    grid_np = np.hstack((X.flatten()[:, None], T.flatten()[:, None])).astype(dde.config.real(np))
    x_grid = torch.tensor(grid_np, dtype=torch.get_default_dtype(), device=device)

    data = ForwardGridData(x_grid, run_params, device)
    net = dde.nn.FNN([2] + [64]*4 + [2], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    
    iterations = 500
    
    # Warmup
    model.train(iterations=10, display_every=1000)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    
    model.train(iterations=iterations, display_every=1000)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    
    train_time = end_time - start_time
    
    # Inference Time
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_inf = time.time()
    
    model.predict(grid_np)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_inf = time.time()
    
    inf_time = end_inf - start_inf
    
    print(f"RES_TRAIN:{train_time}")
    print(f"RES_INF:{inf_time}")


def run_hardware_comparison():
    import matplotlib.pyplot as plt
    print("--- Starting Hardware Complexity Experiment ---")
    
    script_path = os.path.abspath(__file__)
    python_exe = sys.executable
    
    devices = ['CPU', 'GPU']
    train_times = []
    inf_times = []
    
    for dev in devices:
        print(f"\nEvaluating on {dev}...")
        cmd = [python_exe, script_path, "--run-benchmark"]
        if dev == 'CPU':
            cmd.append("--cpu")
            
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        t_train = 0.0
        t_inf = 0.0
        for line in result.stdout.split('\n'):
            if line.startswith("RES_TRAIN:"):
                t_train = float(line.split(":")[1])
            elif line.startswith("RES_INF:"):
                t_inf = float(line.split(":")[1])
                
        if t_train == 0.0:
            print(f"Error running benchmark on {dev}. Output:\n{result.stderr}")
        else:
            print(f"Training Time (500 epochs): {t_train:.3f} s")
            print(f"Inference Time: {t_inf:.3f} s")
            
        train_times.append(t_train)
        inf_times.append(t_inf)

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].bar(devices, train_times, color=['blue', 'green'])
    axes[0].set_ylabel('Training Time (s) for 500 epochs')
    axes[0].set_title('Training Speed Comparison')
    for i, v in enumerate(train_times):
        axes[0].text(i, v + 0.1, f"{v:.2f}s", ha='center', va='bottom')
        
    axes[1].bar(devices, inf_times, color=['orange', 'red'])
    axes[1].set_ylabel('Inference Time (s)')
    axes[1].set_title('Inference Speed Comparison')
    for i, v in enumerate(inf_times):
        axes[1].text(i, v + 0.001, f"{v:.4f}s", ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig('complexity_hardware.png')
    print("\nSaved plot to complexity_hardware.png")
    plt.show()

if __name__ == "__main__":
    if "--run-benchmark" in sys.argv:
        run_single_benchmark()
    else:
        run_hardware_comparison()
