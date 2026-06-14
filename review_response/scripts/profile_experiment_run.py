import argparse
import subprocess
import time
import datetime
import os
import platform
import pandas as pd
import torch

def count_parameters(model_path):
    if not os.path.exists(model_path):
        return None
    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict):
            return sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
        return None
    except Exception:
        return None

def profile_command(command, run_id, model_path=None):
    start_time = time.time()
    start_dt = datetime.datetime.now()
    
    print(f"Profiling command: {command}")
    
    process = subprocess.Popen(command, shell=True)
    process.wait()
    
    exit_code = process.returncode
    end_time = time.time()
    end_dt = datetime.datetime.now()
    
    duration = end_time - start_time
    
    param_count = count_parameters(model_path) if model_path else None
    
    # Hardware metadata
    hw_meta = f"{platform.system()} {platform.release()} {platform.machine()}"
    gpu_meta = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
    
    record = {
        "run_id": run_id,
        "command": command,
        "start_time": start_dt,
        "end_time": end_dt,
        "duration_sec": duration,
        "exit_code": exit_code,
        "param_count": param_count,
        "os": hw_meta,
        "gpu": gpu_meta
    }
    
    out_csv = "review_response/experiment_runs/raw_results/EXP006_profiling_raw.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    
    df = pd.DataFrame([record])
    if not os.path.exists(out_csv):
        df.to_csv(out_csv, index=False)
    else:
        df.to_csv(out_csv, mode='a', header=False, index=False)
        
    print(f"Saved profiling record to {out_csv}")
    return exit_code

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wrapper to profile a command")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run")
    parser.add_argument("--run_id", type=str, default="UNKNOWN", help="Run ID")
    parser.add_argument("--model_path", type=str, default="", help="Path to model to count parameters")
    args = parser.parse_args()
    
    cmd_str = " ".join(args.command)
    profile_command(cmd_str, args.run_id, args.model_path if args.model_path else None)
