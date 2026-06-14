import argparse
import subprocess
import os
import time
import datetime

def run_manifest(script_path, log_dir, parallel=False):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(script_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    commands = []
    current_run_id = None
    for line in lines:
        line = line.strip()
        if line.startswith("# Run ID:"):
            current_run_id = line.split(":", 1)[1].strip()
        elif line and not line.startswith("#"):
            if current_run_id:
                commands.append((current_run_id, line))
            else:
                commands.append((f"run_{int(time.time())}", line))

    print(f"Found {len(commands)} commands to run.")
    
    # Sequential execution for now
    for run_id, cmd in commands:
        print(f"[{datetime.datetime.now()}] Starting run {run_id}")
        log_file = os.path.join(log_dir, f"{run_id}.log")
        start_time = time.time()
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Command: {cmd}\n")
            f.write(f"Start Time: {datetime.datetime.now()}\n")
            f.write("-" * 40 + "\n")
            f.flush()
            
            try:
                # Use subprocess to run the command and capture output
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    f.write(line)
                    f.flush()
                    # Optional: print to console
                    # print(line, end="")
                process.wait()
                exit_code = process.returncode
            except Exception as e:
                f.write(f"\nExecution failed: {e}\n")
                exit_code = -1
                
            end_time = time.time()
            f.write("-" * 40 + "\n")
            f.write(f"End Time: {datetime.datetime.now()}\n")
            f.write(f"Duration: {end_time - start_time:.2f} seconds\n")
            f.write(f"Exit Code: {exit_code}\n")
            
        print(f"[{datetime.datetime.now()}] Finished run {run_id} with exit code {exit_code}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, required=True, help="Path to generated command script (.ps1 or .sh)")
    parser.add_argument("--log_dir", type=str, default="review_response/experiment_runs/logs", help="Directory to save logs")
    parser.add_argument("--parallel", action="store_true", help="Run commands in parallel (not fully implemented)")
    args = parser.parse_args()
    
    run_manifest(args.script, args.log_dir, args.parallel)
