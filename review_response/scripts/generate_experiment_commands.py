import argparse
import yaml
import os
import uuid

def generate_commands(config_path, output_path, shell_type="ps1"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    commands = []
    
    experiment_id = config.get("experiment_id", "EXP_UNKNOWN")
    runs = config.get("runs", [])
    
    for run in runs:
        # Generate a unique run_id if not provided
        run_id = run.get("run_id", f"{experiment_id}_{uuid.uuid4().hex[:8]}")
        base_cmd = run.get("command", "")
        
        # Append run_id and experiment_id if they are not in the command
        if "--run_id" not in base_cmd:
            base_cmd += f" --run_id {run_id}"
        if "--experiment_id" not in base_cmd:
            base_cmd += f" --experiment_id {experiment_id}"
            
        if shell_type == "ps1":
            cmd_str = f"# Run ID: {run_id}\n{base_cmd}"
        else:
            cmd_str = f"# Run ID: {run_id}\n{base_cmd}"
            
        commands.append(cmd_str)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        if shell_type == "sh":
            f.write("#!/bin/bash\n\n")
        for cmd in commands:
            f.write(cmd + "\n\n")
            
    print(f"Generated {len(commands)} commands to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output", type=str, required=True, help="Path to output script (.ps1 or .sh)")
    parser.add_argument("--shell", type=str, choices=["ps1", "sh"], default="ps1")
    args = parser.parse_args()
    
    generate_commands(args.config, args.output, args.shell)
