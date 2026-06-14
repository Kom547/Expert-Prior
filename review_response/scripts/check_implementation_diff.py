import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Compare current implementation with STA-Expert on the server")
    parser.add_argument("--target_dir", type=str, default="/data/lxy/STA-Expert", help="Target directory to compare against")
    parser.add_argument("--exclude_dirs", type=str, nargs="+", default=[
        ".git", "__pycache__", "logs", "expert_data", "expert_model", 
        "review_response", "evaluation_result", "runs", ".idea"
    ], help="Directories to exclude from diff")
    parser.add_argument("--output_file", type=str, default="review_response/experiment_runs/implementation_diff.patch", help="Output patch file")
    
    args = parser.parse_args()
    
    current_dir = os.getcwd()
    
    if not os.path.exists(args.target_dir):
        print(f"Error: Target directory {args.target_dir} does not exist.")
        print(f"Please ensure you are running this script on the remote server where {args.target_dir} exists.")
        return
        
    print(f"Comparing current directory ({current_dir}) with {args.target_dir}...")
    
    # Build diff command
    exclude_flags = []
    for d in args.exclude_dirs:
        exclude_flags.append(f"--exclude={d}")
    exclude_flags.extend(["--exclude=*.pyc", "--exclude=*.zip", "--exclude=*.csv", "--exclude=*.npz", "--exclude=*.pth"])
    
    # -ur means unified diff, recursive
    cmd = ["diff", "-ur"] + exclude_flags + [args.target_dir, current_dir]
    
    # Run diff and save to file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, "w", encoding='utf-8') as f:
        # Popen doesn't wait automatically, so we use communicate
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
        _, stderr = process.communicate()
        
    if stderr:
        print(f"Warnings during diff:\n{stderr}")
        
    print(f"Full unified diff saved to: {args.output_file}")
    
    # Print summary (files that differ)
    print("\n--- Summary of Changed Files ---")
    summary_cmd = ["diff", "-qr"] + exclude_flags + [args.target_dir, current_dir]
    summary_process = subprocess.Popen(summary_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    summary_stdout, _ = summary_process.communicate()
    
    # Filter out "Only in" messages to focus on modified files
    changed_files_count = 0
    for line in summary_stdout.splitlines():
        if not line.startswith("Only in"):
            print(line)
            changed_files_count += 1
            
    if changed_files_count == 0:
        print("No files differ between the two implementations (ignoring excluded dirs and new files).")

if __name__ == "__main__":
    main()
