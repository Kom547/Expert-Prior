import os
import glob
import pandas as pd
import yaml

TARGET_ENVS = ['TrafficEnv1-v0', 'TrafficEnv3-v5', 'TrafficEnv5-v0', 'TrafficEnv8-v1']
TARGET_VICTIMS = ['PPO', 'SAC']  # TD3 usually for Env3
TARGET_METHODS = ['Vanilla', 'ValuePenalty']
TARGET_SEEDS = [0, 10, 20, 30, 40]

def scan_results(eval_dir="evaluation_result", logs_dir="logs/adv_eval"):
    records = []
    
    # Scan evaluation_result/*.csv
    if os.path.exists(eval_dir):
        for csv_file in glob.glob(f"{eval_dir}/*.csv"):
            try:
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    records.append(row.to_dict())
            except Exception as e:
                print(f"Failed to read {csv_file}: {e}")
                
    # Scanning logs/adv_eval/**/rollout_log.csv can be complex, skipping for simplicity unless required
    
    if not records:
        print("No existing results found.")
        return pd.DataFrame()
        
    return pd.DataFrame(records)

def check_missing(df):
    missing_runs = []
    
    # Simplify check if df is empty
    if df.empty:
        existing_keys = set()
    else:
        # Assuming df has 'env_name', 'algo', 'method', 'seed'
        # Currently evaluation_v3.py doesn't record seed natively in the CSV, 
        # but we might need to rely on run_id or just general combinations.
        if 'seed' in df.columns:
            existing_keys = set(zip(df['env_name'], df['algo'], df['method'], df['seed']))
        else:
            print("Warning: 'seed' column not found in results. Treating all seeds as missing for combinations without any result.")
            existing_keys = set(zip(df.get('env_name', []), df.get('algo', []), df.get('method', [])))

    for env in TARGET_ENVS:
        for victim in TARGET_VICTIMS:
            # Simple rule: TD3 is mainly for Env3 in some setups, but let's stick to PPO/SAC
            for method in TARGET_METHODS:
                for seed in TARGET_SEEDS:
                    if df.empty:
                        missing_runs.append((env, victim, method, seed))
                    elif 'seed' in df.columns:
                        if (env, victim, method, seed) not in existing_keys:
                            missing_runs.append((env, victim, method, seed))
                    else:
                        # Just checking if any result exists for the env/victim/method combination
                        if (env, victim, method) not in existing_keys:
                            missing_runs.append((env, victim, method, seed))
                            
    return missing_runs

def generate_missing_yaml(missing_runs, output_yaml="review_response/configs/exp004_missing.yaml"):
    runs = []
    # Base expert model path based on the server scripts you provided
    expert_model_base_path = "/data/lxy/STA-Expert/expert_model/MoEs/f_0.5"

    for env, victim, method, seed in missing_runs:
        cmd = f"python evaluation_v3.py --env_name {env} --algo {victim} --seed {seed} --result_saving --result_filename EXP004_main_results"
        if method != "Vanilla":
            cmd += f" --expert_attack --expert_model_path {expert_model_base_path} --expert_prior {method}"
        runs.append({"command": cmd})
        
    config = {
        "experiment_id": "EXP004",
        "runs": runs
    }
    
    os.makedirs(os.path.dirname(output_yaml), exist_ok=True)
    with open(output_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    print(f"Generated missing runs config at {output_yaml} with {len(runs)} commands.")

def main():
    print("Scanning for existing results...")
    df = scan_results()
    
    if not df.empty:
        os.makedirs("review_response/experiment_runs/raw_results", exist_ok=True)
        df.to_csv("review_response/experiment_runs/raw_results/EXP004_main_results.csv", index=False)
        print("Standardized existing results to review_response/experiment_runs/raw_results/EXP004_main_results.csv")
        
    missing_runs = check_missing(df)
    print(f"Found {len(missing_runs)} missing combinations.")
    
    generate_missing_yaml(missing_runs)

if __name__ == "__main__":
    main()
