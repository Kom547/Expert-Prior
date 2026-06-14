import os

def generate_exp002_commands():
    sources = ["vanilla_14_relaxed", "heuristic_random_time_direction"]
    env = "TrafficEnv3-v5"
    victim = "PPO"
    seeds = [0, 10, 20, 30, 40]
    
    out_dir = "review_response/experiment_runs/commands"
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "EXP002_source_agnosticism.ps1"), 'w') as f:
        f.write("# EXP002: Demo Source Agnosticism Commands\n\n")
        
        for source in sources:
            f.write(f"# === SOURCE: {source} ===\n")
            for seed in seeds:
                run_base = f"EXP002_{source}_{env}_{victim}_seed{seed}"
                
                # 1. Generation
                if source == "heuristic_random_time_direction":
                    data_dir = f"expert_data/EXP002_{source}_{seed}"
                    f.write(f"# Run ID: {run_base}_gen\n")
                    f.write(f"python review_response/scripts/generate_heuristic_demo_source.py --env_name {env} --algo {victim} --seed {seed} --out_dir {data_dir} --run_id {run_base}_gen --experiment_id EXP002\n\n")
                else:
                    data_dir = f"expert_data/EXP002_{source}_{seed}"
                    f.write(f"# Run ID: {run_base}_gen\n")
                    # Assuming vanilla relaxed is just regular evaluation collection but with specific eps/steps
                    f.write(f"python evaluation_v3.py --env_name {env} --algo {victim} --seed {seed} --expert_recording --attack_eps 0.1 --adv_steps 14 --expert_data_path EXP002_{source}_{seed} --run_id {run_base}_gen --experiment_id EXP002\n\n")
                
                # 2. MoE Training
                model_dir = f"expert_model/{run_base}"
                f.write(f"# Run ID: {run_base}_train_moe\n")
                f.write(f"python expert_imitation_learning_MoE.py --expert_data_path {data_dir} --expert_model_savepath {model_dir} --seed {seed} --run_id {run_base}_train_moe --experiment_id EXP002\n\n")
                
                # 3. AEGRL Training
                f.write(f"# Run ID: {run_base}_train_aegrl\n")
                f.write(f"python advTrain.py --env_name {env} --algo {victim} --adv_algo {victim}_FGSM --seed {seed} --expert_attack --expert_model_path {model_dir} --expert_prior ValuePenalty --addition_msg source_{source} --run_id {run_base}_train_aegrl --experiment_id EXP002\n\n")
                
                # 4. AEGRL Eval
                f.write(f"# Run ID: {run_base}_eval_aegrl\n")
                f.write(f"python evaluation_v3.py --env_name {env} --algo {victim} --adv_algo {victim}_FGSM --seed {seed} --expert_attack --expert_model_path {model_dir} --expert_prior ValuePenalty --addition_msg source_{source} --result_saving --result_filename EXP002_agnosticism --run_id {run_base}_eval_aegrl --experiment_id EXP002\n\n")
                
    print("Generated EXP002 commands successfully.")

if __name__ == "__main__":
    generate_exp002_commands()
