import os
import uuid

def generate_exp001_commands():
    # 3x3 victims, 5 seeds
    source_victims = ["PPO", "SAC", "TD3"]
    target_victims = ["PPO", "SAC", "TD3"]
    seeds = [0, 10, 20, 30, 40]
    env = "TrafficEnv3-v5"
    eps = 0.1
    adv_steps = 14
    
    out_dir = "review_response/experiment_runs/commands"
    os.makedirs(out_dir, exist_ok=True)
    
    # Phase 1: Demo collection
    with open(os.path.join(out_dir, "EXP001_collect_demo.ps1"), 'w') as f1:
        f1.write("# EXP001 Phase 1: Demo Collection\n")
        for sv in source_victims:
            for seed in seeds:
                run_id = f"EXP001_demo_{sv}_seed{seed}"
                # e.g., evaluation_v3.py --expert_recording --env_name Env --algo sv --seed seed
                cmd = f"python evaluation_v3.py --env_name {env} --algo {sv} --seed {seed} --expert_recording --attack_eps {eps} --adv_steps {adv_steps} --addition_msg source_{sv} --run_id {run_id} --experiment_id EXP001"
                f1.write(f"# Run ID: {run_id}\n{cmd}\n\n")
                
    # Phase 2: MoE Prior Training
    with open(os.path.join(out_dir, "EXP001_train_prior.ps1"), 'w') as f2:
        f2.write("# EXP001 Phase 2: MoE Expert Training\n")
        for sv in source_victims:
            for seed in seeds:
                run_id = f"EXP001_prior_{sv}_seed{seed}"
                data_path = f"expert_data/{env}_{sv}/"
                save_path = f"expert_model/EXP001_{sv}_seed{seed}"
                cmd = f"python expert_imitation_learning_MoE.py --expert_data_path {data_path} --expert_model_savepath {save_path} --seed {seed} --run_id {run_id} --experiment_id EXP001"
                f2.write(f"# Run ID: {run_id}\n{cmd}\n\n")

    # Phase 3: AEGRL Train and Eval
    with open(os.path.join(out_dir, "EXP001_train_eval_aegrl.ps1"), 'w') as f3:
        f3.write("# EXP001 Phase 3: AEGRL Train/Eval\n")
        for sv in source_victims:
            for tv in target_victims:
                for seed in seeds:
                    run_id = f"EXP001_aegrl_src_{sv}_tgt_{tv}_seed{seed}"
                    expert_path = f"expert_model/EXP001_{sv}_seed{seed}"
                    
                    # Train
                    train_cmd = f"python advTrain.py --env_name {env} --algo {tv} --adv_algo {tv}_FGSM --seed {seed} --expert_attack --expert_model_path {expert_path} --expert_prior ValuePenalty --addition_msg src_{sv}_tgt_{tv} --run_id {run_id}_train --experiment_id EXP001"
                    
                    # Eval
                    eval_cmd = f"python evaluation_v3.py --env_name {env} --algo {tv} --adv_algo {tv}_FGSM --seed {seed} --expert_attack --expert_model_path {expert_path} --expert_prior ValuePenalty --addition_msg src_{sv}_tgt_{tv} --result_saving --result_filename EXP001_cross_victim --run_id {run_id}_eval --experiment_id EXP001"
                    
                    f3.write(f"# Run ID: {run_id}_train\n{train_cmd}\n\n")
                    f3.write(f"# Run ID: {run_id}_eval\n{eval_cmd}\n\n")
                    
    print("Generated EXP001 commands successfully.")

if __name__ == "__main__":
    generate_exp001_commands()
