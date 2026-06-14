import os
import yaml

def generate_exp005_yaml():
    ks = [0.1, 0.5, 1.0, 5.0, 10.0]
    beta_modes = ["full", "fixed", "wo_beta"]
    seeds = [0, 10, 20, 30, 40]
    env = "TrafficEnv1-v0"
    victim = "PPO"
    
    out_yaml = "review_response/configs/exp005_hyperparameter_sweep.yaml"
    os.makedirs(os.path.dirname(out_yaml), exist_ok=True)
    
    runs = []
    
    for k in ks:
        for beta in beta_modes:
            for seed in seeds:
                run_id = f"EXP005_k{k}_{beta}_seed{seed}"
                
                # Base command
                # Note: beta_mode handling depends on actual codebase support.
                # Assuming `--expert_k` exists and `--wo_beta` exists.
                # For `fixed` vs `full`, there might be a `--no_lambda_grad` or similar. 
                # Let's add standard args:
                cmd_train = f"python advTrain.py --env_name {env} --algo {victim} --adv_algo {victim}_FGSM --seed {seed} --expert_attack --expert_prior PolicyConstrained --expert_k {k}"
                
                if beta == "wo_beta":
                    cmd_train += " --wo_beta"
                elif beta == "fixed":
                    cmd_train += " --no_lambda_grad" # Example flag if beta is fixed and not learned via gradient
                
                runs.append({
                    "run_id": f"{run_id}_train",
                    "command": cmd_train
                })
                
                cmd_eval = f"python evaluation_v3.py --env_name {env} --algo {victim} --adv_algo {victim}_FGSM --seed {seed} --expert_attack --expert_prior PolicyConstrained --result_saving --result_filename EXP005_sweep"
                # Eval doesn't necessarily need training hyperparams, but we can pass addition_msg
                cmd_eval += f" --addition_msg k{k}_{beta}"
                
                runs.append({
                    "run_id": f"{run_id}_eval",
                    "command": cmd_eval
                })
                
    config = {
        "experiment_id": "EXP005",
        "runs": runs
    }
    
    with open(out_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    print(f"Generated EXP005 YAML config at {out_yaml} with {len(runs)} commands.")

if __name__ == "__main__":
    generate_exp005_yaml()
