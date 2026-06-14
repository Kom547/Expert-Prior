import gymnasium as gym
import os
import json
import Environment.environment

def check_feasibility():
    env_name = "TrafficEnv7-v0"
    report = {
        "env_name": env_name,
        "gym_make": False,
        "reset": False,
        "step": False,
        "victim_model_exists": False,
        "observation_shape": None,
        "feasible": False,
        "blockers": []
    }
    
    # 1. Gym Make
    try:
        env = gym.make(env_name, attack=True)
        report["gym_make"] = True
    except Exception as e:
        report["blockers"].append(f"Failed to make env: {e}")
        
    if report["gym_make"]:
        # 2. Reset
        try:
            obs = env.reset()
            report["reset"] = True
            report["observation_shape"] = env.observation_space.shape
        except Exception as e:
            report["blockers"].append(f"Failed to reset env: {e}")
            
    if report["reset"]:
        # 3. Step
        try:
            # Random action
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            report["step"] = True
        except Exception as e:
            report["blockers"].append(f"Failed to step env: {e}")
            
    # 4. Check victim model path
    victim_path = f"logs/eval/{env_name}/PPO/best_model/best_model.zip"
    if os.path.exists(victim_path):
        report["victim_model_exists"] = True
    else:
        # Check lunar fallback
        if os.path.exists(f"logs/eval/{env_name}/PPO/lunar.zip"):
            report["victim_model_exists"] = True
        else:
            report["blockers"].append(f"No PPO victim model found for {env_name} at {victim_path}")
            
    # Feasibility
    if report["gym_make"] and report["reset"] and report["step"] and report["victim_model_exists"]:
        report["feasible"] = True
        
    os.makedirs("review_response/experiment_runs/metadata", exist_ok=True)
    os.makedirs("review_response/experiment_runs/blockers", exist_ok=True)
    os.makedirs("review_response/experiment_runs/commands", exist_ok=True)
    
    if report["feasible"]:
        with open("review_response/experiment_runs/metadata/EXP008_feasibility.json", 'w') as f:
            json.dump(report, f, indent=4)
            
        with open("review_response/experiment_runs/commands/EXP008_third_scenario.ps1", 'w') as f:
            f.write("# EXP008: Third Scenario (TrafficEnv7-v0)\n")
            f.write(f"python advTrain.py --env_name {env_name} --algo PPO --adv_algo PPO_FGSM --seed 0 --expert_attack --expert_prior ValuePenalty --run_id EXP008_aegrl_0 --experiment_id EXP008\n")
            f.write(f"python advTrain.py --env_name {env_name} --algo PPO --adv_algo PPO_FGSM --seed 0 --run_id EXP008_vanilla_0 --experiment_id EXP008\n")
        print("Scenario is feasible! Generated EXP008 commands.")
    else:
        with open("review_response/experiment_runs/blockers/EXP008_blocker_report.json", 'w') as f:
            json.dump(report, f, indent=4)
        print("Scenario is NOT feasible. Check blocker report.")
        print(json.dumps(report, indent=4))

if __name__ == "__main__":
    check_feasibility()
