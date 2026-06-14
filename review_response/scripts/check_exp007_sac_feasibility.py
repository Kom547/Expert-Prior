import os
import json

def check_sac_feasibility():
    report = {
        "scenario": "SAC Adversary Training",
        "feasible": False,
        "blockers": [],
        "required_implementations": []
    }
    
    # Check if 'SAC' is imported or used in advTrain.py
    with open("advTrain.py", 'r', encoding='utf-8') as f:
        content = f.read()
        
    if "AdversarialSAC" in content or "SAC" in content:
        # Check if algorithm.py has AdversarialSAC
        with open("algorithm.py", 'r', encoding='utf-8') as f2:
            content2 = f2.read()
            if "AdversarialSAC" in content2:
                report["feasible"] = True
            else:
                report["blockers"].append("advTrain.py references SAC but algorithm.py does not define AdversarialSAC.")
                report["required_implementations"].append("algorithm.AdversarialSAC")
    else:
        report["blockers"].append("advTrain.py does not support training SAC adversary (only PPO variants found).")
        report["required_implementations"].append("algorithm.AdversarialSAC")
        report["required_implementations"].append("advTrain.create_model support for SAC")
        
    os.makedirs("review_response/experiment_runs/metadata", exist_ok=True)
    os.makedirs("review_response/experiment_runs/blockers", exist_ok=True)
    os.makedirs("review_response/experiment_runs/commands", exist_ok=True)
    
    if report["feasible"]:
        with open("review_response/experiment_runs/metadata/EXP007_feasibility.json", 'w') as f:
            json.dump(report, f, indent=4)
            
        with open("review_response/experiment_runs/commands/EXP007_sac_adversary.ps1", 'w') as f:
            f.write("# EXP007: SAC Adversary commands\n")
            f.write("python advTrain.py --env_name TrafficEnv1-v0 --algo PPO --adv_algo SAC --seed 0 --expert_attack --expert_prior ValuePenalty --run_id EXP007_sac_aegrl --experiment_id EXP007\n")
            f.write("python advTrain.py --env_name TrafficEnv1-v0 --algo PPO --adv_algo SAC --seed 0 --run_id EXP007_sac_vanilla --experiment_id EXP007\n")
        print("SAC adversary training is feasible. Generated commands.")
    else:
        with open("review_response/experiment_runs/blockers/EXP007_blocker_report.json", 'w') as f:
            json.dump(report, f, indent=4)
        print("SAC adversary training is NOT feasible. Check blocker report.")
        print(json.dumps(report, indent=4))

if __name__ == "__main__":
    check_sac_feasibility()
