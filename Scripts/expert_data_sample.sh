#!/bin/bash

# Ensure log directory exists
mkdir -p logs

# Log file path
logfile="logs/log"

# Global notify start
echo "Starting all evaluations..."

commands=(
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv3-v5 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.05 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_PPO_Vanilla_eps0.05_3 --best_model"
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv3-v5 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.06 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_PPO_Vanilla_eps0.06_2 --best_model"
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv3-v5 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.07 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_PPO_Vanilla_eps0.07_1 --best_model"
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv3-v5 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.08 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_PPO_Vanilla_eps0.08_4 --best_model"
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv3-v5 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.09 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_PPO_Vanilla_eps0.09_1 --best_model"
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv3-v5 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.1 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_PPO_Vanilla_eps0.10_2 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv3-v5 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.05 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_SAC_Vanilla_eps0.05_0 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv3-v5 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.06 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_SAC_Vanilla_eps0.06_0 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv3-v5 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.07 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_SAC_Vanilla_eps0.07_0 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv3-v5 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.08 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_SAC_Vanilla_eps0.08_3 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv3-v5 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.09 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_SAC_Vanilla_eps0.09_3 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv3-v5 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.1 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_SAC_Vanilla_eps0.10_1 --best_model"
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv8-v1 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.05 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_PPO_Vanilla_eps0.05_0 --best_model"
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv8-v1 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.06 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_PPO_Vanilla_eps0.06_3 --best_model"
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv8-v1 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.07 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_PPO_Vanilla_eps0.07_3 --best_model"
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv8-v1 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.08 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_PPO_Vanilla_eps0.08_2 --best_model"
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv8-v1 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.09 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_PPO_Vanilla_eps0.09_3 --best_model"
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv8-v1 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.1 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_PPO_Vanilla_eps0.10_3 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv8-v1 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.05 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_SAC_Vanilla_eps0.05_0 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv8-v1 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.06 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_SAC_Vanilla_eps0.06_1 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv8-v1 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.07 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_SAC_Vanilla_eps0.07_4 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv8-v1 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.08 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_SAC_Vanilla_eps0.08_1 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv8-v1 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.09 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_SAC_Vanilla_eps0.09_0 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv8-v1 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.1 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_SAC_Vanilla_eps0.10_1 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv3-v5 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.05 --expert_recording --addition_msg fix_vec_beta_ens2_TrafficEnv3-v5_TD3_Vanilla_eps0.05_3 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv3-v5 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.06 --expert_recording --addition_msg fix_vec_beta_ens2_TrafficEnv3-v5_TD3_Vanilla_eps0.06_3 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv3-v5 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.07 --expert_recording --addition_msg fix_vec_beta_ens2_TrafficEnv3-v5_TD3_Vanilla_eps0.07_3 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv3-v5 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.08 --expert_recording --addition_msg fix_vec_beta_ens2_TrafficEnv3-v5_TD3_Vanilla_eps0.08_3 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv3-v5 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.09 --expert_recording --addition_msg fix_vec_beta_ens2_TrafficEnv3-v5_TD3_Vanilla_eps0.09_3 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv3-v5 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.1 --expert_recording --addition_msg fix_vec_beta_ens2_TrafficEnv3-v5_TD3_Vanilla_eps0.10_3 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv8-v1 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.05 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_TD3_Vanilla_eps0.05_3 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv8-v1 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.06 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_TD3_Vanilla_eps0.06_4 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv8-v1 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.07 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_TD3_Vanilla_eps0.07_4 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv8-v1 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.08 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_TD3_Vanilla_eps0.08_4 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv8-v1 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.09 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_TD3_Vanilla_eps0.09_3 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv8-v1 --expert_data_path all_csv --train_step 100 --random_seed --attack --attack_eps 0.1 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_TD3_Vanilla_eps0.10_1 --best_model"

)

# Declare associative array to map PIDs to commands
declare -A pid_cmd

# Start each command in background with notification and PID tracking
for cmd in "${commands[@]}"; do
    echo "Starting command: $cmd"
    eval "$cmd" >> "$logfile" 2>&1 &
    pid=$!
    pid_cmd[$pid]="$cmd"
    echo "Started [$pid]: $cmd"
done

# Wait for all background jobs and report per-command completion
for pid in "${!pid_cmd[@]}"; do
    if wait "$pid"; then
        echo "Finished [$pid]: ${pid_cmd[$pid]}"
    else
        echo "Failed  [$pid]: ${pid_cmd[$pid]}" >&2
    fi
done

# Global notify end
echo "All evaluations completed."
