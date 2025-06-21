#!/bin/bash

# Ensure log directory exists
mkdir -p logs

# Log file path
logfile="logs/log"

# Maximum number of parallel processes
max_parallel=5

# Function to run commands with parallelism control
run_commands() {
    local commands=("$@")
    local count=0

    for cmd in "${commands[@]}"; do
        echo "Starting command: $cmd"
        eval "$cmd" >> "$logfile" 2>&1 &
        pid=$!
        echo "Started [$pid]: $cmd"
        ((count++))

        if [ $count -ge $max_parallel ]; then
            wait -n
            count=$((count - 1))
        fi
    done

    # Wait for all remaining background jobs
    wait
}

# Original commands
original_commands=(
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv3-v5 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.05 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_PPO_Vanilla_eps0.05_3 --best_model"
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv3-v5 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.06 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_PPO_Vanilla_eps0.06_2 --best_model"
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv3-v5 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.07 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_PPO_Vanilla_eps0.07_1 --best_model"
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv3-v5 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.08 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_PPO_Vanilla_eps0.08_4 --best_model"
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv3-v5 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.09 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_PPO_Vanilla_eps0.09_1 --best_model"
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv3-v5 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.1 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_PPO_Vanilla_eps0.10_2 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv3-v5 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.05 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_SAC_Vanilla_eps0.05_0 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv3-v5 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.06 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_SAC_Vanilla_eps0.06_0 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv3-v5 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.07 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_SAC_Vanilla_eps0.07_0 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv3-v5 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.08 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_SAC_Vanilla_eps0.08_3 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv3-v5 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.09 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_SAC_Vanilla_eps0.09_3 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv3-v5 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.1 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv3-v5_SAC_Vanilla_eps0.10_1 --best_model"
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv8-v1 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.05 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_PPO_Vanilla_eps0.05_0 --best_model"
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv8-v1 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.06 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_PPO_Vanilla_eps0.06_3 --best_model"
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv8-v1 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.07 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_PPO_Vanilla_eps0.07_3 --best_model"
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv8-v1 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.08 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_PPO_Vanilla_eps0.08_2 --best_model"
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv8-v1 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.09 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_PPO_Vanilla_eps0.09_3 --best_model"
    "python evaluation_v3.py --algo PPO --env_name TrafficEnv8-v1 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.1 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_PPO_Vanilla_eps0.10_3 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv8-v1 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.05 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_SAC_Vanilla_eps0.05_0 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv8-v1 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.06 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_SAC_Vanilla_eps0.06_1 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv8-v1 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.07 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_SAC_Vanilla_eps0.07_4 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv8-v1 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.08 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_SAC_Vanilla_eps0.08_1 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv8-v1 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.09 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_SAC_Vanilla_eps0.09_0 --best_model"
    "python evaluation_v3.py --algo SAC --env_name TrafficEnv8-v1 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.1 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_SAC_Vanilla_eps0.10_1 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv3-v5 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.05 --expert_recording --addition_msg fix_vec_beta_ens2_TrafficEnv3-v5_TD3_Vanilla_eps0.05_3 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv3-v5 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.06 --expert_recording --addition_msg fix_vec_beta_ens2_TrafficEnv3-v5_TD3_Vanilla_eps0.06_3 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv3-v5 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.07 --expert_recording --addition_msg fix_vec_beta_ens2_TrafficEnv3-v5_TD3_Vanilla_eps0.07_3 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv3-v5 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.08 --expert_recording --addition_msg fix_vec_beta_ens2_TrafficEnv3-v5_TD3_Vanilla_eps0.08_3 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv3-v5 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.09 --expert_recording --addition_msg fix_vec_beta_ens2_TrafficEnv3-v5_TD3_Vanilla_eps0.09_3 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv3-v5 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.1 --expert_recording --addition_msg fix_vec_beta_ens2_TrafficEnv3-v5_TD3_Vanilla_eps0.10_3 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv8-v1 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.05 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_TD3_Vanilla_eps0.05_3 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv8-v1 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.06 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_TD3_Vanilla_eps0.06_4 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv8-v1 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.07 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_TD3_Vanilla_eps0.07_4 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv8-v1 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.08 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_TD3_Vanilla_eps0.08_4 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv8-v1 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.09 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_TD3_Vanilla_eps0.09_3 --best_model"
    "python evaluation_v3.py --algo TD3 --env_name TrafficEnv8-v1 --expert_data_path all_nofilter --train_step 1000 --random_seed --attack --attack_eps 0.1 --expert_recording --addition_msg vec_beta_ens2_TrafficEnv8-v1_TD3_Vanilla_eps0.10_1 --best_model"
)

# Duplicate commands with all_filter
filtered_commands=()
for cmd in "${original_commands[@]}"; do
    filtered_cmd=$(echo "$cmd" | sed 's/--expert_data_path all_nofilter/--expert_data_path all_filter/')
    filtered_commands+=("$filtered_cmd")
done

# Training commands
training_commands=(
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/nf_nbal_0.5 --expert_data_path /data/lxy/STA-Expert/expert_data/all_nofilter --balance False --sampling_ratio 0.5"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/nf_nbal_0.4 --expert_data_path /data/lxy/STA-Expert/expert_data/all_nofilter --balance False --sampling_ratio 0.4"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/nf_nbal_0.3 --expert_data_path /data/lxy/STA-Expert/expert_data/all_nofilter --balance False --sampling_ratio 0.3"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/nf_nbal_0.2 --expert_data_path /data/lxy/STA-Expert/expert_data/all_nofilter --balance False --sampling_ratio 0.2"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/nf_nbal_0.1 --expert_data_path /data/lxy/STA-Expert/expert_data/all_nofilter --balance False --sampling_ratio 0.1"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/nf_nbal_0.05 --expert_data_path /data/lxy/STA-Expert/expert_data/all_nofilter --balance False --sampling_ratio 0.05"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/nf_nbal_0.01 --expert_data_path /data/lxy/STA-Expert/expert_data/all_nofilter --balance False --sampling_ratio 0.01"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/nf_0.5 --expert_data_path /data/lxy/STA-Expert/expert_data/all_nofilter --sampling_ratio 0.5"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/nf_0.4 --expert_data_path /data/lxy/STA-Expert/expert_data/all_nofilter --sampling_ratio 0.4"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/nf_0.3 --expert_data_path /data/lxy/STA-Expert/expert_data/all_nofilter --sampling_ratio 0.3"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/nf_0.2 --expert_data_path /data/lxy/STA-Expert/expert_data/all_nofilter --sampling_ratio 0.2"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/nf_0.1 --expert_data_path /data/lxy/STA-Expert/expert_data/all_nofilter --sampling_ratio 0.1"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/nf_0.05 --expert_data_path /data/lxy/STA-Expert/expert_data/all_nofilter --sampling_ratio 0.05"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/nf_0.01 --expert_data_path /data/lxy/STA-Expert/expert_data/all_nofilter --sampling_ratio 0.01"
)
fil_training_commands=(
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/f_nbal_0.5 --expert_data_path /data/lxy/STA-Expert/expert_data/all_filter --balance False --sampling_ratio 0.5"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/f_nbal_0.4 --expert_data_path /data/lxy/STA-Expert/expert_data/all_filter --balance False --sampling_ratio 0.4"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/f_nbal_0.3 --expert_data_path /data/lxy/STA-Expert/expert_data/all_filter --balance False --sampling_ratio 0.3"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/f_nbal_0.2 --expert_data_path /data/lxy/STA-Expert/expert_data/all_filter --balance False --sampling_ratio 0.2"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/f_nbal_0.1 --expert_data_path /data/lxy/STA-Expert/expert_data/all_filter --balance False --sampling_ratio 0.1"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/f_nbal_0.05 --expert_data_path /data/lxy/STA-Expert/expert_data/all_filter --balance False --sampling_ratio 0.05"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/f_nbal_0.01 --expert_data_path /data/lxy/STA-Expert/expert_data/all_filter --balance False --sampling_ratio 0.01"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/f_0.5 --expert_data_path /data/lxy/STA-Expert/expert_data/all_filter --sampling_ratio 0.5"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/f_0.4 --expert_data_path /data/lxy/STA-Expert/expert_data/all_filter --sampling_ratio 0.4"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/f_0.3 --expert_data_path /data/lxy/STA-Expert/expert_data/all_filter --sampling_ratio 0.3"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/f_0.2 --expert_data_path /data/lxy/STA-Expert/expert_data/all_filter --sampling_ratio 0.2"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/f_0.1 --expert_data_path /data/lxy/STA-Expert/expert_data/all_filter --sampling_ratio 0.1"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/f_0.05 --expert_data_path /data/lxy/STA-Expert/expert_data/all_filter --sampling_ratio 0.05"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/f_0.01 --expert_data_path /data/lxy/STA-Expert/expert_data/all_filter --sampling_ratio 0.01"
)
training_commands1=(
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/nf_nbal_0.005 --expert_data_path /data/lxy/STA-Expert/expert_data/all_nofilter --balance False --sampling_ratio 0.005"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/nf_nbal_0.003 --expert_data_path /data/lxy/STA-Expert/expert_data/all_nofilter --balance False --sampling_ratio 0.003"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/nf_nbal_0.001 --expert_data_path /data/lxy/STA-Expert/expert_data/all_nofilter --balance False --sampling_ratio 0.001"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/nf_nbal_0.0001 --expert_data_path /data/lxy/STA-Expert/expert_data/all_nofilter --balance False --sampling_ratio 0.0001"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/nf_0.005 --expert_data_path /data/lxy/STA-Expert/expert_data/all_nofilter --sampling_ratio 0.005"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/nf_0.003 --expert_data_path /data/lxy/STA-Expert/expert_data/all_nofilter --sampling_ratio 0.003"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/nf_0.001 --expert_data_path /data/lxy/STA-Expert/expert_data/all_nofilter --sampling_ratio 0.001"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/nf_0.0001 --expert_data_path /data/lxy/STA-Expert/expert_data/all_nofilter --sampling_ratio 0.0001"

)
fil_training_commands1=(
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/f_nbal_0.005 --expert_data_path /data/lxy/STA-Expert/expert_data/all_filter --balance False --sampling_ratio 0.005"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/f_nbal_0.003 --expert_data_path /data/lxy/STA-Expert/expert_data/all_filter --balance False --sampling_ratio 0.003"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/f_nbal_0.001 --expert_data_path /data/lxy/STA-Expert/expert_data/all_filter --balance False --sampling_ratio 0.001"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/f_nbal_0.0001 --expert_data_path /data/lxy/STA-Expert/expert_data/all_filter --balance False --sampling_ratio 0.0001"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/f_0.005 --expert_data_path /data/lxy/STA-Expert/expert_data/all_filter --sampling_ratio 0.005"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/f_0.003 --expert_data_path /data/lxy/STA-Expert/expert_data/all_filter --sampling_ratio 0.003"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/f_0.001 --expert_data_path /data/lxy/STA-Expert/expert_data/all_filter --sampling_ratio 0.001"
    "python expert_imitation_learning_MoE.py --expert_model_savepath expert_model/MoEs/f_0.0001 --expert_data_path /data/lxy/STA-Expert/expert_data/all_filter --sampling_ratio 0.0001"
    )
# Prepare evaluation commands with logging
evaluation_commands=()
for cmd in "${original_commands[@]}" "${filtered_commands[@]}"; do
    evaluation_commands+=("$cmd >> $logfile 2>&1")
done

# Prepare training commands with logging
training_commands_all=()
for cmd in "${training_commands1[@]}" "${fil_training_commands1[@]}"; do
    training_commands_all+=("$cmd >> $logfile 2>&1")
done

# Step 1: Run evaluation commands
#echo "Starting evaluation commands..."
#run_commands "${evaluation_commands[@]}"

# Step 2: Run training commands
echo "Starting training commands..."
run_commands "${training_commands_all[@]}"

# Step 3: Define trained model paths from training commands
model_paths=()
#for ratio in 0.5 0.4 0.3 0.2 0.1 0.05 0.01; do
for ratio in 0.005 0.003 0.001 0.0001; do
    model_paths+=("expert_model/MoEs/nf_nbal_${ratio}")
    model_paths+=("expert_model/MoEs/nf_${ratio}")
    model_paths+=("expert_model/MoEs/f_nbal_${ratio}")
    model_paths+=("expert_model/MoEs/f_${ratio}")
done

# Step 4: Generate additional evaluation commands for testing trained models
additional_commands=()
job_index=0
envs=("TrafficEnv3-v5" "TrafficEnv8-v1")
algos=("PPO" "SAC" "TD3")
epsilons=(0.05 0.06 0.07 0.08 0.09 0.10)
for model_path in "${model_paths[@]}"; do
    for env in "${envs[@]}"; do
        for algo in "${algos[@]}"; do
            for eps in "${epsilons[@]}"; do
                addition_msg="vec_beta_ens2_${env}_${algo}_Vanilla_eps${eps}_0"
                log_file="logs/log"
                cuda_number=$(( job_index % 2 ))
                cmd="python evaluation_v3.py --algo ${algo} --env_name ${env} --expert_attack --expert_model_path ${model_path} --result_saving --result_filename MoE_test_all_1 --train_step 100 --attack_eps ${eps} --cuda_number ${cuda_number} --attack --best_model --addition_msg ${addition_msg} > ${log_file} 2>&1"
                additional_commands+=("$cmd")
                job_index=$(( job_index + 1 ))
            done
        done
    done
done

# Step 5: Run additional evaluation commands
echo "Starting additional evaluation commands for trained models..."
run_commands "${additional_commands[@]}"

echo "All tasks completed."
