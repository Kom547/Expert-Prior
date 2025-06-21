#!/bin/bash

# 生成所有命令，保存到数组
model_paths=()
for ratio in 0.5; do
    model_paths+=("expert_model/MoEs/f_${ratio}")
done

envs=("TrafficEnv3-v5")
algos=("TD3")
epsilons=(0.06 0.07 0.09)
adv_steps=(4 7 9)

# 函数定义
run_cmd() {
    local model_path="$1"
    local env="$2"
    local algo="$3"
    local eps="$4"
    local adv_steps="$5"
    local job_index="$6"

    local cuda_number=$(( job_index % 2 ))
    local addition_msg="fix_vec_beta_ens2_${env}_${algo}_Vanilla_eps${eps}_0"
    #local addition_msg="DARRL_18_Vanilla_0.05_as4_0"
    local log_file="logs/test_log/${env}_${algo}_eps${eps}_as${adv_steps}.log"
    LOG_DIR=$(dirname "$log_file")
    if [ ! -d "$LOG_DIR" ]; then
        echo "Creating log directory: $LOG_DIR"
        mkdir -p "$LOG_DIR"
    fi

    python evaluation_v3.py \
        --algo "$algo" \
        --env_name "$env" \
        --expert_attack \
        --expert_model_path "$model_path" \
        --result_saving \
        --result_filename MoE_rb0.5_test_withas \
        --train_step 100 \
        --attack_eps "$eps" \
        --cuda_number "$cuda_number" \
        --attack \
        --best_model \
        --adv_steps "$adv_steps" \
        --addition_msg "$addition_msg" > "$log_file" 2>&1 \
        --fni_model_path "policy_v272_20250604_18weights"
}

export -f run_cmd

# 使用 parallel 并行执行，遍历 adv_stepss
job_index=0
for adv_steps in "${adv_steps[@]}"; do
    parallel --jobs 10 '
        run_cmd "{1}" "{2}" "{3}" "{4}" "'"$adv_steps"'" "'"$job_index"'"
    ' ::: "${model_paths[@]}" ::: "${envs[@]}" ::: "${algos[@]}" ::: "${epsilons[@]}"
done
