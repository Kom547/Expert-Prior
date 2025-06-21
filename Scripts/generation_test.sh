#!/bin/bash

# 删除可能已有的命令列表文件，重新生成
gpu0_cmd="gpu0_cmds.txt"
gpu1_cmd="gpu1_cmds.txt"
rm -f "$gpu0_cmd" "$gpu1_cmd"
# 创建logs/test_log目录，如果不存在的话
mkdir -p logs/test_log
job_index=0

env=TrafficEnv3-v5
#for algo in PPO SAC; do
#  if [ $algo = "PPO" ]; then
#    base_algo=SAC
#  else
#    base_algo=PPO
#  fi
#  #algo=TD3
#  for eps in $(seq 0.05 0.01 0.1); do
#    #for seed in $(seq 0 1 4); do
#    addition_msg="vec_beta_ens2_${env}_${base_algo}_Vanilla_eps${eps}_0"
#    cuda_number=$(( job_index % 2 ))
#    #cuda_number=0
#    cmd="python evaluation_v3.py --algo ${algo} --env_name ${env} --expbase_algo ${base_algo} --result_saving --result_filename generation_test --train_step 100 --attack_eps ${eps} --cuda_number ${cuda_number} --attack --best_model --addition_msg ${addition_msg}"
#    log_file="logs/test_log/${addition_msg}.log"
#    if [ $cuda_number -eq 0 ]; then
#        echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu0_cmd"
#    else
#        echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu1_cmd"
#    fi
#    job_index=$(( job_index + 1 ))
#    #done
#  done
#done

#for algo in PPO SAC; do
#  #algo=TD3
#  for eps in $(seq 0.05 0.01 0.1); do
#    #for seed in $(seq 0 1 4); do
#    addition_msg="vec_beta_ens2_${env}_${algo}_Vanilla_eps0.10_0"
#    cuda_number=$(( job_index % 2 ))
#    #cuda_number=0
#    cmd="python evaluation_v3.py --algo ${algo} --env_name ${env} --result_saving --result_filename generation_test --train_step 100 --attack_eps ${eps} --cuda_number ${cuda_number} --attack --best_model --addition_msg ${addition_msg}"
#    log_file="logs/test_log/${addition_msg}.log"
#    if [ $cuda_number -eq 0 ]; then
#        echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu0_cmd"
#    else
#        echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu1_cmd"
#    fi
#    job_index=$(( job_index + 1 ))
#    #done
#  done
#done
for env in TrafficEnv3-v5 TrafficEnv8-v1; do
#env=TrafficEnv8-v1
#for algo in PPO SAC TD3; do
algo=TD3
  for eps in $(seq 0.05 0.01 0.1); do
    #for seed in $(seq 0 1 4); do
    addition_msg="vec_beta_ens2_${env}_${algo}_Vanilla_eps${eps}_0"
    cuda_number=$(( job_index % 2 ))
    #cuda_number=0
    cmd="python evaluation_v3.py --algo ${algo} --env_name ${env} --expert_attack --expert_model_path /data/lxy/STA-Expert/expert_model/MoE_ens_env3env8_500 --result_saving --result_filename TD3Test --train_step 100 --attack_eps ${eps} --cuda_number ${cuda_number} --attack --best_model --addition_msg ${addition_msg}"
    log_file="logs/test_log/expert.log"
    if [ $cuda_number -eq 0 ]; then
        echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu0_cmd"
    else
        echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu1_cmd"
    fi
    job_index=$(( job_index + 1 ))
    done
  #done
done
#for env in TrafficEnv3-v5 TrafficEnv8-v1; do
##env=TrafficEnv8-v1
#for algo in PPO SAC; do
#  for eps in $(seq 0.05 0.01 0.1); do
#    #for seed in $(seq 0 1 4); do
#    addition_msg="vec_beta_ens2_${env}_${algo}_Vanilla_eps${eps}_0"
#    cuda_number=$(( job_index % 2 ))
#    #cuda_number=0
#    cmd="python evaluation_v3.py --algo ${algo} --env_name ${env} --expert_attack --expert_model_path /data/lxy/STA-Expert/expert_model/MoE_ensv2_noES --result_saving --result_filename MoETest --train_step 100 --attack_eps ${eps} --cuda_number ${cuda_number} --attack --best_model --addition_msg ${addition_msg}"
#    log_file="logs/test_log/${addition_msg}.log"
#    if [ $cuda_number -eq 0 ]; then
#        echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu0_cmd"
#    else
#        echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu1_cmd"
#    fi
#    job_index=$(( job_index + 1 ))
#    done
#  done
#done
echo "共生成 $job_index 条命令！"
echo "GPU0 的任务数：$(wc -l < "$gpu0_cmd")"
echo "GPU1 的任务数：$(wc -l < "$gpu1_cmd")"

echo "启动 GPU0 上的任务..."
( cat "$gpu0_cmd" | xargs -I {} -P 10 bash -c '{}' ) &

echo "启动 GPU1 上的任务..."
( cat "$gpu1_cmd" | xargs -I {} -P 10 bash -c '{}' ) &

wait
echo "所有实验任务已完成."