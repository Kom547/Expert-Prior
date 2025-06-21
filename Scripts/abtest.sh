#!/bin/bash
# 删除可能已有的命令列表文件，重新生成
gpu0_cmd="gpu0_cmds.txt"
gpu1_cmd="gpu1_cmds.txt"
rm -f "$gpu0_cmd" "$gpu1_cmd"
# 创建logs/test_log目录，如果不存在的话
mkdir -p logs/test_log
job_index=0

#vanilla
for env in TrafficEnv3-v5 TrafficEnv8-v1; do
    for algo in PPO SAC TD3; do
      for eps in 0.05 0.08 0.10; do
        for seed in $(seq 0 1 4); do
        for advsteps in 4 7 9; do
        addition_msg="MoE_rb0.5_ts500_${env}_${algo}_Vanilla_eps${eps}_as${advsteps}_${seed}"
        cuda_number=$(( job_index % 2 ))
        #cuda_number=0
        #cmd="python advTrain.py --no_wandb --adv_steps ${advsteps} --cuda_seed ${seed} --n_steps 128 --train_step 500 --addition_msg ${addition_msg} --num_envs 8 --attack --env_name ${env} --algo ${algo} --attack_eps ${eps} --cuda_number ${cuda_number}"
        cmd="python evaluation_v3.py --result_saving --adv_steps ${advsteps} --result_filename AttackEvaluation_0616_lunar_withas --algo ${algo} --env_name ${env} --train_step 100 --attack_eps ${eps} --attack --cuda_number ${cuda_number} --addition_msg ${addition_msg}"

        log_file="logs/test_log/${addition_msg}.log"
        if [ $cuda_number -eq 0 ]; then
            echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu0_cmd"
        else
            echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu1_cmd"
        fi
        job_index=$(( job_index + 1 ))
        done
        done
        done
      done
    done

# duizhaozu
#for env in TrafficEnv3-v5 TrafficEnv8-v1; do
#  for eprior in ValuePenalty PolicyConstrained; do
#    for algo in PPO SAC TD3; do
#      for eps in 0.05 0.08 0.10; do
##        if [ $env == TrafficEnv3-v5 ] && [ $algo == PPO ] && [ $eps == 0.05 ]; then
##        continue
##        fi
#        for seed in $(seq 0 1 4); do
#        for advsteps in 4 7 9; do
#        addition_msg="MoE_rb0.5_beta_${env}_${algo}_${eprior}_eps${eps}_as${advsteps}_${seed}"
#        cuda_number=$(( job_index % 2 ))
#        #cuda_number=0
#        #cmd="python advTrain.py --no_wandb --adv_steps ${advsteps} --cuda_seed ${seed} --n_steps 128 --train_step 200 --addition_msg ${addition_msg} --num_envs 8 --attack --env_name ${env} --algo ${algo} --attack_eps ${eps} --expert_model_path /data/lxy/STA-Expert/expert_model/MoEs/f_0.5 --expert_attack --cuda_number ${cuda_number} --expert_prior ${eprior}"
#        cmd="python evaluation_v3.py --result_saving --result_filename AttackEvaluation_MoE_rb05_fix --algo ${algo} --env_name ${env} --train_step 100 --attack_eps ${eps} --attack --cuda_number ${cuda_number} --best_model --addition_msg ${addition_msg}"
#
#        log_file="logs/test_log/${addition_msg}.log"
#        if [ $cuda_number -eq 0 ]; then
#            echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu0_cmd"
#        else
#            echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu1_cmd"
#        fi
#        job_index=$(( job_index + 1 ))
#        done
#        done
#        done
#      done
#    done
#  done
#xiaorong1
#for env in TrafficEnv3-v5 TrafficEnv8-v1; do
#  eprior=ValuePenalty
#  for algo in PPO SAC TD3; do
#    for eps in  0.05 0.08 0.10; do
##      if [ $env == TrafficEnv3-v5 ] && [ $algo == PPO ] && [ $eps == 0.05 ]; then
##        continue
##      fi
#      for seed in $(seq 0 1 4); do
#      for advsteps in 4 7 9; do
#      addition_msg="MoE_rb0.5_nobeta_${env}_${algo}_${eprior}_eps${eps}_as${advsteps}_${seed}"
#      cuda_number=$(( job_index % 2 ))
#      #cuda_number=0
#      # 拼接完整的命令，根据原命令模板
#      #cmd="python advTrain.py --no_wandb --wo_beta --adv_steps ${advsteps} --cuda_seed ${seed} --n_steps 128 --train_step 200 --addition_msg ${addition_msg} --num_envs 8 --attack --env_name ${env} --algo ${algo} --attack_eps ${eps} --expert_model_path /data/lxy/STA-Expert/expert_model/MoEs/f_0.5 --expert_attack --cuda_number ${cuda_number} --expert_prior ${eprior}"
#      cmd="python evaluation_v3.py --result_saving --result_filename AttackEvaluation_MoE_rb05_fix --algo ${algo} --env_name ${env} --train_step 100 --attack_eps ${eps} --attack --cuda_number ${cuda_number} --best_model --addition_msg ${addition_msg}"
#
#      # 分发到对应 GPU 的命令文件中
#      log_file="logs/test_log/${addition_msg}.log"
#      if [ $cuda_number -eq 0 ]; then
#          echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu0_cmd"
#      else
#          echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu1_cmd"
#      fi
#      job_index=$(( job_index + 1 ))
#      done
#      done
#    done
#  done
#done
#xiaorong2
#for env in TrafficEnv3-v5 TrafficEnv8-v1; do
#  for eprior in ValuePenalty PolicyConstrained; do
#  for algo in PPO SAC TD3; do
#    for eps in 0.05 0.08 0.10; do
#      for seed in $(seq 0 1 4); do
#      for advsteps in 4 7 9; do
#      if [ $eprior == PolicyConstrained ]; then
#        addition_msg="MoE_rb0.5_beta_nlg_k6.5_ts500_${env}_${algo}_${eprior}_eps${eps}_as${advsteps}_${seed}"
#        cuda_number=$(( job_index % 2 ))
#        #cuda_number=0
#        cmd="python advTrain.py --no_wandb --expert_k 6.5 --no_lambda_grad --adv_steps ${advsteps} --cuda_seed ${seed} --n_steps 128 --train_step 500 --addition_msg ${addition_msg} --num_envs 8 --attack --env_name ${env} --algo ${algo} --attack_eps ${eps} --expert_model_path /data/lxy/STA-Expert/expert_model/MoEs/f_0.5 --expert_attack --cuda_number ${cuda_number} --expert_prior ${eprior}"
#      else
#        addition_msg="MoE_rb0.5_beta_k6.5_ts500_${env}_${algo}_${eprior}_eps${eps}_as${advsteps}_${seed}"
#        cuda_number=$(( job_index % 2 ))
#        #cuda_number=0
#        cmd="python advTrain.py --no_wandb --expert_k 6.5 --adv_steps ${advsteps} --cuda_seed ${seed} --n_steps 128 --train_step 500 --addition_msg ${addition_msg} --num_envs 8 --attack --env_name ${env} --algo ${algo} --attack_eps ${eps} --expert_model_path /data/lxy/STA-Expert/expert_model/MoEs/f_0.5 --expert_attack --cuda_number ${cuda_number} --expert_prior ${eprior}"
#        #cmd="python evaluation_v3.py --result_saving --result_filename AttackEvaluation_MoE_rb05_fix --algo ${algo} --env_name ${env} --train_step 100 --attack_eps ${eps} --attack --cuda_number ${cuda_number} --best_model --addition_msg ${addition_msg}"
#      fi
#      log_file="logs/test_log/${addition_msg}.log"
#      if [ $cuda_number -eq 0 ]; then
#          echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu0_cmd"
#      else
#          echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu1_cmd"
#      fi
#      job_index=$(( job_index + 1 ))
#      done
#      done
#    done
#  done
#  done
#  done
#xiaorong3
#for env in TrafficEnv3-v5 TrafficEnv8-v1; do
#  eprior=PolicyConstrained
#  for algo in PPO SAC TD3; do
#    for eps in 0.05 0.08 0.10; do
##      if [ $env == TrafficEnv3-v5 ] && [ $algo == PPO ] && [ $eps == 0.05 ]; then
##        continue
##      fi
#      for seed in $(seq 0 1 4); do
#      for advsteps in 4 7 9; do
#      addition_msg="MoE_rb0.5_beta_nlg_${env}_${algo}_${eprior}_eps${eps}_as${advsteps}_${seed}"
#      cuda_number=$(( job_index % 2 ))
#      #cuda_number=0
#      #cmd="python advTrain.py --no_wandb --no_lambda_grad --adv_steps ${advsteps} --cuda_seed ${seed} --n_steps 128 --train_step 200 --addition_msg ${addition_msg} --num_envs 8 --attack --env_name ${env} --algo ${algo} --attack_eps ${eps} --expert_model_path /data/lxy/STA-Expert/expert_model/MoEs/f_0.5 --expert_attack --cuda_number ${cuda_number} --expert_prior ${eprior}"
#      cmd="python evaluation_v3.py --result_saving --result_filename AttackEvaluation_MoE_rb05_fix --algo ${algo} --env_name ${env} --train_step 100 --attack_eps ${eps} --attack --cuda_number ${cuda_number} --best_model --addition_msg ${addition_msg}"
#      # 分发到对应 GPU 的命令文件中
#      log_file="logs/test_log/${addition_msg}.log"
#      if [ $cuda_number -eq 0 ]; then
#          echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu0_cmd"
#      else
#          echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu1_cmd"
#      fi
#      job_index=$(( job_index + 1 ))
#      done
#      done
#      done
#      done
#      done
#xiaorong4
#for env in TrafficEnv3-v5 TrafficEnv8-v1; do
#for eprior in ValuePenalty PolicyConstrained; do
#  for algo in PPO SAC TD3; do
#    for eps in 0.05 0.08 0.10; do
##      if [ $env == TrafficEnv3-v5 ] && [ $algo == PPO ] && [ $eps == 0.05 ]; then
##        continue
##      fi
#      for seed in $(seq 0 1 4); do
#      for advsteps in 4 7 9; do
#      addition_msg="MoE_rb0.5_single_beta_${env}_${algo}_${eprior}_eps${eps}_as${advsteps}_${seed}"
#      cuda_number=$(( job_index % 2 ))
#      #cuda_number=0
#      #cmd="python advTrain.py --no_wandb --expert_cnt 2 --adv_steps ${advsteps} --cuda_seed ${seed} --n_steps 128 --train_step 200 --addition_msg ${addition_msg} --num_envs 8 --attack --env_name ${env} --algo ${algo} --attack_eps ${eps} --expert_model_path /data/lxy/STA-Expert/expert_model/MoEs/f_0.5 --expert_attack --cuda_number ${cuda_number} --expert_prior ${eprior}"
#      cmd="python evaluation_v3.py --result_saving --result_filename AttackEvaluation_MoE_rb05_fix --algo ${algo} --env_name ${env} --train_step 100 --attack_eps ${eps} --attack --cuda_number ${cuda_number} --best_model --addition_msg ${addition_msg}"
#      log_file="logs/test_log/${addition_msg}.log"
#      if [ $cuda_number -eq 0 ]; then
#          echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu0_cmd"
#      else
#          echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu1_cmd"
#      fi
#      job_index=$(( job_index + 1 ))
#      done
#      done
#    done
#  done
#done
#done
for env in TrafficEnv3-v5 TrafficEnv8-v1; do
  eprior=PolicyConstrained
  for algo in PPO SAC TD3; do
    for eps in 0.05 0.08 0.10; do
#      if [ $env == TrafficEnv3-v5 ] && [ $algo == PPO ] && [ $eps == 0.05 ]; then
#        continue
#      fi
      for seed in $(seq 0 1 4); do
      for advsteps in 4 7 9; do
      addition_msg="MoE_rb0.5_beta_nlg_k6.5_ts500_${env}_${algo}_${eprior}_eps${eps}_as${advsteps}_${seed}"
      cuda_number=$(( job_index % 2 ))
      #cuda_number=0
      #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      #cmd="python advTrain.py --no_wandb --expert_k 6.5 --no_lambda_grad --adv_steps ${advsteps} --cuda_seed ${seed} --n_steps 4096 --train_step 500 --addition_msg ${addition_msg} --num_envs 1 --attack --env_name ${env} --algo ${algo} --attack_eps ${eps} --expert_model_path /data/lxy/STA-Expert/expert_model/MoEs/f_0.5 --expert_attack --cuda_number ${cuda_number} --expert_prior ${eprior}"
      cmd="python evaluation_v3.py --adv_steps ${advsteps} --result_saving --result_filename AttackEvaluation_0616_lunar_withas --algo ${algo} --env_name ${env} --train_step 100 --attack_eps ${eps} --attack --cuda_number ${cuda_number} --addition_msg ${addition_msg}"
      # 分发到对应 GPU 的命令文件中
      log_file="logs/test_log/${addition_msg}.log"
      if [ $cuda_number -eq 0 ]; then
          echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu0_cmd"
      else
          echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu1_cmd"
      fi
      job_index=$(( job_index + 1 ))
      done
      done
      done
      done
      done
#for env in TrafficEnv3-v5 TrafficEnv8-v1; do
#  eprior=PolicyConstrained
#  for algo in PPO SAC TD3; do
#    for eps in 0.05 0.08 0.10; do
##      if [ $env == TrafficEnv3-v5 ] && [ $algo == PPO ] && [ $eps == 0.05 ]; then
##        continue
##      fi
#      for seed in $(seq 0 1 4); do
#      for advsteps in 4 7 9; do
#      addition_msg="MoE_rb0.5_beta_nlg_k6.5_${env}_${algo}_${eprior}_eps${eps}_as${advsteps}_${seed}"
#      cuda_number=$(( job_index % 2 ))
#      #cuda_number=0
#      #cmd="python advTrain.py --no_wandb --expert_k 6.5 --no_lambda_grad --adv_steps ${advsteps} --cuda_seed ${seed} --n_steps 128 --train_step 200 --addition_msg ${addition_msg} --num_envs 8 --attack --env_name ${env} --algo ${algo} --attack_eps ${eps} --expert_model_path /data/lxy/STA-Expert/expert_model/MoEs/f_0.5 --expert_attack --cuda_number ${cuda_number} --expert_prior ${eprior}"
#      cmd="python evaluation_v3.py --result_saving --result_filename AttackEvaluation_MoE_rb05_fix --algo ${algo} --env_name ${env} --train_step 100 --attack_eps ${eps} --attack --cuda_number ${cuda_number} --best_model --addition_msg ${addition_msg}"
#      # 分发到对应 GPU 的命令文件中
#      log_file="logs/test_log/${addition_msg}.log"
#      if [ $cuda_number -eq 0 ]; then
#          echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu0_cmd"
#      else
#          echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu1_cmd"
#      fi
#      job_index=$(( job_index + 1 ))
#      done
#      done
#      done
#      done
#      done
for env in TrafficEnv3-v5 TrafficEnv8-v1; do
  for eprior in ValuePenalty PolicyConstrained; do
  for algo in PPO SAC TD3; do
    for eps in 0.05 0.08 0.10; do
      for seed in $(seq 0 1 4); do
      for advsteps in 4 7 9; do
        addition_msg="MoE_rb0.5_nobeta_ts500_${env}_${algo}_${eprior}_eps${eps}_as${advsteps}_${seed}"
        cuda_number=$(( job_index % 2 ))
        #cuda_number=0
        #cmd="python advTrain.py --no_wandb --wo_beta --adv_steps ${advsteps} --cuda_seed ${seed} --n_steps 128 --train_step 500 --addition_msg ${addition_msg} --num_envs 8 --attack --env_name ${env} --algo ${algo} --attack_eps ${eps} --expert_model_path /data/lxy/STA-Expert/expert_model/MoEs/f_0.5 --expert_attack --cuda_number ${cuda_number} --expert_prior ${eprior}"
        cmd="python evaluation_v3.py --adv_steps ${advsteps} --result_saving --result_filename AttackEvaluation_0616_lunar_withas --algo ${algo} --env_name ${env} --train_step 100 --attack_eps ${eps} --attack --cuda_number ${cuda_number} --addition_msg ${addition_msg}"

      log_file="logs/test_log/${addition_msg}.log"
      if [ $cuda_number -eq 0 ]; then
          echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu0_cmd"
      else
          echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu1_cmd"
      fi
      job_index=$(( job_index + 1 ))
      done
      done
    done
  done
  done
  done
echo "共生成 $job_index 条命令！"
echo "GPU0 的任务数：$(wc -l < "$gpu0_cmd")"
echo "GPU1 的任务数：$(wc -l < "$gpu1_cmd")"

# 分别启动两个 GPU 的任务，每个 GPU 最多并行运行 10 个任务
echo "启动 GPU0 上的任务..."
( cat "$gpu0_cmd" | xargs -I {} -P 10 bash -c '{}' ) &

echo "启动 GPU1 上的任务..."
( cat "$gpu1_cmd" | xargs -I {} -P 10 bash -c '{}' ) &

# 等待所有后台任务运行完毕
wait
echo "所有实验任务已完成."
