# 删除可能已有的命令列表文件，重新生成
gpu0_cmd="gpu0_cmds.txt"
gpu1_cmd="gpu1_cmds.txt"
rm -f "$gpu0_cmd" "$gpu1_cmd"

job_index=0

# 遍历所有组合：algo（SAC, TD3）、env_name（TrafficEnv3-v0, TrafficEnv8-v0）、attack_eps（0.01 ~ 0.1）
#for algo in SAC TD3; do
#  for env in TrafficEnv3-v0 TrafficEnv8-v0; do
#    # 可以额外提取 env 的前缀作为 addition_msg 的一部分（如去掉 -v0），这里暂直接使用 env
#    for eps in $(seq 0.01 0.01 0.1); do
#      # 定义 addition_msg，比如 "TrafficEnv3-v0_NoExpert_eps0.02"
#      addition_msg="${env}_${algo}_NoExpert_eps${eps}"
#      for eps2 in $(seq 0.01 0.01 0.1); do
#        # 根据 job_index 进行轮询分配：取模 2 得到 gpu 编号（0 或 1）
#        cuda_number=$(( job_index % 2 ))
#        # 拼接完整的命令，根据原命令模板
#        cmd="python evaluation.py --adv_algo PPO_FGSM --addition_msg ${addition_msg} --attack_eps ${eps2} --algo ${algo} --env_name ${env} --train_step 100 --attack --adv_steps 9 --cuda_number ${cuda_number} --attack_method fgsm --best_model"
#        # 分发到对应 GPU 的命令文件中
#        if [ $cuda_number -eq 0 ]; then
#           echo "$cmd" >> "$gpu0_cmd"
#        else
#           echo "$cmd" >> "$gpu1_cmd"
#        fi
#        job_index=$(( job_index + 1 ))
#      done
#    done
#  done
#done
for eps2 in $(seq 0.03 0.01 0.1); do
      # 根据 job_index 进行轮询分配：取模 2 得到 gpu 编号（0 或 1）
      cuda_number=$(( job_index % 2 ))
      # 拼接完整的命令，根据原命令模板
      cmd="python evaluation.py --expert_recording --adv_algo PPO_FGSM --addition_msg Env3_NoExpert_eps0.04 --attack_eps ${eps2} --algo PPO --env_name TrafficEnv3-v0 --train_step 100 --attack --adv_steps 9 --cuda_number ${cuda_number} --attack_method fgsm --best_model"
      # 分发到对应 GPU 的命令文件中
      if [ $cuda_number -eq 0 ]; then
         echo "$cmd" >> "$gpu0_cmd"
      else
         echo "$cmd" >> "$gpu1_cmd"
      fi
      job_index=$(( job_index + 1 ))
    done
for eps2 in $(seq 0.03 0.01 0.1); do
      # 根据 job_index 进行轮询分配：取模 2 得到 gpu 编号（0 或 1）
      cuda_number=$(( job_index % 2 ))
      # 拼接完整的命令，根据原命令模板
      cmd="python evaluation.py --expert_recording --adv_algo PPO_FGSM --addition_msg TrafficEnv3-v0_SAC_NoExpert_eps0.08 --attack_eps ${eps2} --algo SAC --env_name TrafficEnv3-v0 --train_step 100 --attack --adv_steps 9 --cuda_number ${cuda_number} --attack_method fgsm --best_model"
      # 分发到对应 GPU 的命令文件中
      if [ $cuda_number -eq 0 ]; then
         echo "$cmd" >> "$gpu0_cmd"
      else
         echo "$cmd" >> "$gpu1_cmd"
      fi
      job_index=$(( job_index + 1 ))
    done
for eps2 in $(seq 0.03 0.01 0.1); do
      # 根据 job_index 进行轮询分配：取模 2 得到 gpu 编号（0 或 1）
      cuda_number=$(( job_index % 2 ))
      # 拼接完整的命令，根据原命令模板
      cmd="python evaluation.py --expert_recording --adv_algo PPO_FGSM --addition_msg TrafficEnv3-v0_TD3_NoExpert_eps0.10 --attack_eps ${eps2} --algo TD3 --env_name TrafficEnv3-v0 --train_step 100 --attack --adv_steps 9 --cuda_number ${cuda_number} --attack_method fgsm --best_model"
      # 分发到对应 GPU 的命令文件中
      if [ $cuda_number -eq 0 ]; then
         echo "$cmd" >> "$gpu0_cmd"
      else
         echo "$cmd" >> "$gpu1_cmd"
      fi
      job_index=$(( job_index + 1 ))
    done
for eps2 in $(seq 0.03 0.01 0.1); do
      # 根据 job_index 进行轮询分配：取模 2 得到 gpu 编号（0 或 1）
      cuda_number=$(( job_index % 2 ))
      # 拼接完整的命令，根据原命令模板
      cmd="python evaluation.py --expert_recording --adv_algo PPO_FGSM --addition_msg NoExpert_eps0.09 --attack_eps ${eps2} --algo PPO --env_name TrafficEnv8-v0 --train_step 100 --attack --adv_steps 9 --cuda_number ${cuda_number} --attack_method fgsm --best_model"
      # 分发到对应 GPU 的命令文件中
      if [ $cuda_number -eq 0 ]; then
         echo "$cmd" >> "$gpu0_cmd"
      else
         echo "$cmd" >> "$gpu1_cmd"
      fi
      job_index=$(( job_index + 1 ))
    done
for eps2 in $(seq 0.03 0.01 0.1); do
      # 根据 job_index 进行轮询分配：取模 2 得到 gpu 编号（0 或 1）
      cuda_number=$(( job_index % 2 ))
      # 拼接完整的命令，根据原命令模板
      cmd="python evaluation.py --expert_recording --adv_algo PPO_FGSM --addition_msg TrafficEnv8-v0_SAC_NoExpert_eps0.04 --attack_eps ${eps2} --algo SAC --env_name TrafficEnv8-v0 --train_step 100 --attack --adv_steps 9 --cuda_number ${cuda_number} --attack_method fgsm --best_model"
      # 分发到对应 GPU 的命令文件中
      if [ $cuda_number -eq 0 ]; then
         echo "$cmd" >> "$gpu0_cmd"
      else
         echo "$cmd" >> "$gpu1_cmd"
      fi
      job_index=$(( job_index + 1 ))
    done
for eps2 in $(seq 0.03 0.01 0.1); do
    # 根据 job_index 进行轮询分配：取模 2 得到 gpu 编号（0 或 1）
    cuda_number=$(( job_index % 2 ))
    # 拼接完整的命令，根据原命令模板
    cmd="python evaluation.py --expert_recording --adv_algo PPO_FGSM --addition_msg TrafficEnv8-v0_TD3_NoExpert_eps0.07 --attack_eps ${eps2} --algo TD3 --env_name TrafficEnv8-v0 --train_step 100 --attack --adv_steps 9 --cuda_number ${cuda_number} --attack_method fgsm --best_model"
    # 分发到对应 GPU 的命令文件中
    if [ $cuda_number -eq 0 ]; then
       echo "$cmd" >> "$gpu0_cmd"
    else
       echo "$cmd" >> "$gpu1_cmd"
    fi
    job_index=$(( job_index + 1 ))
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