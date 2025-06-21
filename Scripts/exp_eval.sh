# 删除可能已有的命令列表文件，重新生成
gpu0_cmd="gpu0_cmds.txt"
gpu1_cmd="gpu1_cmds.txt"
rm -f "$gpu0_cmd" "$gpu1_cmd"

job_index=0

# 遍历所有组合：algo（SAC, TD3）、env_name（TrafficEnv3-v0, TrafficEnv8-v0）、attack_eps（0.01 ~ 0.1）
for algo in PPO SAC TD3; do
  for env in TrafficEnv3-v0 TrafficEnv8-v0; do
    # 可以额外提取 env 的前缀作为 addition_msg 的一部分（如去掉 -v0），这里暂直接使用 env
    for eps2 in $(seq 0.01 0.01 0.1); do
      # 根据 job_index 进行轮询分配：取模 2 得到 gpu 编号（0 或 1）
      cuda_number=$(( job_index % 2 ))
      # 拼接完整的命令，根据原命令模板
      cmd="python evaluation_exp.py --expert_model_path /data/lxy/STA-Expert/expert_model/ensemblev2_2 --env_name ${env} --train_step 100 --attack_eps ${eps2} --algo ${algo}"
      # 分发到对应 GPU 的命令文件中
      if [ $cuda_number -eq 0 ]; then
         echo "$cmd" >> "$gpu0_cmd"
      else
         echo "$cmd" >> "$gpu1_cmd"
      fi
      job_index=$(( job_index + 1 ))
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
st_attack_fgsm_exp_alternation.py --env_name TrafficEnv8-v0 --algo PPO --attack_eps 0.03 --attack --adv_steps 9 --addition_msg beta_env8_PPO_0.03_advsteps9_VP_250 --train_step 250 --expert_attack --expert_model_path /data/lxy/STA-Expert/expert_model/ensemblev2_2 --adv_algo PPO_FGSM --expert_prior ValuePenalty