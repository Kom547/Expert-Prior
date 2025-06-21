#!/bin/bash
# 删除可能已有的命令列表文件，重新生成
gpu0_cmd="gpu0_cmds.txt"
gpu1_cmd="gpu1_cmds.txt"
rm -f "$gpu0_cmd" "$gpu1_cmd"
# 创建logs/test_log目录，如果不存在的话
mkdir -p logs/test_log
job_index=0

# 遍历所有组合：algo（SAC, TD3）、env_name（TrafficEnv3-v0, TrafficEnv8-v0）、attack_eps（0.01 ~ 0.1）
for env in TrafficEnv3-v5 TrafficEnv8-v1; do
#env=TrafficEnv3-v5
  for eprior in ValuePenalty PolicyConstrained; do
    for algo in PPO SAC; do
      if [ "$env" = "TrafficEnv3-v5" ] && [ "$algo" = "PPO" ]; then
        continue
      fi
      # 可以额外提取 env 的前缀作为 addition_msg 的一部分（如去掉 -v0），这里暂直接使用 env
      for eps in $(seq 0.05 0.01 0.1); do
        for seed in $(seq 0 1 4); do
        # 定义 addition_msg，比如 "TrafficEnv3-v0_NoExpert_eps0.02"
        addition_msg="vec_beta_ens2_${env}_${algo}_${eprior}_eps${eps}_${seed}"
        # 根据 job_index 进行轮询分配：取模 2 得到 gpu 编号（0 或 1）
        cuda_number=$(( job_index % 2 ))
        #cuda_number=0
        # 拼接完整的命令，根据原命令模板
        #python st_attack_fgsm_exp_alternation.py --env_name TrafficEnv3-v0 --algo PPO --attack_eps 0.05 --attack --adv_steps 9 --addition_msg beta_env3_PPO_0.05_advsteps9_VP_250 --train_step 250 --expert_attack --expert_model_path /data/lxy/STA-Expert/expert_model/ensemblev2_2 --adv_algo PPO_FGSM --expert_prior ValuePenalty
        cmd="python evaluation.py --algo ${algo} --env_name ${env} --expert_prior ${eprior} --train_step 100 --attack_eps ${eps} --random_seed --attack --best_model --addition_msg ${addition_msg}"
        #cmd="python st_attack_fgsm.py --adv_algo PPO_FGSM --addition_msg ${addition_msg} --attack_eps ${eps} --algo ${algo} --env_name ${env} --train_step 250 --attack --adv_steps 9 --n_steps 512 --cuda_number ${cuda_number} --attack_method fgsm"
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

echo "共生成 $job_index 条命令！"
echo "GPU0 的任务数：$(wc -l < "$gpu0_cmd")"
echo "GPU1 的任务数：$(wc -l < "$gpu1_cmd")"

# 分别启动两个 GPU 的任务，每个 GPU 最多并行运行 10 个任务
echo "启动 GPU0 上的任务..."
( cat "$gpu0_cmd" | xargs -I {} -P 20 bash -c '{}' ) &

echo "启动 GPU1 上的任务..."
( cat "$gpu1_cmd" | xargs -I {} -P 20 bash -c '{}' ) &

# 等待所有后台任务运行完毕
wait
echo "所有实验任务已完成."
#python evaluation.py  --algo PPO --env_name TrafficEnv8-v1 --train_step 100 --random_seed --attack --best_model --addition_msg vec_beta_ens2_TrafficEnv8-v1_PPO_ValuePenalty_eps0.09_4