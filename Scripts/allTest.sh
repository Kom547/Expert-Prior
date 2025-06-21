#!/bin/bash
# 删除可能已有的命令列表文件，重新生成
gpu0_cmd="gpu0_cmds1.txt"
gpu1_cmd="gpu1_cmds1.txt"
rm -f "$gpu0_cmd" "$gpu1_cmd"
# 创建logs/test_log目录，如果不存在的话
mkdir -p logs/test_log
job_index=0

for env in TrafficEnv3-v5 TrafficEnv8-v1; do
  for eprior in ValuePenalty PolicyConstrained; do
    for algo in PPO SAC TD3; do
      for eps in $(seq 0.05 0.01 0.1); do
        for seed in $(seq 0 1 4); do
        addition_msg="vec_beta_ens2_${env}_${algo}_${eprior}_eps${eps}_${seed}"
        if [ $env == TrafficEnv3-v5 ] && [ $algo == TD3 ]; then
          addition_msg="fix_vec_beta_ens2_${env}_${algo}_${eprior}_eps${eps}_${seed}"
        fi
        #cuda_number=$(( job_index % 2 ))
        cuda_number=1
        cmd="python evaluation_v3.py --result_saving --result_filename AttackEvaluation_no_random_v3_fix.csv --algo ${algo} --env_name ${env} --expert_prior ${eprior} --train_step 100 --attack_eps ${eps} --cuda_number ${cuda_number} --attack --best_model --addition_msg ${addition_msg}"
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

for env in TrafficEnv3-v5 TrafficEnv8-v1; do
    for algo in PPO SAC TD3; do
      for eps in $(seq 0.05 0.01 0.1); do
        for seed in $(seq 0 1 4); do
        addition_msg="vec_beta_ens2_${env}_${algo}_Vanilla_eps${eps}_${seed}"
        if [ $env == TrafficEnv3-v5 ] && [ $algo == TD3 ]; then
          addition_msg="fix_vec_beta_ens2_${env}_${algo}_Vanilla_eps${eps}_${seed}"
        fi
        #cuda_number=$(( job_index % 2 ))
        cuda_number=1
        cmd="python evaluation_v3.py --result_saving --result_filename AttackEvaluation_no_random_v3_fix.csv --algo ${algo} --env_name ${env} --train_step 100 --attack_eps ${eps} --attack --cuda_number ${cuda_number} --best_model --addition_msg ${addition_msg}"
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


echo "共生成 $job_index 条命令！"
echo "GPU0 的任务数：$(wc -l < "$gpu0_cmd")"
echo "GPU1 的任务数：$(wc -l < "$gpu1_cmd")"

# 分别启动两个 GPU 的任务，每个 GPU 最多并行运行 10 个任务
echo "启动 GPU0 上的任务..."
( cat "$gpu0_cmd" | xargs -I {} -P 5 bash -c '{}' ) &

echo "启动 GPU1 上的任务..."
( cat "$gpu1_cmd" | xargs -I {} -P 5 bash -c '{}' ) &

wait
echo "所有实验任务已完成."
