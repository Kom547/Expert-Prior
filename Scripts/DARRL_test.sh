#!/bin/bash
# 删除可能已有的命令列表文件，重新生成
gpu0_cmd="gpu0_cmds.txt"
gpu1_cmd="gpu1_cmds.txt"
rm -f "$gpu0_cmd" "$gpu1_cmd"
# 创建logs/test_log目录，如果不存在的话
mkdir -p logs/test_log
job_index=0
env=TrafficEnv3-v5
algo=DARRL
eps=0.03
for seed in $(seq 0 1 4); do
        addition_msg="DARRL_23_Vanilla_0.03_${seed}"
        cuda_number=$(( job_index % 2 ))
        #cuda_number=0
        #cmd="python advTrain_tensorborad.py --no_wandb --cuda_seed ${seed} --train_step 200 --addition_msg ${addition_msg} --num_envs 8 --attack --env_name TrafficEnv3-v5 --algo DARRL --attack_eps 0.03 --cuda_number 0 --fni_model_path policy_v277_20250604_23weights"
        cmd="python evaluation_v3.py --result_saving --result_filename AttackEvaluation_DARRL --algo ${algo} --env_name ${env} --train_step 100 --attack_eps ${eps} --attack --cuda_number ${cuda_number} --best_model --addition_msg ${addition_msg} --fni_model_path policy_v277_20250604_23weights"

        log_file="logs/test_log/${addition_msg}.log"
        if [ $cuda_number -eq 0 ]; then
            echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu0_cmd"
        else
            echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu1_cmd"
        fi
        job_index=$(( job_index + 1 ))
        done

for seed in $(seq 0 1 4); do
addition_msg="DARRL_18_Vanilla_0.03_${seed}"
cuda_number=$(( job_index % 2 ))
#cuda_number=0
#cmd="python advTrain_tensorborad.py --no_wandb --cuda_seed ${seed} --train_step 200 --addition_msg ${addition_msg} --num_envs 8 --attack --env_name TrafficEnv3-v5 --algo DARRL --attack_eps 0.03 --cuda_number 0 --fni_model_path policy_v272_20250604_18weights"
cmd="python evaluation_v3.py --result_saving --result_filename AttackEvaluation_DARRL --algo ${algo} --env_name ${env} --train_step 100 --attack_eps ${eps} --attack --cuda_number ${cuda_number} --best_model --addition_msg ${addition_msg} --fni_model_path policy_v272_20250604_18weights"

log_file="logs/test_log/${addition_msg}.log"
if [ $cuda_number -eq 0 ]; then
    echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu0_cmd"
else
    echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu1_cmd"
fi
job_index=$(( job_index + 1 ))
done

for eprior in ValuePenalty PolicyConstrained; do
for seed in $(seq 0 1 4); do
        addition_msg="DARRL_23_${eprior}_0.03_${seed}"
        cuda_number=$(( job_index % 2 ))
        #cuda_number=0
        #cmd="python advTrain_tensorborad.py --expert_model_path /data/lxy/EPRL/expert_model/ensemblev2_2 --expert_attack --expert_prior ${eprior} --no_wandb --cuda_seed ${seed} --train_step 200 --addition_msg ${addition_msg} --num_envs 8 --attack --env_name TrafficEnv3-v5 --algo DARRL --attack_eps 0.03 --cuda_number 0 --fni_model_path policy_v277_20250604_23weights"
        cmd="python evaluation_v3.py --result_saving --result_filename AttackEvaluation_DARRL --algo ${algo} --env_name ${env} --train_step 100 --attack_eps ${eps} --attack --cuda_number ${cuda_number} --best_model --addition_msg ${addition_msg} --fni_model_path policy_v277_20250604_23weights"
        log_file="logs/test_log/${addition_msg}.log"
        if [ $cuda_number -eq 0 ]; then
            echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu0_cmd"
        else
            echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu1_cmd"
        fi
        job_index=$(( job_index + 1 ))
        done
        done
for eprior in ValuePenalty PolicyConstrained; do
for seed in $(seq 0 1 4); do
addition_msg="DARRL_18_${eprior}_0.03_${seed}"
cuda_number=$(( job_index % 2 ))
#cuda_number=0
#cmd="python advTrain_tensorborad.py --expert_model_path /data/lxy/EPRL/expert_model/ensemblev2_2 --expert_attack --expert_prior ${eprior} --no_wandb --cuda_seed ${seed} --train_step 200 --addition_msg ${addition_msg} --num_envs 8 --attack --env_name TrafficEnv3-v5 --algo DARRL --attack_eps 0.03 --cuda_number 0 --fni_model_path policy_v272_20250604_18weights"
cmd="python evaluation_v3.py --result_saving --result_filename AttackEvaluation_DARRL --algo ${algo} --env_name ${env} --train_step 100 --attack_eps ${eps} --attack --cuda_number ${cuda_number} --best_model --addition_msg ${addition_msg} --fni_model_path policy_v272_20250604_18weights"

log_file="logs/test_log/${addition_msg}.log"
if [ $cuda_number -eq 0 ]; then
    echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu0_cmd"
else
    echo "$cmd > $log_file 2>&1 && echo 'Job finished: $addition_msg' || echo 'Job failed: $addition_msg'" >> "$gpu1_cmd"
fi
job_index=$(( job_index + 1 ))
done
done

echo "共生成 $job_index 条命令！"
echo "GPU0 的任务数：$(wc -l < "$gpu0_cmd")"
echo "GPU1 的任务数：$(wc -l < "$gpu1_cmd")"

# 分别启动两个 GPU 的任务，每个 GPU 最多并行运行 10 个任务
echo "启动 GPU0 上的任务..."
( cat "$gpu0_cmd" | xargs -I {} -P 2 bash -c '{}' ) &

echo "启动 GPU1 上的任务..."
( cat "$gpu1_cmd" | xargs -I {} -P 2 bash -c '{}' ) &

# 等待所有后台任务运行完毕
wait
echo "所有实验任务已完成."