#!/bin/bash

# 生成所有命令，保存到数组
model_paths=()
for ratio in 0.5 0.4 0.3 0.2 0.1 0.05 0.01; do
	    model_paths+=("expert_model/MoEs/nf_nbal_${ratio}")
	        model_paths+=("expert_model/MoEs/nf_${ratio}")
		    model_paths+=("expert_model/MoEs/f_nbal_${ratio}")
		        model_paths+=("expert_model/MoEs/f_${ratio}")
		done

		envs=("TrafficEnv3-v5" "TrafficEnv8-v1")
		algos=("PPO" "SAC" "TD3")
		epsilons=(0.05 0.06 0.07 0.08 0.09 0.10)

		# 函数定义
		run_cmd() {
			    local model_path="$1"
			        local env="$2"
				    local algo="$3"
				        local eps="$4"
					    local job_index="$5"

					        local cuda_number=$(( job_index % 2 ))
						    local addition_msg="vec_beta_ens2_${env}_${algo}_Vanilla_eps${eps}_0"
						        local log_file="logs/test_log/${env}_${algo}_eps${eps}_0.log"

							    python evaluation_v3.py \
								            --algo "$algo" \
									            --env_name "$env" \
										            --expert_attack \
											            --expert_model_path "$model_path" \
												            --result_saving \
													            --result_filename MoE_test_all_1 \
														            --train_step 100 \
															            --attack_eps "$eps" \
																            --cuda_number "$cuda_number" \
																	            --attack \
																		            --best_model \
																			            --addition_msg "$addition_msg" > "$log_file" 2>&1
																			    }

																		    export -f run_cmd

																		    # 使用 parallel 并行执行
																		    job_index=0
																		    parallel --jobs 4 '
																		        run_cmd "{1}" "{2}" "{3}" "{4}" "'"$job_index"'"
																			' ::: "${model_paths[@]}" ::: "${envs[@]}" ::: "${algos[@]}" ::: "${epsilons[@]}"
