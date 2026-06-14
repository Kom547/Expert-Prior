# EXP002: Demo Source Agnosticism Commands

# === SOURCE: vanilla_14_relaxed ===
# Run ID: EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed0_gen
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --seed 0 --expert_recording --attack_eps 0.1 --adv_steps 14 --expert_data_path EXP002_vanilla_14_relaxed_0 --run_id EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed0_gen --experiment_id EXP002

# Run ID: EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed0_train_moe
python expert_imitation_learning_MoE.py --expert_data_path expert_data/EXP002_vanilla_14_relaxed_0 --expert_model_savepath expert_model/EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed0 --seed 0 --run_id EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed0_train_moe --experiment_id EXP002

# Run ID: EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed0_train_aegrl
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 0 --expert_attack --expert_model_path expert_model/EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed0 --expert_prior ValuePenalty --addition_msg source_vanilla_14_relaxed --run_id EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed0_train_aegrl --experiment_id EXP002

# Run ID: EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed0_eval_aegrl
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 0 --expert_attack --expert_model_path expert_model/EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed0 --expert_prior ValuePenalty --addition_msg source_vanilla_14_relaxed --result_saving --result_filename EXP002_agnosticism --run_id EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed0_eval_aegrl --experiment_id EXP002

# Run ID: EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed10_gen
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --seed 10 --expert_recording --attack_eps 0.1 --adv_steps 14 --expert_data_path EXP002_vanilla_14_relaxed_10 --run_id EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed10_gen --experiment_id EXP002

# Run ID: EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed10_train_moe
python expert_imitation_learning_MoE.py --expert_data_path expert_data/EXP002_vanilla_14_relaxed_10 --expert_model_savepath expert_model/EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed10 --seed 10 --run_id EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed10_train_moe --experiment_id EXP002

# Run ID: EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed10_train_aegrl
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 10 --expert_attack --expert_model_path expert_model/EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed10 --expert_prior ValuePenalty --addition_msg source_vanilla_14_relaxed --run_id EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed10_train_aegrl --experiment_id EXP002

# Run ID: EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed10_eval_aegrl
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 10 --expert_attack --expert_model_path expert_model/EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed10 --expert_prior ValuePenalty --addition_msg source_vanilla_14_relaxed --result_saving --result_filename EXP002_agnosticism --run_id EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed10_eval_aegrl --experiment_id EXP002

# Run ID: EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed20_gen
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --seed 20 --expert_recording --attack_eps 0.1 --adv_steps 14 --expert_data_path EXP002_vanilla_14_relaxed_20 --run_id EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed20_gen --experiment_id EXP002

# Run ID: EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed20_train_moe
python expert_imitation_learning_MoE.py --expert_data_path expert_data/EXP002_vanilla_14_relaxed_20 --expert_model_savepath expert_model/EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed20 --seed 20 --run_id EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed20_train_moe --experiment_id EXP002

# Run ID: EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed20_train_aegrl
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 20 --expert_attack --expert_model_path expert_model/EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed20 --expert_prior ValuePenalty --addition_msg source_vanilla_14_relaxed --run_id EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed20_train_aegrl --experiment_id EXP002

# Run ID: EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed20_eval_aegrl
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 20 --expert_attack --expert_model_path expert_model/EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed20 --expert_prior ValuePenalty --addition_msg source_vanilla_14_relaxed --result_saving --result_filename EXP002_agnosticism --run_id EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed20_eval_aegrl --experiment_id EXP002

# Run ID: EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed30_gen
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --seed 30 --expert_recording --attack_eps 0.1 --adv_steps 14 --expert_data_path EXP002_vanilla_14_relaxed_30 --run_id EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed30_gen --experiment_id EXP002

# Run ID: EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed30_train_moe
python expert_imitation_learning_MoE.py --expert_data_path expert_data/EXP002_vanilla_14_relaxed_30 --expert_model_savepath expert_model/EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed30 --seed 30 --run_id EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed30_train_moe --experiment_id EXP002

# Run ID: EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed30_train_aegrl
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 30 --expert_attack --expert_model_path expert_model/EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed30 --expert_prior ValuePenalty --addition_msg source_vanilla_14_relaxed --run_id EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed30_train_aegrl --experiment_id EXP002

# Run ID: EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed30_eval_aegrl
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 30 --expert_attack --expert_model_path expert_model/EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed30 --expert_prior ValuePenalty --addition_msg source_vanilla_14_relaxed --result_saving --result_filename EXP002_agnosticism --run_id EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed30_eval_aegrl --experiment_id EXP002

# Run ID: EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed40_gen
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --seed 40 --expert_recording --attack_eps 0.1 --adv_steps 14 --expert_data_path EXP002_vanilla_14_relaxed_40 --run_id EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed40_gen --experiment_id EXP002

# Run ID: EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed40_train_moe
python expert_imitation_learning_MoE.py --expert_data_path expert_data/EXP002_vanilla_14_relaxed_40 --expert_model_savepath expert_model/EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed40 --seed 40 --run_id EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed40_train_moe --experiment_id EXP002

# Run ID: EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed40_train_aegrl
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 40 --expert_attack --expert_model_path expert_model/EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed40 --expert_prior ValuePenalty --addition_msg source_vanilla_14_relaxed --run_id EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed40_train_aegrl --experiment_id EXP002

# Run ID: EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed40_eval_aegrl
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 40 --expert_attack --expert_model_path expert_model/EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed40 --expert_prior ValuePenalty --addition_msg source_vanilla_14_relaxed --result_saving --result_filename EXP002_agnosticism --run_id EXP002_vanilla_14_relaxed_TrafficEnv3-v5_PPO_seed40_eval_aegrl --experiment_id EXP002

# === SOURCE: heuristic_random_time_direction ===
# Run ID: EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed0_gen
python review_response/scripts/generate_heuristic_demo_source.py --env_name TrafficEnv3-v5 --algo PPO --seed 0 --out_dir expert_data/EXP002_heuristic_random_time_direction_0 --run_id EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed0_gen --experiment_id EXP002

# Run ID: EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed0_train_moe
python expert_imitation_learning_MoE.py --expert_data_path expert_data/EXP002_heuristic_random_time_direction_0 --expert_model_savepath expert_model/EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed0 --seed 0 --run_id EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed0_train_moe --experiment_id EXP002

# Run ID: EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed0_train_aegrl
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 0 --expert_attack --expert_model_path expert_model/EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed0 --expert_prior ValuePenalty --addition_msg source_heuristic_random_time_direction --run_id EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed0_train_aegrl --experiment_id EXP002

# Run ID: EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed0_eval_aegrl
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 0 --expert_attack --expert_model_path expert_model/EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed0 --expert_prior ValuePenalty --addition_msg source_heuristic_random_time_direction --result_saving --result_filename EXP002_agnosticism --run_id EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed0_eval_aegrl --experiment_id EXP002

# Run ID: EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed10_gen
python review_response/scripts/generate_heuristic_demo_source.py --env_name TrafficEnv3-v5 --algo PPO --seed 10 --out_dir expert_data/EXP002_heuristic_random_time_direction_10 --run_id EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed10_gen --experiment_id EXP002

# Run ID: EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed10_train_moe
python expert_imitation_learning_MoE.py --expert_data_path expert_data/EXP002_heuristic_random_time_direction_10 --expert_model_savepath expert_model/EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed10 --seed 10 --run_id EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed10_train_moe --experiment_id EXP002

# Run ID: EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed10_train_aegrl
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 10 --expert_attack --expert_model_path expert_model/EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed10 --expert_prior ValuePenalty --addition_msg source_heuristic_random_time_direction --run_id EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed10_train_aegrl --experiment_id EXP002

# Run ID: EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed10_eval_aegrl
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 10 --expert_attack --expert_model_path expert_model/EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed10 --expert_prior ValuePenalty --addition_msg source_heuristic_random_time_direction --result_saving --result_filename EXP002_agnosticism --run_id EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed10_eval_aegrl --experiment_id EXP002

# Run ID: EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed20_gen
python review_response/scripts/generate_heuristic_demo_source.py --env_name TrafficEnv3-v5 --algo PPO --seed 20 --out_dir expert_data/EXP002_heuristic_random_time_direction_20 --run_id EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed20_gen --experiment_id EXP002

# Run ID: EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed20_train_moe
python expert_imitation_learning_MoE.py --expert_data_path expert_data/EXP002_heuristic_random_time_direction_20 --expert_model_savepath expert_model/EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed20 --seed 20 --run_id EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed20_train_moe --experiment_id EXP002

# Run ID: EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed20_train_aegrl
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 20 --expert_attack --expert_model_path expert_model/EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed20 --expert_prior ValuePenalty --addition_msg source_heuristic_random_time_direction --run_id EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed20_train_aegrl --experiment_id EXP002

# Run ID: EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed20_eval_aegrl
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 20 --expert_attack --expert_model_path expert_model/EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed20 --expert_prior ValuePenalty --addition_msg source_heuristic_random_time_direction --result_saving --result_filename EXP002_agnosticism --run_id EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed20_eval_aegrl --experiment_id EXP002

# Run ID: EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed30_gen
python review_response/scripts/generate_heuristic_demo_source.py --env_name TrafficEnv3-v5 --algo PPO --seed 30 --out_dir expert_data/EXP002_heuristic_random_time_direction_30 --run_id EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed30_gen --experiment_id EXP002

# Run ID: EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed30_train_moe
python expert_imitation_learning_MoE.py --expert_data_path expert_data/EXP002_heuristic_random_time_direction_30 --expert_model_savepath expert_model/EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed30 --seed 30 --run_id EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed30_train_moe --experiment_id EXP002

# Run ID: EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed30_train_aegrl
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 30 --expert_attack --expert_model_path expert_model/EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed30 --expert_prior ValuePenalty --addition_msg source_heuristic_random_time_direction --run_id EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed30_train_aegrl --experiment_id EXP002

# Run ID: EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed30_eval_aegrl
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 30 --expert_attack --expert_model_path expert_model/EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed30 --expert_prior ValuePenalty --addition_msg source_heuristic_random_time_direction --result_saving --result_filename EXP002_agnosticism --run_id EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed30_eval_aegrl --experiment_id EXP002

# Run ID: EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed40_gen
python review_response/scripts/generate_heuristic_demo_source.py --env_name TrafficEnv3-v5 --algo PPO --seed 40 --out_dir expert_data/EXP002_heuristic_random_time_direction_40 --run_id EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed40_gen --experiment_id EXP002

# Run ID: EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed40_train_moe
python expert_imitation_learning_MoE.py --expert_data_path expert_data/EXP002_heuristic_random_time_direction_40 --expert_model_savepath expert_model/EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed40 --seed 40 --run_id EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed40_train_moe --experiment_id EXP002

# Run ID: EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed40_train_aegrl
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 40 --expert_attack --expert_model_path expert_model/EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed40 --expert_prior ValuePenalty --addition_msg source_heuristic_random_time_direction --run_id EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed40_train_aegrl --experiment_id EXP002

# Run ID: EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed40_eval_aegrl
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 40 --expert_attack --expert_model_path expert_model/EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed40 --expert_prior ValuePenalty --addition_msg source_heuristic_random_time_direction --result_saving --result_filename EXP002_agnosticism --run_id EXP002_heuristic_random_time_direction_TrafficEnv3-v5_PPO_seed40_eval_aegrl --experiment_id EXP002

