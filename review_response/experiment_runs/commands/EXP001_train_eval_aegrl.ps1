# EXP001 Phase 3: AEGRL Train/Eval
# Run ID: EXP001_aegrl_src_PPO_tgt_PPO_seed0_train
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 0 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed0 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_PPO --run_id EXP001_aegrl_src_PPO_tgt_PPO_seed0_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_PPO_seed0_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 0 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed0 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_PPO --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_PPO_tgt_PPO_seed0_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_PPO_seed10_train
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 10 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed10 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_PPO --run_id EXP001_aegrl_src_PPO_tgt_PPO_seed10_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_PPO_seed10_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 10 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed10 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_PPO --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_PPO_tgt_PPO_seed10_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_PPO_seed20_train
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 20 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed20 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_PPO --run_id EXP001_aegrl_src_PPO_tgt_PPO_seed20_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_PPO_seed20_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 20 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed20 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_PPO --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_PPO_tgt_PPO_seed20_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_PPO_seed30_train
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 30 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed30 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_PPO --run_id EXP001_aegrl_src_PPO_tgt_PPO_seed30_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_PPO_seed30_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 30 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed30 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_PPO --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_PPO_tgt_PPO_seed30_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_PPO_seed40_train
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 40 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed40 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_PPO --run_id EXP001_aegrl_src_PPO_tgt_PPO_seed40_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_PPO_seed40_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 40 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed40 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_PPO --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_PPO_tgt_PPO_seed40_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_SAC_seed0_train
python advTrain.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 0 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed0 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_SAC --run_id EXP001_aegrl_src_PPO_tgt_SAC_seed0_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_SAC_seed0_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 0 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed0 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_SAC --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_PPO_tgt_SAC_seed0_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_SAC_seed10_train
python advTrain.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 10 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed10 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_SAC --run_id EXP001_aegrl_src_PPO_tgt_SAC_seed10_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_SAC_seed10_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 10 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed10 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_SAC --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_PPO_tgt_SAC_seed10_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_SAC_seed20_train
python advTrain.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 20 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed20 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_SAC --run_id EXP001_aegrl_src_PPO_tgt_SAC_seed20_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_SAC_seed20_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 20 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed20 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_SAC --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_PPO_tgt_SAC_seed20_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_SAC_seed30_train
python advTrain.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 30 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed30 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_SAC --run_id EXP001_aegrl_src_PPO_tgt_SAC_seed30_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_SAC_seed30_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 30 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed30 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_SAC --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_PPO_tgt_SAC_seed30_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_SAC_seed40_train
python advTrain.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 40 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed40 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_SAC --run_id EXP001_aegrl_src_PPO_tgt_SAC_seed40_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_SAC_seed40_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 40 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed40 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_SAC --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_PPO_tgt_SAC_seed40_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_TD3_seed0_train
python advTrain.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 0 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed0 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_TD3 --run_id EXP001_aegrl_src_PPO_tgt_TD3_seed0_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_TD3_seed0_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 0 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed0 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_TD3 --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_PPO_tgt_TD3_seed0_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_TD3_seed10_train
python advTrain.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 10 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed10 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_TD3 --run_id EXP001_aegrl_src_PPO_tgt_TD3_seed10_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_TD3_seed10_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 10 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed10 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_TD3 --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_PPO_tgt_TD3_seed10_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_TD3_seed20_train
python advTrain.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 20 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed20 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_TD3 --run_id EXP001_aegrl_src_PPO_tgt_TD3_seed20_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_TD3_seed20_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 20 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed20 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_TD3 --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_PPO_tgt_TD3_seed20_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_TD3_seed30_train
python advTrain.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 30 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed30 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_TD3 --run_id EXP001_aegrl_src_PPO_tgt_TD3_seed30_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_TD3_seed30_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 30 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed30 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_TD3 --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_PPO_tgt_TD3_seed30_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_TD3_seed40_train
python advTrain.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 40 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed40 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_TD3 --run_id EXP001_aegrl_src_PPO_tgt_TD3_seed40_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_PPO_tgt_TD3_seed40_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 40 --expert_attack --expert_model_path expert_model/EXP001_PPO_seed40 --expert_prior ValuePenalty --addition_msg src_PPO_tgt_TD3 --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_PPO_tgt_TD3_seed40_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_PPO_seed0_train
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 0 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed0 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_PPO --run_id EXP001_aegrl_src_SAC_tgt_PPO_seed0_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_PPO_seed0_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 0 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed0 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_PPO --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_SAC_tgt_PPO_seed0_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_PPO_seed10_train
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 10 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed10 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_PPO --run_id EXP001_aegrl_src_SAC_tgt_PPO_seed10_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_PPO_seed10_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 10 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed10 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_PPO --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_SAC_tgt_PPO_seed10_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_PPO_seed20_train
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 20 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed20 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_PPO --run_id EXP001_aegrl_src_SAC_tgt_PPO_seed20_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_PPO_seed20_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 20 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed20 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_PPO --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_SAC_tgt_PPO_seed20_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_PPO_seed30_train
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 30 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed30 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_PPO --run_id EXP001_aegrl_src_SAC_tgt_PPO_seed30_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_PPO_seed30_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 30 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed30 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_PPO --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_SAC_tgt_PPO_seed30_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_PPO_seed40_train
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 40 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed40 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_PPO --run_id EXP001_aegrl_src_SAC_tgt_PPO_seed40_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_PPO_seed40_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 40 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed40 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_PPO --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_SAC_tgt_PPO_seed40_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_SAC_seed0_train
python advTrain.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 0 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed0 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_SAC --run_id EXP001_aegrl_src_SAC_tgt_SAC_seed0_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_SAC_seed0_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 0 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed0 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_SAC --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_SAC_tgt_SAC_seed0_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_SAC_seed10_train
python advTrain.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 10 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed10 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_SAC --run_id EXP001_aegrl_src_SAC_tgt_SAC_seed10_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_SAC_seed10_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 10 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed10 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_SAC --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_SAC_tgt_SAC_seed10_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_SAC_seed20_train
python advTrain.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 20 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed20 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_SAC --run_id EXP001_aegrl_src_SAC_tgt_SAC_seed20_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_SAC_seed20_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 20 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed20 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_SAC --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_SAC_tgt_SAC_seed20_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_SAC_seed30_train
python advTrain.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 30 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed30 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_SAC --run_id EXP001_aegrl_src_SAC_tgt_SAC_seed30_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_SAC_seed30_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 30 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed30 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_SAC --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_SAC_tgt_SAC_seed30_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_SAC_seed40_train
python advTrain.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 40 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed40 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_SAC --run_id EXP001_aegrl_src_SAC_tgt_SAC_seed40_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_SAC_seed40_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 40 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed40 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_SAC --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_SAC_tgt_SAC_seed40_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_TD3_seed0_train
python advTrain.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 0 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed0 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_TD3 --run_id EXP001_aegrl_src_SAC_tgt_TD3_seed0_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_TD3_seed0_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 0 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed0 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_TD3 --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_SAC_tgt_TD3_seed0_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_TD3_seed10_train
python advTrain.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 10 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed10 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_TD3 --run_id EXP001_aegrl_src_SAC_tgt_TD3_seed10_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_TD3_seed10_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 10 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed10 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_TD3 --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_SAC_tgt_TD3_seed10_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_TD3_seed20_train
python advTrain.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 20 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed20 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_TD3 --run_id EXP001_aegrl_src_SAC_tgt_TD3_seed20_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_TD3_seed20_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 20 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed20 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_TD3 --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_SAC_tgt_TD3_seed20_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_TD3_seed30_train
python advTrain.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 30 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed30 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_TD3 --run_id EXP001_aegrl_src_SAC_tgt_TD3_seed30_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_TD3_seed30_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 30 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed30 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_TD3 --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_SAC_tgt_TD3_seed30_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_TD3_seed40_train
python advTrain.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 40 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed40 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_TD3 --run_id EXP001_aegrl_src_SAC_tgt_TD3_seed40_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_SAC_tgt_TD3_seed40_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 40 --expert_attack --expert_model_path expert_model/EXP001_SAC_seed40 --expert_prior ValuePenalty --addition_msg src_SAC_tgt_TD3 --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_SAC_tgt_TD3_seed40_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_PPO_seed0_train
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 0 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed0 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_PPO --run_id EXP001_aegrl_src_TD3_tgt_PPO_seed0_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_PPO_seed0_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 0 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed0 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_PPO --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_TD3_tgt_PPO_seed0_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_PPO_seed10_train
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 10 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed10 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_PPO --run_id EXP001_aegrl_src_TD3_tgt_PPO_seed10_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_PPO_seed10_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 10 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed10 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_PPO --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_TD3_tgt_PPO_seed10_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_PPO_seed20_train
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 20 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed20 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_PPO --run_id EXP001_aegrl_src_TD3_tgt_PPO_seed20_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_PPO_seed20_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 20 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed20 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_PPO --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_TD3_tgt_PPO_seed20_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_PPO_seed30_train
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 30 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed30 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_PPO --run_id EXP001_aegrl_src_TD3_tgt_PPO_seed30_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_PPO_seed30_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 30 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed30 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_PPO --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_TD3_tgt_PPO_seed30_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_PPO_seed40_train
python advTrain.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 40 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed40 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_PPO --run_id EXP001_aegrl_src_TD3_tgt_PPO_seed40_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_PPO_seed40_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo PPO --adv_algo PPO_FGSM --seed 40 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed40 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_PPO --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_TD3_tgt_PPO_seed40_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_SAC_seed0_train
python advTrain.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 0 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed0 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_SAC --run_id EXP001_aegrl_src_TD3_tgt_SAC_seed0_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_SAC_seed0_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 0 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed0 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_SAC --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_TD3_tgt_SAC_seed0_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_SAC_seed10_train
python advTrain.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 10 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed10 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_SAC --run_id EXP001_aegrl_src_TD3_tgt_SAC_seed10_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_SAC_seed10_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 10 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed10 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_SAC --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_TD3_tgt_SAC_seed10_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_SAC_seed20_train
python advTrain.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 20 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed20 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_SAC --run_id EXP001_aegrl_src_TD3_tgt_SAC_seed20_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_SAC_seed20_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 20 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed20 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_SAC --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_TD3_tgt_SAC_seed20_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_SAC_seed30_train
python advTrain.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 30 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed30 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_SAC --run_id EXP001_aegrl_src_TD3_tgt_SAC_seed30_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_SAC_seed30_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 30 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed30 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_SAC --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_TD3_tgt_SAC_seed30_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_SAC_seed40_train
python advTrain.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 40 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed40 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_SAC --run_id EXP001_aegrl_src_TD3_tgt_SAC_seed40_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_SAC_seed40_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo SAC --adv_algo SAC_FGSM --seed 40 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed40 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_SAC --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_TD3_tgt_SAC_seed40_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_TD3_seed0_train
python advTrain.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 0 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed0 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_TD3 --run_id EXP001_aegrl_src_TD3_tgt_TD3_seed0_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_TD3_seed0_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 0 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed0 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_TD3 --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_TD3_tgt_TD3_seed0_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_TD3_seed10_train
python advTrain.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 10 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed10 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_TD3 --run_id EXP001_aegrl_src_TD3_tgt_TD3_seed10_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_TD3_seed10_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 10 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed10 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_TD3 --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_TD3_tgt_TD3_seed10_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_TD3_seed20_train
python advTrain.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 20 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed20 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_TD3 --run_id EXP001_aegrl_src_TD3_tgt_TD3_seed20_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_TD3_seed20_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 20 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed20 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_TD3 --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_TD3_tgt_TD3_seed20_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_TD3_seed30_train
python advTrain.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 30 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed30 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_TD3 --run_id EXP001_aegrl_src_TD3_tgt_TD3_seed30_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_TD3_seed30_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 30 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed30 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_TD3 --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_TD3_tgt_TD3_seed30_eval --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_TD3_seed40_train
python advTrain.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 40 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed40 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_TD3 --run_id EXP001_aegrl_src_TD3_tgt_TD3_seed40_train --experiment_id EXP001

# Run ID: EXP001_aegrl_src_TD3_tgt_TD3_seed40_eval
python evaluation_v3.py --env_name TrafficEnv3-v5 --algo TD3 --adv_algo TD3_FGSM --seed 40 --expert_attack --expert_model_path expert_model/EXP001_TD3_seed40 --expert_prior ValuePenalty --addition_msg src_TD3_tgt_TD3 --result_saving --result_filename EXP001_cross_victim --run_id EXP001_aegrl_src_TD3_tgt_TD3_seed40_eval --experiment_id EXP001

