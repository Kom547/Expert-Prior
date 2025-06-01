# coding=utf8
import argparse

from grpc import protos_and_services


def get_config():
    parser = argparse.ArgumentParser(
        description='IL_STA', formatter_class=argparse.RawDescriptionHelpFormatter)

    # env parameters
    parser.add_argument('--env_name', default="TrafficEnv1-v0", help='name of the environment to run')
    parser.add_argument('--addition_msg', default="", help='additional message of the training process')
    parser.add_argument('--algo', default="PPO", help='training algorithm')
    parser.add_argument('--adv_algo', default="PPO_FGSM", help='training adv algorithm')
    parser.add_argument('--path', default="./logs/eval/", help='path of the trained model')
    parser.add_argument('--train_step', type=int, default=250, help='number of training episodes')
    parser.add_argument('--T_horizon', type=int, default=30, help='number of training steps per episode')
    parser.add_argument('--print_interval', default=10)
    parser.add_argument('--speed_range', default=15.0, help='Maximum speed')
    parser.add_argument('--max_a', default=7.6, help='Maximum Acceleration')
    parser.add_argument('--state_dim', default=26)
    parser.add_argument('--action_dim', default=1)
    parser.add_argument('--best_model', action='store_true', help='whether to load best model')
    parser.add_argument('--eval', default=False, help='eval flag')
    parser.add_argument('--adv_model_path', default="", help='')
    parser.add_argument('--result_saving', action='store_true', help='whether to save')
    parser.add_argument('--result_filename', default="result", help='path to save result')
    # training parameters
    parser.add_argument('--use_cuda', default=True, help='Use GPU if available')
    parser.add_argument('--cuda_number', type=int, default=0, help='CUDA device number to use')
    parser.add_argument('--cuda_seed', type=int, default=0, help='random seed for network')
    parser.add_argument('--save_freq', type=int, default=100000, help='frequency of saving the model')
    parser.add_argument('--n_steps', type=int, default=512, help='control n_rollout_steps, for PPO')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--n_epochs', type=int, default=5, help='number of training epochs')
    parser.add_argument('--seed', type=int, default=0, help='random seed for sumo')
    parser.add_argument('--clip_range', type=float, default=0.2, help='cliprange for PPO')


    # log parameters
    parser.add_argument('--no_wandb', action='store_true', help='whether to use wandb logging')
    # fni parameters
    parser.add_argument('--fni_model_path', default="", help='model path for FNI/DARRL/IL')

    # parallel parameters
    parser.add_argument('--num_envs', type=int, default=1, help="Number of environments")

    # random env parameters
    parser.add_argument('--random_seed', action='store_true', help='whether random seed')

    # Attack parameters
    parser.add_argument('--attack', action='store_true', help='whether to change the env for attack')
    parser.add_argument('--adv_steps', type=int, default=9, help='number of adversarial attack steps')
    parser.add_argument('--attack_method', default="fgsm", help='which attack method to be applied')
    parser.add_argument('--attack_eps', type=float, default=0.1, help='epsilon-ball around init state')
    parser.add_argument('--attack_iteration',type=int, default=50, help='iterations for attack method')
    parser.add_argument('--step_size', default=0.0075, help='step size for fgsm')

    # ablation parameters
    parser.add_argument('--padding', action='store_true', help='whether to use padding or not')
    parser.add_argument('--clipping', action='store_true', help='whether to use clipping or not')
    parser.add_argument('--decouple', action='store_true', help='whether to use decoupled or not')
    parser.add_argument('--use_act', action='store_true', help='whether to use ACT to assist adversary')
    parser.add_argument('--use_js', action='store_true', help='whether to use JS divergence to help training')
    parser.add_argument('--v_action_obs_flag', action='store_true',
                        help='whether to add the action of victim agent into the obs of the adversary')
    parser.add_argument('--remain_attack_time_flag', action='store_true',
                        help='whether to add the remain attack times into the obs of the adversary')

    # baseline parameters
    parser.add_argument('--unlimited_attack', action='store_true', help='For unlimited attack times baseline')

    #expert_model parameters
    parser.add_argument('--expert_recording', action='store_true', help='whether to record attack actions')
    parser.add_argument('--expert_attack', action='store_true', help='whether to use expert model for attack')
    parser.add_argument('--expert_train_samples', type=int, default=100,help='number of training samples for expert model')
    parser.add_argument('--expert_prior', default='ValuePenalty',choices=['ValuePenalty','PolicyConstrained'],help='which method of utilizing prior for expert model')
    parser.add_argument('--expbase_algo',type=str,default='',help='which algorithm to use')
    parser.add_argument('--expert_model_path',type=str,default='',help='path to expert model')
    parser.add_argument('--expert_data_path',type=str,default='',help='path to expert data')
    parser.add_argument('--expert_model_savepath', type=str, default='', help='path to save expert model')
    parser.add_argument('--expert_eval_csv', type=str, default='exp_eval.csv', help='path to save expert model evaluation')
    parser.add_argument('--expert_k', type=float, default=0.5, help='k for expert control')
    parser.add_argument('--wo_beta', action='store_true', help='whether to use beta for expert')
    parser.add_argument('--expert_cnt',type= int, default=6, help='number of expert models')
    parser.add_argument('--no_lambda_grad', action='store_true', help='whether to update lambda use GD')


    return parser
