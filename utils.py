import numpy as np
from stable_baselines3 import SAC, PPO, TD3  # 或者您使用的其他算法
import os
import torch as th
from policy import FniNet, SumoNet, DarrlNet

def get_attack_prob(act_list):
    alpha = 4
    beta = 1.5
    # print('act list ', act_list)
    attack_prob = 1 - np.prod([1 - 1 / (1 + np.exp(alpha * (act - beta))) for act in act_list])
    return attack_prob


def get_trained_agent(args, device):
    if args.best_model and args.algo!=TD3:
        if args.addition_msg!='' and args.attack == False:
            model_path = os.path.join(args.path, args.env_name, args.algo, args.addition_msg, 'best_model/best_model')
        else:
            model_path = os.path.join(args.path, args.env_name, args.algo, 'best_model/best_model')
        #model_path = os.path.join(args.path, args.env_name, args.algo, 'best_model/best_model')
        # model_path = os.path.join(args.path, args.env_name, args.algo,  'best_model/best_model')
    else:
        if args.addition_msg!=''and args.attack == False:
            model_path = os.path.join(args.path, args.env_name, args.algo, args.addition_msg, 'lunar')
        else:
            model_path = os.path.join(args.path, args.env_name, args.algo, 'lunar')
        # model_path = os.path.join(args.path, args.env_name, args.algo, args.addition_msg, 'lunar')
        #model_path = os.path.join(args.path, args.env_name, args.algo, 'lunar')
        print('**********************************************************')

    if args.algo == 'PPO':
        print('*******************algo is PPO*******************')
        trained_agent = PPO.load(model_path, device=device)
    elif args.algo == 'SAC':
        print('*******************algo is SAC*******************')
        trained_agent = SAC.load(model_path, device=device)
    elif args.algo == 'TD3':
        print('*******************algo is TD3*******************')
        trained_agent = TD3.load(model_path, device=device)
    else:
        if args.env_name == 'TrafficEnv5-v0':
            state_dim = 29
            action_dim = 2
        else:
            state_dim = 26
            action_dim = 1
        if args.algo == 'FNI':
            # 创建一个新的 Actor 实例
            trained_agent = FniNet(state_dim, action_dim)
            trained_agent.load_state_dict(th.load(os.path.join(args.path, args.env_name, args.algo, args.model_path) + '.pth',
                                                  weights_only=True))
            trained_agent.to(device)
            trained_agent.eval()
        elif args.algo == 'DARRL':
            trained_agent = DarrlNet(state_dim, action_dim)
            trained_agent.load_state_dict(th.load(os.path.join(args.path, args.env_name, args.algo, args.model_path) + '.pth',
                                                  weights_only=True))
            trained_agent.to(device)
            trained_agent.eval()
        elif args.algo == 'IL':
            # trained_agent = SumoNet(state_dim, action_dim, [64, 128, 64])
            lr_schedule = lambda _: 0.0003
            trained_agent = policies.ActorCriticPolicy(env.observation_space, env.action_space, lr_schedule)
            # trained_agent = torch.nn.DataParallel(trained_agent)
            state_dict = th.load(os.path.join(args.path, args.env_name, args.algo, args.model_path) + '.pth',
                                 weights_only=False)
            trained_agent.load_state_dict(state_dict['state_dict'])
            trained_agent = trained_agent.to(device)
        trained_agent.eval()
    return trained_agent