import gymnasium as gym
import numpy as np
import traci
from stable_baselines3 import SAC, PPO, TD3  # 或者您使用的其他算法
from stable_baselines3.common import policies
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import obs_as_tensor

from Environment.environment.env8.traffic_env import Traffic_Env
from config import get_config
import Environment.environment
import os
import torch as th
from perturbation import *
import pandas as pd
from utils import get_attack_prob, get_trained_agent
from PIL import Image
import shutil
from expert_imitation_learning import Actor,load_ensemble_models,sample_from_mixture

# get parameters from config.py
parser = get_config()
args = parser.parse_args()

# 在参数解析之后添加数据保存目录
if args.expert_recording:
    edata_path = './expert_data/{}_{}_{}_{}/{}'.format(args.env_name,args.algo,args.adv_algo,args.attack_eps,args.addition_msg)
    os.makedirs(edata_path, exist_ok=True)  # 创建数据保存目录
    shutil.rmtree(edata_path)  # 清理旧数据
    os.makedirs(edata_path, exist_ok=True)

args.expert_attack = True
#专家模型目录
if args.expert_attack:
    expert_model_path = args.expert_model_path

# 设置随机种子
np.random.seed(args.seed)  # 设置 NumPy 随机种子
th.manual_seed(args.seed)  # 设置 CPU 随机种子
if th.cuda.is_available():
    th.cuda.manual_seed(args.seed)  # 设置 CUDA 随机种子
    th.cuda.manual_seed_all(args.seed)  # 设置所有 GPU 随机种子
th.backends.cudnn.deterministic = True  # 确保 CUDA 的确定性
th.backends.cudnn.benchmark = False  # 禁用 CuDNN 自动优化

# 设置设备
if args.use_cuda and th.cuda.is_available():
    device = th.device(f"cuda:{args.cuda_number}")
else:
    device = th.device("cpu")

# 设置eval标志
args.eval = True
args.attack=True
args.adv_steps = 9
# 创建环境
env = gym.make(args.env_name, attack=args.attack, adv_steps=args.adv_steps, eval=args.eval,random_seed = args.random_seed)
env.unwrapped.start()


#加载专家模型
if args.expert_attack:
    expert_models = load_ensemble_models(args.expert_model_path, (28,), 2, device)

# 加载训练好的自动驾驶模型
trained_agent = get_trained_agent(args, device)

# 进行验证
rewards = []
ep_steps = []

maxSpeed = 15.0
ct = 0
sn = 0
sat = 0
speed_list = []
attack_count_list = []
mean_attack_reward_list = []
log_list = []
last_act_list = []

epa_list = np.zeros(30)

for episode in range(args.train_step):
    # 初始化当前episode的存储
    if args.expert_recording:
        episode_obs = []
        episode_adv_actions = []
        episode_actions = []

    obs, info = env.reset()
    #img = env.render()
    speed = 0
    episode_reward = 0
    episode_steps = 0
    # save_dir = f'./render/{episode}'
    # # 创建目录（如果不存在的话）
    # os.makedirs(save_dir, exist_ok=True)

    for steps in range(args.T_horizon):

        obs_tensor = obs_as_tensor(obs, device)

        if args.attack:
            #获取速度
            speed_list.append(obs[-4])
            #获取初始动作
            if args.algo in ('FNI', 'DARRL'):
                actions, std, _action = trained_agent(obs_tensor[:-2])
                actions = actions.detach().cpu().numpy()
            elif args.algo == 'IL':
                actions, _, _ = trained_agent(obs_tensor[:-2])
                actions = actions.detach().cpu().numpy()
            else:
                actions, _ = trained_agent.predict(obs_tensor[:-2].cpu(), deterministic=True)

            #转化成tensor
            actions_tensor = th.tensor(actions, device=obs_tensor.device)  # 确保在同一设备上
            obs_tensor[-1] = (actions_tensor+1) / 2


            #专家攻击
            if args.expert_attack:
                expert_actions,_,_ = sample_from_mixture(obs, expert_models,device=device)
                adv_actions = expert_actions.cpu().numpy()[0]

            # 记录当前步的obs和adv_actions
            if args.expert_recording:
                episode_obs.append(obs.copy())
                episode_adv_actions.append(adv_actions.copy())

            #对抗动作判断攻击且剩余攻击次数＞0
            adv_action_mask = (adv_actions[0] > 0) & (obs[-2] > 0)
            adv_flag = 1 if adv_actions[0] > 0 else 0
            # if adv_flag:
            #     print('steps ', episode_steps, 'pre actions', actions, 'adv_actions', adv_actions, 'attack prob ', attack_prob)
            # log_list.append([args.algo, args.attack_method, args.adv_steps, adv_flag, steps, attack_prob])
            if adv_flag:
                print('steps ', episode_steps, 'attack ','pre actions', actions, 'adv_actions', adv_actions)
            log_list.append([args.algo, args.attack_method, args.adv_steps, adv_flag, steps])
            # print(adv_action_MAD,actions)
            #通过扰动观察获取对抗动作
            if adv_action_mask or args.unlimited_attack:
                epa_list[episode_steps]+=1
                #获取扰动观察
                if args.attack_method == 'fgsm':
                    adv_state = FGSM_v2(adv_actions[1], victim_agent=trained_agent, last_state=obs_tensor[:-2],
                                        device=device,epsilon=args.attack_eps,num_iterations=args.attack_iteration)
                elif args.attack_method == 'pgd':
                    adv_state = PGD(adv_actions[1], trained_agent, obs_tensor[:-2], device=device,epsilon=args.attack_eps,num_iterations=args.attack_iteration)

                #获取对抗动作
                if args.attack_method == 'direct':
                    action = adv_actions[1]
                else:
                    if args.algo in ('FNI', 'DARRL'):
                        adv_action_fromState, _, _ = trained_agent(adv_state)
                        action = adv_action_fromState.detach().cpu().numpy()
                    elif args.algo == 'IL':
                        adv_action_fromState, _, _ = trained_agent(adv_state)
                        action = adv_action_fromState.detach().cpu().numpy()
                    else:
                        adv_action_fromState, _ = trained_agent.predict(adv_state.cpu(), deterministic=True)
                        print(episode_steps, 'attack', '{} action is'.format(args.attack_method), adv_action_fromState)
                        action = adv_action_fromState
            else:
                action = actions
            # action = adv_action_FGSM[0]
            action = np.column_stack((action, adv_action_mask))
            obs, reward, done, terminate, info = env.step(action[0])
        else:
            #不攻击的情况
            speed_list.append(obs[-2])
            # actions = trained_agent.policy(obs_tensor.unsqueeze(0))
            # actions1 = trained_agent.policy(obs_tensor.unsqueeze(0), deterministic=True)
            if args.algo in ('FNI', 'DARRL'):
                actions, std, _action = trained_agent(obs_tensor)
                actions = actions.cpu().detach().numpy()
            elif args.algo == 'IL':
                # actions, _, _ = trained_agent(obs_tensor)
                # actions = actions.cpu().detach().numpy()
                actions, _ = trained_agent.predict(obs, deterministic=True)
            else:
                actions, _ = trained_agent.predict(obs, deterministic=True)
            # 记录正常情况下的动作
            if args.expert_recording:
                episode_obs.append(obs.copy())
                episode_actions.append(actions.copy())
            # actions3,_ = trained_agent.predict(obs)
            # print(actions,actions1,actions2,actions3)
            obs, reward, done, terminate, info = env.step(actions)


        episode_reward += reward
        episode_steps += 1
        if done:
            if round(args.adv_steps - obs[-2] * args.adv_steps) != 0:
                mean_attack_reward_list.append(1 / round(args.adv_steps - obs[-2] * args.adv_steps))
            ct += 1
            if args.attack:
                if round(args.adv_steps - obs[-2] * args.adv_steps):
                    sat += 1
            break
    xa = info['x_position']
    ya = info['y_position']
    if args.unlimited_attack:
        attack_count_list.append(episode_steps)
    else:
        attack_count_list.append(round(args.adv_steps - obs[-2] * args.adv_steps))
    if args.env_name == 'TrafficEnv1-v0' or args.env_name == 'TrafficEnv3-v0' or args.env_name == 'TrafficEnv6-v0':
        if xa < -50.0 and ya > 4.0 and done is False:
            sn += 1
    elif args.env_name == 'TrafficEnv2-v0':
        if xa > 50.0 and ya > -5.0 and done is False:
            sn += 1
    elif args.env_name == 'TrafficEnv4-v0':
        if ya < -50.0 and done is False:
            sn += 1
    elif args.env_name == 'TrafficEnv8-v0':
        if ya == 10.0 and done is False:
            sn += 1
    rewards.append(episode_reward)
    ep_steps.append(episode_steps)

    # 在episode结束后保存数据 攻击成功的回合才保存
    if args.expert_recording and episode_steps<29:
        save_path = f'{edata_path}/sa_episode_{episode}.npz'
        if args.attack:
            np.savez(
                save_path,
                obs=np.array(episode_obs, dtype=np.float32),
                adv_actions=np.array(episode_adv_actions, dtype=np.float32)
            )
        else:
            np.savez(
                save_path,
                obs=np.array(episode_obs, dtype=np.float32),
                actions=np.array(episode_actions, dtype=np.float32)
            )
print('epa: ',epa_list)
# 计算平均奖励和步数
mean_reward = np.mean(rewards)
std_reward = np.std(rewards)
mean_steps = np.mean(steps)
std_steps = np.std(steps)

# 计算碰撞率
cr = ct / args.train_step * 100
sr = sn / args.train_step * 100
if args.attack:
    if sum(1 for x in attack_count_list if x > 0) > 0:
        asr = sat / sum(1 for x in attack_count_list if x > 0) * 100
    else:
        asr = 0.00
else:
    asr = 0.00

# 计算平均速度
mean_speed = np.mean(speed_list)
std_speed = np.std(speed_list)

# 计算平均攻击次数
attack_list = [x for x in attack_count_list if x != 0]
mean_attack_times = np.mean(attack_list)
std_attack_times = np.std(attack_list)

# 计算单次攻击的收益
mean_attack_reward = np.mean(mean_attack_reward_list)
std_attack_reward = np.std(mean_attack_reward_list)

attack_score = 0.3*(1-(mean_attack_times-2)/9.0)+0.7*cr
print('attack_score: ',attack_score)

print('attack lists ', attack_count_list, 'attack times ', len(attack_list))
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
print(f"Mean steps: {mean_steps:.2f} +/- {std_steps:.2f}")
print(f"Mean speed: {mean_speed * maxSpeed:.2f} +/- {std_speed * maxSpeed:.2f}")
print(f"Mean attack times: {mean_attack_times:.2f} +/- {std_attack_times:.2f}")
print(f"Collision rate: {cr:.2f}")
print(f"Success rate: {sr:.2f}")
print(f"Success attack rate: {asr:.2f}")
print(f"Reward per attack: {mean_attack_reward:.2f} +/- {std_attack_reward:.2f}")


#Save evaluation results to CSV

data = {
    'env_name': [args.env_name],
    'train_step': [args.train_step],
    'algo': [args.algo],
    'attack_eps': [args.attack_eps],
    'mean_attack_times': [mean_attack_times],
    'collision_rate': [cr],
    'attack_score':[attack_score]
}
df = pd.DataFrame(data)
csv_file = args.expert_eval_csv
if not os.path.exists(csv_file):
    df.to_csv(csv_file, index=False)
else:
    df.to_csv(csv_file, mode='a', header=False, index=False)