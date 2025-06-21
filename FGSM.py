import numpy as np
import torch
import torch.nn as nn
from policy import FniNet as Actor
import torch.optim as optim
import matplotlib.pyplot as plt
import os
def fgsm_attack(adv_action, victim_agent, last_state, epsilon=0.1,alpha=0.0075,
                 device='cuda:0',num_iterations=50, state_min=0, state_max=1):
    """
    Perform Iterative FGSM attack on the last_state to make the victim_agent predict adv_action.

    Parameters:
    - adv_action (np.ndarray or torch.Tensor): The desired adversarial action.
    - victim_agent (stable_baselines3.SAC): The trained SAC agent.
    - last_state (np.ndarray): The state to perturb.
    - epsilon (float): The maximum perturbation magnitude.
    - alpha (float): The step size for each iteration.
    - num_iterations (int): The number of iterations.

    Returns:
    - perturbed_state (np.ndarray): The perturbed state.
    """
    #alpha = epsilon/num_iterations
    if not isinstance(adv_action, torch.Tensor):
        adv_action = torch.tensor(adv_action, dtype=torch.float32, device=device)

        # Convert last_state to torch tensor and add batch dimension
    state = torch.tensor(last_state, dtype=torch.float32, device=device).unsqueeze(0)
    state.requires_grad = True
    # Set the policy to evaluation mode
    # Disable gradient computation for policy parameters to speed up
    if isinstance(victim_agent,Actor):
        victim_agent.eval()
        for param in victim_agent.parameters():
            param.requires_grad = False
    else:
        victim_agent.policy.eval()
        for param in victim_agent.policy.parameters():
            param.requires_grad = False



    # Initialize the perturbation
    perturbation = torch.zeros_like(state, device=device)

    # Iterative FGSM
    for i in range(num_iterations):
        # Zero existing gradients
        if state.grad is not None:
            state.grad.data.zero_()
        # Forward pass: compute the predicted action
        if isinstance(victim_agent, Actor):
            #_,_,action_pred = victim_agent(state+perturbation)
            action_pred,_,_ = victim_agent(state+perturbation)
        else:
            adv_state = state + perturbation
            action_pred = victim_agent.policy(adv_state, deterministic=True)
            # print('action pred is', action_pred)
            if action_pred[0].dim() > 1:
                action_pred = action_pred[0].squeeze(0)
            else:
                action_pred = action_pred[0]
            # print('action_pred is', action_pred)
        # Compute the loss between predicted action and adversarial action
        loss_fn = nn.MSELoss()
        #l1_lambda = 0.01
        loss = loss_fn(action_pred, adv_action.unsqueeze(0))

        # Backward pass: compute gradient of loss w.r.t. perturbation
        loss.backward()

        grad_sign = torch.clamp(state.grad.sign(), min=-1.0, max=1.0)
        # Collect the sign of the gradient
        #grad_sign = state.grad.sign()

        # Update the perturbation (for targeted attack, subtract the gradient)
        perturbation = perturbation - alpha * grad_sign
        # Project the perturbation to be within the epsilon ball
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)

        # If state bounds are provided, clamp the perturbed state
        if state_min is not None and state_max is not None:
            state_min_tensor = torch.tensor(state_min, dtype=torch.float32,
                                            device=device).unsqueeze(0)
            state_max_tensor = torch.tensor(state_max, dtype=torch.float32,
                                            device=device).unsqueeze(0)
            perturbed_state = state + perturbation
            perturbed_state = torch.max(torch.min(perturbed_state, state_max_tensor), state_min_tensor)
            perturbation = perturbed_state - state
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)


        # Detach to prevent accumulating gradients
        perturbation = perturbation.detach()

        # 打印调试信息
        #print(f"Iteration {i + 1}:")
        #print(f"Loss: {loss.item():.4f}")
        # print(f"Action_pred: {action_pred}")
        # print(f"Adv_action: {adv_action.cpu().numpy()}")
        # print(f"Perturbation: {perturbation.squeeze().cpu().numpy()}\n")
    # Apply the final perturbation
    perturbed_state = state + perturbation
    if state_min is not None and state_max is not None:
        state_min_tensor = torch.tensor(state_min, dtype=torch.float32, device=device).unsqueeze(0)
        state_max_tensor = torch.tensor(state_max, dtype=torch.float32, device=device).unsqueeze(0)
        final_perturbed_state = torch.max(torch.min(perturbed_state, state_max_tensor), state_min_tensor)
    # Detach the tensor and convert back to numpy
    if isinstance(victim_agent, Actor):
        perturbed_state = final_perturbed_state
    else:
        perturbed_state = final_perturbed_state.detach().cpu().numpy()[0]

    return perturbed_state


def SGLD(adv_action_mean,adv_action_stddev, victim_agent, last_states, attack_eps=0.05,
                 attack_steps=500):

    last_states = last_states[:,:-1]
    action_mean, action_stddev, _ = victim_agent(last_states.cpu())

    eps = attack_eps
    steps = attack_steps
    #print(last_states)
    if steps > 0:
        step_eps = eps / steps
        clamp_min = torch.max((last_states - eps),torch.zeros_like(last_states))
        clamp_max = torch.min((last_states + eps),torch.ones_like(last_states))
        # SGLD noise factor. We simply set beta=1.
        noise_factor = np.sqrt(2 / step_eps)
        noise = torch.randn_like(last_states) * noise_factor
        # The first step has gradient zero, so add the noise and projection directly.
        states = last_states + noise.sign() * step_eps
        states.data = torch.min(torch.max(states.data, clamp_min), clamp_max)

        with torch.enable_grad():
            for i in range(steps):
                states = states.clone().detach().requires_grad_()
                #print('s',states)
                new_action_mean, new_action_stddev,_ = victim_agent(states.cpu())
                new_action_mean=new_action_mean.to('cuda:0')
                new_action_stddev = new_action_stddev.to('cuda:0')
                #print(new_actions)
                #print(new_action_mean,adv_action_mean,new_action_stddev,adv_action_stddev)

                #print(new_action_stddev)
                kl_div = torch.log(adv_action_stddev / new_action_stddev) + (
                        new_action_stddev ** 2 + (new_action_mean - adv_action_mean) ** 2) / (
                                 2 * adv_action_stddev ** 2) - 0.5
                MADLoss = kl_div  # 对多维动作空间求和
                print('loss',MADLoss)
                # MADLoss = th.sum(adv_action_probs * th.log(adv_action_probs/new_action_probs))
                MADLoss.backward(retain_graph=True)
                # Reduce noise at every step.
                noise_factor = np.sqrt(2 * step_eps) / (i + 2)
                # Project noisy gradient to step boundary.
                update = (states.grad + noise_factor * torch.randn_like(last_states)).sign() * step_eps
                # Clamp to +/- eps.
                #print('update',update)
                states.data = torch.min(torch.max(states.data - update, clamp_min), clamp_max)

            victim_agent.zero_grad()
        return states.detach()
    else:
        return last_states

#废弃
def FAB_FGSM(victim_agent, adv_actions,last_states, eps=0.1,T=10, delta_vanish=0.01,device='cuda:0'):
    # Initialize variables
    if not isinstance(adv_actions, torch.Tensor):
        adv_actions = torch.tensor(adv_actions, dtype=torch.float32, device="cuda:0").unsqueeze(0)
    #print('last',last_states)
    m_t = torch.zeros_like(last_states)
    s_t = torch.zeros_like(last_states)
    beta1 = 0.9
    #beta2 = 0.1
    alpha = eps/np.sqrt(T+1)
    gamma = 0
    weight_decay = 0.01
    optim_epsilon = 1e-7

    clamp_min = torch.max((last_states - eps), torch.zeros_like(last_states))
    clamp_max = torch.min((last_states + eps), torch.ones_like(last_states))

    # noise_factor = np.sqrt(2 / alpha)
    # noise = torch.randn_like(last_states) * noise_factor
    # # The first step has gradient zero, so add the noise and projection directly.
    states_adv = last_states
    # Gradient of the loss w.r.t input

    loss_list=[]
    for t in range(T):
        states_adv.requires_grad_ = True

        t = t+1
        beta2 = 1-0.9/t

        #print(states_adv)
        if isinstance(victim_agent, Actor):
            new_actions,_, _ = victim_agent(states_adv)
            victim_agent.zero_grad()
        else:
            new_actions = victim_agent.policy(states_adv, deterministic=True)
            # print('action pred is', action_pred)
            if new_actions[0].dim() > 1:
                new_actions = new_actions[0].squeeze(0)
            else:
                new_actions = new_actions[0]
            victim_agent.policy.zero_grad()

        #print('ac',new_actions,adv_actions)
        loss = -nn.MSELoss()(new_actions, adv_actions).to(device)
        loss_list.append(loss.item())
        loss.backward()

        grad = states_adv.grad
        # Compute γ'
        gamma += np.sqrt(1 - beta2**(t)) / (1 - beta1**(t))

        # Update m_t and s_t
        m_t = beta1 * m_t + (1 - beta1) * grad
        s_temp = s_t
        s_t = beta2 * s_t + (1 - beta2) * (grad - m_t) ** 2
        s_t = torch.max(s_t, s_temp)

        states_adv = states_adv*(1-alpha*weight_decay)

        # Update m_t' and s_t'
        m_t_hat = m_t / (1 - beta1**t)
        s_t_hat = (s_t + optim_epsilon) / (1 - beta2**t) + delta_vanish/t

        # Update adversarial example
        step = alpha / gamma * torch.sign(m_t_hat / (s_t_hat+optim_epsilon))
        #print(step)
        states_adv = states_adv + step

        # Clip to ensure perturbation constraint
        states_adv = torch.clamp(states_adv, clamp_min, clamp_max)
    # 创建保存图像的目录（如果不存在）
    save_dir = "lossForFABFGSM"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 绘制 loss2 随迭代次数的变化并保存图像
    plt.plot(range(len(loss_list)), loss_list, label='Loss over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs Iterations')
    plt.legend()

    # 保存图像到指定文件夹
    plt.savefig(os.path.join(save_dir, 'loss_iterations.png'))
    return states_adv
def FAB_FGSM_v2(victim_agent,adv_action, last_state,  epsilon=0.1,
                 device='cuda:0', T=50, delta_vanish=0.01):
    # print(last_state)
    m_t = torch.zeros_like(last_state)
    s_t = torch.zeros_like(last_state)
    beta1 = 0.9
    alpha = epsilon / np.sqrt(T + 1)
    #alpha=0.1
    gamma = 0
    weight_decay = 0.01
    optim_epsilon = 1e-7

    clamp_min = torch.max((last_state - epsilon), torch.zeros_like(last_state))
    clamp_max = torch.min((last_state + epsilon), torch.ones_like(last_state))

    last_state = last_state.to(device)

    if not isinstance(adv_action, torch.Tensor):
        adv_action = torch.tensor(adv_action, dtype=torch.float32, device=device)
    loss = nn.MSELoss()

    ori_state = last_state.data
    cost_list = []
    adv_state_list = []
    for t in range(T):
        last_state.requires_grad = True

        t = t + 1
        beta2 = 1 - 0.9 / t

        if isinstance(victim_agent, Actor):
            outputs, _, _ = victim_agent(last_state)
            victim_agent.zero_grad()
        else:
            outputs = victim_agent.policy(last_state, deterministic=True)
            # print('action pred is', action_pred)
            if outputs[0].dim() > 1:
                outputs = outputs[0].squeeze(0)
            else:
                outputs = outputs[0]
            victim_agent.policy.zero_grad()

        cost = -loss(outputs, adv_action).to(device)
        cost_list.append(cost.item())
        cost.backward()

        grad = last_state.grad
        # Compute γ'
        gamma += np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

        # Update m_t and s_t
        m_t = beta1 * m_t + (1 - beta1) * grad
        s_temp = s_t
        s_t = beta2 * s_t + (1 - beta2) * (grad - m_t) ** 2
        s_t = torch.max(s_t, s_temp)

        last_state = last_state * (1 - alpha * weight_decay)

        # Update m_t' and s_t'
        m_t_hat = m_t / (1 - beta1 ** t)
        s_t_hat = (s_t + optim_epsilon) / (1 - beta2 ** t) + delta_vanish / t

        # Update adversarial example
        step = alpha / gamma * torch.sign(m_t_hat / (s_t_hat + optim_epsilon))
        #step = alpha / t * torch.sign(m_t_hat / (s_t_hat + optim_epsilon))
        # print(step)
        last_state = last_state + step

        # Clip to ensure perturbation constraint
        last_state = torch.clamp(last_state, clamp_min, clamp_max).detach()


    # # 创建保存图像的目录（如果不存在）
    # save_dir = "lossForFABFGSM"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #
    # # 绘制 loss2 随迭代次数的变化并保存图像
    # plt.plot(range(len(cost_list)), cost_list, label='Loss over iterations')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.title('Loss vs Iterations')
    # plt.legend()
    #
    # # 保存图像到指定文件夹
    # plt.savefig(os.path.join(save_dir, 'loss_iterations_T{}_eps{}_ss.png'.format(T,epsilon)))
    return last_state
def PGD(adv_action, victim_agent, last_state, epsilon=0.1,alpha=0.0075,
                 device='cuda:0',num_iterations=50):
    #print(last_state)
    clamp_min = torch.max((last_state - epsilon), torch.zeros_like(last_state))
    clamp_max = torch.min((last_state + epsilon), torch.ones_like(last_state))

    last_state = last_state.to(last_state.device)
    last_state = last_state + torch.empty_like(last_state).uniform_(-epsilon, epsilon)
    last_state = torch.clamp(last_state,clamp_min,clamp_max)
    if not isinstance(adv_action, torch.Tensor):
        adv_action = torch.tensor(adv_action, dtype=torch.float32, device=device)
    loss = nn.MSELoss()

    ori_state = last_state.data
    cost_list = []
    adv_state_list = []
    for i in range(num_iterations):
        last_state.requires_grad = True

        if isinstance(victim_agent, Actor):
            outputs,_, _ = victim_agent(last_state)
            victim_agent.zero_grad()
        else:
            outputs = victim_agent.policy(last_state.unsqueeze(0), deterministic=True)
            # print('action pred is', action_pred)
            if outputs[0].dim() > 1:
                outputs = outputs[0].squeeze(0)
            else:
                outputs = outputs[0]
            victim_agent.policy.zero_grad()

        #print('o,a',outputs,adv_action)
        cost = -loss(outputs, adv_action).to(device)
        cost_list.append(cost.item())
        cost.backward()

        adv_state = last_state + alpha * last_state.grad.sign()
        eta = torch.clamp(adv_state - ori_state, min=-epsilon, max=epsilon)
        last_state = torch.clamp(ori_state + eta, min=clamp_min, max=clamp_max).detach_()

    # # 创建保存图像的目录（如果不存在）
    # save_dir = "lossForPGD"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #
    # # 绘制 loss2 随迭代次数的变化并保存图像
    # plt.plot(range(len(cost_list)), cost_list, label='Loss over iterations')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.title('Loss vs Iterations')
    # plt.legend()
    #
    # # 保存图像到指定文件夹
    # plt.savefig(os.path.join(save_dir, 'loss_iterations.png'))
    return last_state

def FGSM_v2(adv_action, victim_agent, last_state, epsilon=0.1,
                 device='cuda:0',num_iterations=50):
    alpha = epsilon/num_iterations
    #print(last_state)
    clamp_min = torch.max((last_state - epsilon), torch.zeros_like(last_state))
    clamp_max = torch.min((last_state + epsilon), torch.ones_like(last_state))

    last_state = last_state.to(last_state.device)
    if not isinstance(adv_action, torch.Tensor):
        adv_action = torch.tensor(adv_action, dtype=torch.float32, device=device)
    loss = nn.MSELoss()

    ori_state = last_state.data
    cost_list = []
    adv_state_list = []
    for i in range(num_iterations):
        last_state.requires_grad = True

        if isinstance(victim_agent, Actor):
            outputs,_, _ = victim_agent(last_state)
            victim_agent.zero_grad()
        else:
            outputs = victim_agent.policy(last_state.unsqueeze(0), deterministic=True)
            # print('action pred is', action_pred)
            if outputs[0].dim() > 1:
                outputs = outputs[0].squeeze(0)
            else:
                outputs = outputs[0]
            victim_agent.policy.zero_grad()

        #print('o,a',outputs,adv_action)
        cost = -loss(outputs, adv_action).to(device)
        #print(cost)
        cost_list.append(cost.item())
        cost.backward()

        last_state = torch.clamp(last_state + alpha * last_state.grad.sign(),min=clamp_min, max=clamp_max).detach_()

    # # 创建保存图像的目录（如果不存在）
    # save_dir = "lossForFGSM"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #
    # # 绘制 loss2 随迭代次数的变化并保存图像
    # plt.plot(range(len(cost_list)), cost_list, label='Loss over iterations')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.title('Loss vs Iterations')
    # plt.legend()
    #
    # # 保存图像到指定文件夹
    # plt.savefig(os.path.join(save_dir, 'loss_iterations.png'))
    return last_state
def cw_attack(victim_agent, last_state, adv_action, targeted=True, epsilon=0.1,c=1e-4,lr=0.01, iters=150):
    clamp_min = torch.max((last_state - epsilon), torch.zeros_like(last_state))
    clamp_max = torch.min((last_state + epsilon), torch.ones_like(last_state))
    if not isinstance(adv_action, torch.Tensor):
        adv_action = torch.tensor(adv_action, dtype=torch.float32, device=last_state.device)
    # 初始化扰动变量 w，它是优化目标，初始为原始图像
    w = torch.zeros_like(last_state, requires_grad=True).to(last_state.device)
    factor=1e-6
    last_state=torch.clamp(last_state, -1+factor, 1-factor)
    # 需要优化的目标是 perturbation w
    optimizer = optim.Adam([w], lr=lr)

    # 计算 tanh 逆变换的 x 原始值
    def to_tanh_space(x):
        return 0.5 * (torch.tanh(x) + 1)

    # 计算 x 的逆变换
    def from_tanh_space(x):
        _x = torch.clamp((x - 0.5) * 2, -1 + factor, 1- factor)
        return torch.atanh(_x)

    # 原始输入图像的逆变换形式
    original_state = from_tanh_space(last_state)
    #print('l,o',last_state,original_state)
    loss_list=[]
    for step in range(iters):
        # 通过 tanh 变换生成对抗样本 x'
        #print('w,w+o',w,original_state+w)
        adv_state = to_tanh_space(original_state + w)
        # 加入 epsilon 邻域的约束 (L_infinity 范数)
        perturbation = adv_state - last_state
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        adv_state = torch.clamp(adv_state + perturbation, clamp_min, clamp_max)
        #print('a',adv_state)
        # 计算模型输出
        #outputs,_,_ = victim_agent(adv_state)
        if isinstance(victim_agent, Actor):
            outputs,_, _ = victim_agent(adv_state)
        else:
            outputs = victim_agent.policy(adv_state, deterministic=True)
            # print('action pred is', action_pred)
            if outputs[0].dim() > 1:
                outputs = outputs[0].squeeze(0)
            else:
                outputs = outputs[0]

        # 创建一个标签，用于 targeted 或 non-targeted 的攻击目标
        if targeted:
            # real = torch.nn.functional.one_hot(labels, num_classes=outputs.shape[1]).float().to(images.device)
            # other = torch.ones_like(real) - real
            # loss1 = torch.sum(other * torch.clamp(outputs, min=0) - real * torch.clamp(outputs, min=0))
            loss1 = nn.MSELoss()(outputs,adv_action)
        else:
            # 非目标攻击
            # correct_logit = outputs.gather(1, labels.unsqueeze(1)).squeeze(1)
            # max_other_logit = torch.max(outputs - torch.eye(outputs.shape[1])[labels].to(images.device) * 1e4, dim=1)[0]
            # loss1 = torch.clamp(correct_logit - max_other_logit + kappa, min=0)
            #ori_action,_,_ = victim_agent(original_state)
            if isinstance(victim_agent, Actor):
                ori_action, _, _ = victim_agent(original_state)
            else:
                ori_action = victim_agent.policy(original_state, deterministic=True)
                # print('action pred is', action_pred)
                if ori_action[0].dim() > 1:
                    ori_action = ori_action[0].squeeze(0)
                else:
                    ori_action = ori_action[0]
            loss1 = -nn.MSELoss()(outputs,ori_action)

        # 损失函数由两部分组成：1. 分类损失 2. L2 范数约束
        l2_loss = torch.norm(adv_state - last_state, p=2)
        loss = c * l2_loss + loss1.mean()
        loss_list.append(loss.item())

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



    # # 创建保存图像的目录（如果不存在）
    # save_dir = "lossForCW"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #
    # # 绘制 loss2 随迭代次数的变化并保存图像
    # plt.plot(range(len(loss_list)), loss_list, label='Loss over iterations')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.title('Loss vs Iterations')
    # plt.legend()
    #
    # # 保存图像到指定文件夹
    # plt.savefig(os.path.join(save_dir, 'loss_iterations_iter{}.png'.format(iters)))
    # 返回优化后的对抗样本
    return to_tanh_space(original_state + w).detach()

def cw_attack_v2(victim_agent, last_state, adv_action, targeted=True, epsilon=0.1,c=1e-4, lr=0.01, iters=1000):

    if not isinstance(adv_action, torch.Tensor):
        adv_action = torch.tensor(adv_action, dtype=torch.float32, device=last_state.device)

    BINARY_SEARCH_STEPS = 1  # number of times to adjust the constant with binary search
    MAX_ITERATIONS = iters  # number of iterations to perform gradient descent

    # C的初始化边界
    lower_bound = 0
    upper_bound = 1e10

    # 若攻击成功 记录最好的l2 loss 以及 adv_state
    o_bestl2 = 1e10
    o_bestscore = -1
    o_bestattack = np.zeros(last_state.shape)

    #epsilon-ball around last_state
    clamp_min = torch.max((last_state - epsilon), torch.zeros_like(last_state))
    clamp_max = torch.min((last_state + epsilon), torch.ones_like(last_state))

    #用于将tanh空间转换时的状态范围约束在epsilon-ball内
    mul = (clamp_max-clamp_min)/2
    plus = (clamp_max+clamp_min)/2

    factor = 1e-6
    last_state = torch.clamp(last_state, -1 + factor, 1 - factor)

    # 计算 tanh 逆变换的 x 原始值
    def to_tanh_space(x):
        return mul*torch.tanh(x)+plus

    # 计算 x 的逆变换
    def from_tanh_space(x):
        return torch.atanh((x-plus)/mul*0.999999)

    # 原始输入图像的逆变换形式
    original_state = from_tanh_space(last_state)
    #print('l,o',last_state,original_state)
    loss_list=[]

    #外循环，用于寻找最优c值 , c值用来控制loss中loss1和loss2的权重
    for outer_step in range(BINARY_SEARCH_STEPS):
        print("best_l2={},confidence={}".format(o_bestl2,c))

        # 初始化扰动变量 w，它是优化目标，初始为原始图像
        w = torch.zeros_like(last_state, requires_grad=True).to(last_state.device)
        # 需要优化的目标是 perturbation w
        optimizer = optim.Adam([w], lr=lr)
        #内循环，用于优化以生成对抗状态
        for step in range(1,MAX_ITERATIONS+1):
            # 通过 tanh 变换生成对抗样本 x'
            adv_state = to_tanh_space(original_state + w)

            # 计算模型输出
            if isinstance(victim_agent, Actor):
                outputs,_, _ = victim_agent(adv_state)
            else:
                outputs = victim_agent.policy(adv_state.unsqueeze(0), deterministic=True)
                if outputs[0].dim() > 1:
                    outputs = outputs[0].squeeze(0)
                else:
                    outputs = outputs[0]

            # 创建一个标签，用于 targeted 或 non-targeted 的攻击目标
            if targeted:
                loss1 = nn.MSELoss()(outputs,adv_action)
            else:
                # 非目标攻击
                if isinstance(victim_agent, Actor):
                    ori_action, _, _ = victim_agent(last_state)
                else:
                    ori_action = victim_agent.policy(last_state, deterministic=True)
                    # print('action pred is', action_pred)
                    if ori_action[0].dim() > 1:
                        ori_action = ori_action[0].squeeze(0)
                    else:
                        ori_action = ori_action[0]
                loss1 = -nn.MSELoss()(outputs,ori_action)

            # 损失函数由两部分组成：1. 分类损失 2. L2 范数约束
            l2_loss = torch.norm(adv_state - last_state, p=2)
            loss = c * l2_loss + loss1.mean()
            loss_list.append(loss.item())

            # print out loss every 10%
            if step % (MAX_ITERATIONS // 10) == 0:
                print("iteration={} loss={} loss1={} loss2={} action={}".format(step, loss, loss1, l2_loss,outputs))

            l2 = l2_loss
            #攻击成功的情况 即成功误导agent做出指定action
            if (l2 < o_bestl2) and (outputs == adv_action):
                print("attack success l2={} target_action={}".format(l2, adv_action))
                o_bestl2 = l2
                o_bestscore = outputs
                o_bestattack = adv_state.data.cpu().numpy()

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        confidence_old = 1
        if (o_bestscore == adv_action) and o_bestscore != -1:
            # 攻击成功，减小c
            upper_bound = min(upper_bound, c)
            if upper_bound < 1e9:
                confidence_old = c
                c = (lower_bound + upper_bound) / 2
            else:
                lower_bound = max(lower_bound, c)
                confidence_old = c
                if upper_bound < 1e9:
                    c = (lower_bound + upper_bound) / 2
                else:
                    c *= 10
            # torch.sign
        print("outer_step={} confidence {}->{}".format(outer_step, confidence_old, c))

    # 返回优化后的对抗样本
    return o_bestattack

    # # 创建保存图像的目录（如果不存在）
    # save_dir = "lossForCW"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #
    # # 绘制 loss2 随迭代次数的变化并保存图像
    # plt.plot(range(len(loss_list)), loss_list, label='Loss over iterations')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.title('Loss vs Iterations')
    # plt.legend()
    #
    # # 保存图像到指定文件夹
    # plt.savefig(os.path.join(save_dir, 'loss_iterations_iter{}.png'.format(iters)))


