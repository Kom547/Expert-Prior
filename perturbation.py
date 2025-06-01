import torch
import torch.nn as nn
from policy import FniNet, DarrlNet, SumoNet


def PGD(adv_action, victim_agent, last_state, epsilon=0.1, alpha=0.0075,
        device='cuda:1', num_iterations=50):
    if len(last_state.shape) == 1:
        last_state = last_state.unsqueeze(0)
    # print(last_state)
    clamp_min = torch.max((last_state - epsilon), -torch.ones_like(last_state))
    clamp_max = torch.min((last_state + epsilon), torch.ones_like(last_state))

    last_state = last_state.to(last_state.device)
    last_state = last_state + torch.empty_like(last_state).uniform_(-epsilon, epsilon)
    last_state = torch.clamp(last_state, clamp_min, clamp_max)
    if not isinstance(adv_action, torch.Tensor):
        adv_action = torch.tensor(adv_action, dtype=torch.float32, device=device)
    loss = nn.MSELoss()

    ori_state = last_state.data
    cost_list = []
    adv_state_list = []
    for i in range(num_iterations):
        last_state.requires_grad = True
        if isinstance(victim_agent, (FniNet, DarrlNet)):
            # _,_,action_pred = victim_agent(state+perturbation)
            outputs, _, _ = victim_agent(last_state)
            victim_agent.zero_grad()
        elif isinstance(victim_agent, SumoNet):
            outputs = victim_agent(last_state)
            victim_agent.zero_grad()
        else:
            outputs = victim_agent.policy(last_state.unsqueeze(0), deterministic=True)
            # print('action pred is', action_pred)
            if outputs[0].dim() > 1:
                outputs = outputs[0].squeeze(0)
            else:
                outputs = outputs[0]
            victim_agent.policy.zero_grad()

        if adv_action.ndimension() == 0:
            adv_action = adv_action.unsqueeze(0)
        cost = -loss(outputs, adv_action).to(device)
        cost_list.append(cost.item())
        cost.backward()

        adv_state = last_state + alpha * last_state.grad.sign()
        eta = torch.clamp(adv_state - ori_state, min=-epsilon, max=epsilon)
        last_state = torch.clamp(ori_state + eta, min=clamp_min, max=clamp_max).detach_()

    return last_state


def FGSM_v2(adv_action, victim_agent, last_state, epsilon=0.1,
            device='cuda:1', num_iterations=50):
    if len(last_state.shape) == 1:
        last_state = last_state.unsqueeze(0)
    alpha = epsilon / num_iterations
    clamp_min = torch.max((last_state - epsilon), -torch.ones_like(last_state))
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

        if isinstance(victim_agent, (FniNet, DarrlNet)):
            # _,_,action_pred = victim_agent(state+perturbation)
            outputs, _, _ = victim_agent(last_state)
            victim_agent.zero_grad()
        elif isinstance(victim_agent, SumoNet):
            outputs = victim_agent(last_state)
            victim_agent.zero_grad()
        else:
            outputs = victim_agent.policy(last_state.unsqueeze(0), deterministic=True)
            # print('action pred is', action_pred)
            if outputs[0].dim() > 1:
                outputs = outputs[0].squeeze(0)
            else:
                outputs = outputs[0]
            victim_agent.policy.zero_grad()

        # print('o,a',outputs,adv_action)
        if adv_action.ndimension() == 0:
            adv_action = adv_action.unsqueeze(0)
        cost = -loss(outputs, adv_action).to(device)
        # print(cost)
        cost_list.append(cost.item())
        cost.backward()

        last_state = torch.clamp(last_state + alpha * last_state.grad.sign(), min=clamp_min, max=clamp_max).detach_()
    return last_state
