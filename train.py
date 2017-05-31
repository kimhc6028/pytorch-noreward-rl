import math
import os
import sys

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import env_wrapper
from model import ActorCritic
from torch.autograd import Variable
from torchvision import datasets, transforms

import time

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, optimizer=None):
    
    mse_loss = torch.nn.MSELoss()
    nll_loss = torch.nn.NLLLoss()

    torch.manual_seed(args.seed + rank)

    env = env_wrapper.create_doom(args.record, outdir=args.outdir)
    num_outputs = env.action_space.n
    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    episode_length = 0

    while True:
        episode_length += 1
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)
        values = []
        log_probs = []
        rewards = []
        entropies = []

        inverses = []
        forwards = []
        actions = []
        vec_st1s = []

        for step in range(args.num_steps):
            value, logit, (hx, cx) = model(
                (Variable(state.unsqueeze(0)), (hx, cx)),
                icm = False
            )
            s_t = state
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))

            oh_action = torch.Tensor(1, num_outputs)
            oh_action.zero_()
            oh_action.scatter_(1,action,1)
            oh_action = Variable(oh_action)
            a_t = oh_action
            actions.append(oh_action)

            state, reward, done, _ = env.step(action.numpy()[0][0])
            state = torch.from_numpy(state)

            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)
            s_t1 = state
            vec_st1, inverse, forward = model(
                (
                    Variable(s_t.unsqueeze(0)),
                    Variable(s_t1.unsqueeze(0)),
                    a_t
                ),
                icm = True
            )            

            reward_intrinsic = args.eta * ((vec_st1 - forward).pow(2)).sum(1) / 2.
            #reward_intrinsic = args.eta * ((vec_st1 - forward).pow(2)).sum(1).sqrt() / 2.
            reward_intrinsic = reward_intrinsic.data.numpy()[0][0]
            reward += reward_intrinsic

            if done:
                episode_length = 0
                state = env.reset()
                state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            vec_st1s.append(vec_st1)
            inverses.append(inverse)
            forwards.append(forward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model(
                (Variable(state.unsqueeze(0)), (hx, cx)),
                icm = False
            )
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        inverse_loss = 0
        forward_loss = 0

        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - 0.01 * entropies[i]
            
            cross_entropy = - (actions[i] * torch.log(inverses[i] + 1e-15)).sum(1)
            inverse_loss = inverse_loss + cross_entropy
            forward_err = forwards[i] - vec_st1s[i]
            forward_loss = forward_loss + 0.5 * (forward_err.pow(2)).sum(1)


        optimizer.zero_grad()

        ((1-args.beta) * inverse_loss + args.beta * forward_loss).backward(retain_variables=True)
        (args.lmbda * (policy_loss + 0.5 * value_loss)).backward()

        #(((1-args.beta) * inverse_loss + args.beta * forward_loss) + args.lmbda * (policy_loss + 0.5 * value_loss)).backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), 40)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
