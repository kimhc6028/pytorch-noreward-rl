from __future__ import print_function

import argparse
import os
import sys
import cv2

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
#from envs import create_atari_env
import env_wrapper
from model import ActorCritic
from train import train
from test import test
import my_optim

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=10000, metavar='M',
                    help='maximum length of an episode (default: 10000)')
parser.add_argument('--env-name', default='PongDeterministic-v3', metavar='ENV',
                    help='environment to train on (default: PongDeterministic-v3)')
parser.add_argument('--no-shared', default=False, metavar='O',
                    help='use an optimizer without shared momentum.')
####################
parser.add_argument('--eta', type=float, default=0.01, metavar='LR',
                    help='scaling factor for intrinsic reward')
parser.add_argument('--beta', type=float, default=0.2, metavar='LR',
                    help='balance between inverse & forward')
parser.add_argument('--lmbda', type=float, default=0.1, metavar='LR',
                    help='lambda : balance between A3C & icm')

parser.add_argument('--outdir', default="../output", help='Output log directory')
parser.add_argument('--record', action='store_true', help="Record the policy running video")



if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    #env = create_atari_env(args.env_name)
    env = env_wrapper.create_doom(args.record, outdir=args.outdir)
    shared_model = ActorCritic(
        env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
