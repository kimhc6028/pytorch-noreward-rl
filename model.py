import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):
    
    def __init__(self):
        super(ActorCritic, self).__init__()
        num_inputs = 3
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        #self.lstm = nn.LSTMCell(32 * 3 * 3, 256)

        # See - Q Network defn in https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb
        '''
        self.conv1 = nn.Conv2d(3, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 8, 7, stride=1, padding=1)
        '''
        num_cnn_out = 8*6*6

        self.lstm = nn.LSTMCell(num_cnn_out, 256)

        #num_outputs = 3 # action_space.n
        num_outputs = 4 # action_space.n
        ########################
        self.critic_linear1 = nn.Linear(288, 256)
        self.actor_linear1 = nn.Linear(288, 256)

        self.critic_linear2 = nn.Linear(256, 1)
        self.actor_linear2 = nn.Linear(256, num_outputs)

        ########################
        #self.critic_linear = nn.Linear(256, 1)
        #self.actor_linear = nn.Linear(256, num_outputs)

        ################################################################
        self.icm_conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.icm_conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.icm_conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.icm_conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        '''
        self.icm_conv1 = nn.Conv2d(3, 32, 8, stride=4, padding=1)
        self.icm_conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.icm_conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.icm_conv4 = nn.Conv2d(64, 8, 7, stride=1, padding=1)
        '''
        #self.icm_lstm = nn.LSTMCell(32 * 3 * 3, 256)

        num_cnn_out = 8*6*6

        self.inverse_linear1 = nn.Linear(num_cnn_out + num_cnn_out, 256)
        self.inverse_linear2 = nn.Linear(256, num_outputs)

        self.forward_linear1 = nn.Linear(num_cnn_out + num_outputs, 256)
        self.forward_linear2 = nn.Linear(256, num_cnn_out)


        #self.inverse_linear1 = nn.Linear(256 + 256, 256)
        #self.inverse_linear2 = nn.Linear(256, num_outputs)


        #self.forward_linear1 = nn.Linear(256 + num_outputs, 256)
        #self.forward_linear2 = nn.Linear(256, 256)
        ################################################################
        self.apply(weights_init)
        self.inverse_linear1.weight.data = normalized_columns_initializer(
            self.inverse_linear1.weight.data, 0.01)
        self.inverse_linear1.bias.data.fill_(0)
        self.inverse_linear2.weight.data = normalized_columns_initializer(
            self.inverse_linear2.weight.data, 1.0)
        self.inverse_linear2.bias.data.fill_(0)
        
        self.forward_linear1.weight.data = normalized_columns_initializer(
            self.forward_linear1.weight.data, 0.01)
        self.forward_linear1.bias.data.fill_(0)
        self.forward_linear2.weight.data = normalized_columns_initializer(
            self.forward_linear2.weight.data, 1.0)
        self.forward_linear2.bias.data.fill_(0)


        '''
        self.icm_lstm.bias_ih.data.fill_(0)
        self.icm_lstm.bias_hh.data.fill_(0)
        '''
        ################################################################

        self.actor_linear1.weight.data = normalized_columns_initializer(
            self.actor_linear1.weight.data, 0.01)
        self.actor_linear1.bias.data.fill_(0)
        self.critic_linear1.weight.data = normalized_columns_initializer(
            self.critic_linear1.weight.data, 1.0)
        self.critic_linear1.bias.data.fill_(0)

        self.actor_linear2.weight.data = normalized_columns_initializer(
            self.actor_linear2.weight.data, 0.01)
        self.actor_linear2.bias.data.fill_(0)
        self.critic_linear2.weight.data = normalized_columns_initializer(
            self.critic_linear2.weight.data, 1.0)
        self.critic_linear2.bias.data.fill_(0)


        '''
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        '''
        self.train()


    def forward(self, inputs, icm):

        if icm == False:
            """A3C"""

            inputs = inputs.permute(2,0,1) # permute from (84,84,3) to (3,84,84)
            inputs = inputs.unsqueeze(0).float()

            x = F.elu(self.conv1(inputs))
            x = F.elu(self.conv2(x))
            x = F.elu(self.conv3(x))
            x = F.elu(self.conv4(x))

            num_cnn_out = 8*6*6

            x = x.view(-1, num_cnn_out)
            #a3c_hx, a3c_cx = self.lstm(x, (a3c_hx, a3c_cx))
            #x = a3c_hx

            critic = self.critic_linear1(x)
            actor = self.actor_linear1(x)
            critic = self.critic_linear2(critic)
            actor = self.actor_linear2(actor)
            
            return critic, actor


            '''
            inputs, (a3c_hx, a3c_cx) = inputs

            inputs = inputs.permute(2,0,1) # permute from (84,84,3) to (3,84,84)
            inputs = inputs.unsqueeze(0).float()

            x = F.elu(self.conv1(inputs))
            x = F.elu(self.conv2(x))
            x = F.elu(self.conv3(x))
            x = F.elu(self.conv4(x))

            num_cnn_out = 8*6*6

            x = x.view(-1, num_cnn_out)
            a3c_hx, a3c_cx = self.lstm(x, (a3c_hx, a3c_cx))
            x = a3c_hx

            critic = self.critic_linear(x)
            actor = self.actor_linear(x)
            return critic, actor, (a3c_hx, a3c_cx)
            '''
        else:
            """icm"""
            s_t, s_t1, a_t = inputs
            '''
            s_t, (icm_hx, icm_cx) = s_t
            s_t1, (icm_hx1, icm_cx1) = s_t1
            '''
            s_t = s_t.permute(2,0,1)
            s_t = s_t.unsqueeze(0).float()

            vec_st = F.elu(self.icm_conv1(s_t))
            vec_st = F.elu(self.icm_conv2(vec_st))
            vec_st = F.elu(self.icm_conv3(vec_st))
            vec_st = F.elu(self.icm_conv4(vec_st))

            s_t1 = s_t1.permute(2,0,1)
            s_t1 = s_t1.unsqueeze(0).float()

            vec_st1 = F.elu(self.icm_conv1(s_t1))
            vec_st1 = F.elu(self.icm_conv2(vec_st1))
            vec_st1 = F.elu(self.icm_conv3(vec_st1))
            vec_st1 = F.elu(self.icm_conv4(vec_st1))

            num_cnn_out = 8*6*6

            vec_st = vec_st.view(-1, num_cnn_out)
            vec_st1 = vec_st1.view(-1, num_cnn_out)

            #icm_hx, icm_cx = self.icm_lstm(vec_st, (icm_hx, icm_cx))
            #icm_hx1, icm_cx1 = self.icm_lstm(vec_st1, (icm_hx1, icm_cx1))

            #vec_st = icm_hx
            #vec_st1 = icm_hx1

            inverse_vec = torch.cat((vec_st, vec_st1), 1)
            forward_vec = torch.cat((vec_st, a_t), 1)

            inverse = self.inverse_linear1(inverse_vec)
            inverse = F.relu(inverse)
            inverse = self.inverse_linear2(inverse)
            inverse = F.softmax(inverse)####

            forward = self.forward_linear1(forward_vec)
            forward = F.relu(forward)
            forward = self.forward_linear2(forward)

            return vec_st1, inverse, forward
            #return vec_st1, inverse, forward, (icm_hx, icm_cx), (icm_hx1, icm_cx1)

