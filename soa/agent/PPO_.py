#相比于baseline1,增加了LSTM网络结构
from PIL import Image
from turtle import right
import gym
import sys, os
sys.path.append("../..")
sys.path.append("..")
from gym_minigrid.window import Window
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
import numpy as np
import time
import datetime
from datetime import datetime
import matplotlib.pyplot as plt
from gym_minigrid.minigrid import Goal
from pathlib import Path
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
import torchvision
from agent.net.all_net import Net_PPO_actor,Net_PPO_critic

from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
import argparse
from img_proccess.heatmap import heatmap


#'s'为grid的矩阵表示，障碍物为-0.9，可机动区域为0.9，敌方移动单位为-0.5，agent为0.3
#'a'1维动作信息，上下左右移动以及不动；
#'p'为2维agent坐标状态向量；
#'g'为2维坐标信息
#'r'是1维奖励值信息
                  
class ppo():
    
    def __init__(self):
        self.traindate = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.filepath = Path('/datadisk/yuzhe/predictor/param/PPO/gridim_ppo'+self.traindate+'/')
        self.writer_dir = '/datadisk/yuzhe/tensor/ppo/logs_' + self.traindate
        self.writer = SummaryWriter(log_dir=self.writer_dir)
        self.actor  = Net_PPO_actor()
        self.critic = Net_PPO_critic()

        self.loss_func  = nn.SmoothL1Loss(reduction='mean')
        self.all_params = chain(self.actor.parameters(),self.critic.parameters())
        self.gamma = 0.99
        self.lr = 0.0001
        self.weight_decay = 0.0001
        self.lr_step_size = 200
        self.lr_gamma = 0.8
        self.batch_size = 128
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr, eps=1e-5)
        self.scheduler_actor  = torch.optim.lr_scheduler.StepLR(self.optimizer_actor, step_size=self.lr_step_size, gamma=self.lr_gamma) #学习率衰减
        self.scheduler_critic  = torch.optim.lr_scheduler.StepLR(self.optimizer_critic, step_size=self.lr_step_size, gamma=self.lr_gamma) #学习率衰减

        self.update_count = 0
        self.net_copy_num = 1000
        self.clip_param = 0.1
        self.K_epochs = 10
        self.entropy_coef = 0.01
        self.use_grad_clip = False
        self.use_lr_decay = False
        self.max_steps = 0
        self.heatmapfilename = None
        self.name = None

    def select_action(self, state_matrix, states_stack, goal,device):
        current_states = states_stack[1:5] #选择新动作依据的是新状态，因此选后面四帧，最后一帧为当前环境状态
        current_states = np.array(current_states)
        current_state_matrix = state_matrix[1:5] #选择新动作依据的是新状态，因此选后面四帧，最后一帧为当前环境状态
        current_state_matrix = np.array(current_state_matrix)
        current_state_matrix   = torch.from_numpy(current_state_matrix).float().unsqueeze(0).to(device) #单样本在放进神经网络前需增维
        current_states = torch.from_numpy(current_states).float().unsqueeze(0).to(device)
        goal_tensor    = torch.from_numpy(goal).float().unsqueeze(0).to(device)
        self.actor.eval() #停止网络中的dropout等运算
        self.critic.eval()
        
        with torch.no_grad():       #禁用梯度计算
            a_prob = self.actor(current_state_matrix,current_states,goal_tensor)  
            dist = Categorical(probs=a_prob)
            a = dist.sample()
            a_logprob = dist.log_prob(a)      
        
        action = a.item()
        a_logp = a_logprob.item()          #直接取数值
        return action, a_logp           

    def save_param(self, i_ep,running_score):
        state = {'model_actor':self.actor.state_dict(), 'model_critic':self.critic.state_dict(),'optimizer_actor':self.optimizer_actor.state_dict(), 'optimizer_critic':self.optimizer_critic.state_dict(),'epoch':i_ep}
        if self.filepath.exists():
            torch.save(state, str(self.filepath)+'/' + self.name + '_net_' + str(i_ep) + 'epoch_'+ str(running_score) + 'running_score'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'.pkl')
        else:
            os.makedirs(self.filepath)
            torch.save(state, str(self.filepath)+'/' + self.name + '_net_' + str(i_ep) + 'epoch_'+ str(running_score) + 'running_score'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'.pkl')
                

    def update(self, buffer,device, i_ep):

        s = torch.tensor(buffer['s'], dtype=torch.float32).to(device)
        p = torch.tensor(buffer['p'], dtype=torch.float32).to(device)
        a = torch.tensor(buffer['a'], dtype=torch.int64).to(device)
        g = torch.tensor(buffer['g'], dtype=torch.float32).to(device)
        r = torch.tensor(buffer['r'], dtype=torch.float32).to(device)  
        old_a_logp = torch.tensor(buffer['a_logp'], dtype=torch.float32).to(device).view(-1, 1)
        
        with torch.no_grad(): 
            target_v = r + self.gamma * self.critic(s[:,1:5],p[:,1:5],g)
            adv = target_v - self.critic(s[:,0:4],p[:,0:4],g)
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.actor.train()
        self.critic.train()
        action_loss_i_ep = 0
        value_loss_i_ep  = 0
        for _ in range(self.K_epochs):

            for sample_index in BatchSampler(SubsetRandomSampler(range(len(buffer))), batch_size = self.batch_size, drop_last=False):
                self.actor.train()
                dist = Categorical(probs=self.actor(s[sample_index][:,0:4],p[sample_index][:,0:4],g[sample_index]))
                dist_entropy = dist.entropy().view(-1, 1)
                a_logp = dist.log_prob(a[sample_index].squeeze()).view(-1, 1)   #这里的log_prob是对策略概率密度函数求对数，由于概率密度函数是相乘，所以取对数的和
                ratio = torch.exp(a_logp - old_a_logp[sample_index])

                surr1 = ratio * adv[sample_index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[sample_index]
                action_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy   #取负号的目的是表示梯度上升
                action_loss = action_loss.mean()
                value_loss = F.smooth_l1_loss(self.critic(s[sample_index][:,0:4],p[sample_index][:,0:4],g[sample_index]), target_v[sample_index])
                

                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                action_loss.backward()
                value_loss.backward()
                if self.use_grad_clip:  # Gradient clip
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 0.5)
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_actor.step()
                self.optimizer_critic.step()
                
                self.writer.add_scalar('loss/action_loss_update', action_loss.item(), self.update_count)
                self.writer.add_scalar('loss/value_loss_update', value_loss.item(), self.update_count)
                self.update_count +=1
                if self.update_count % 100 == 0:
                    print("action_loss=", action_loss.item(),"update_count=",self.update_count)
                    print("value_loss=", value_loss.item(),"update_count=",self.update_count) 
                action_loss_i_ep = action_loss.item()
                value_loss_i_ep = value_loss.item()
        self.writer.add_scalar('loss/action_loss_i_ep', action_loss_i_ep, i_ep)
        self.writer.add_scalar('loss/value_loss_i_ep', value_loss_i_ep, i_ep)  
        if self.use_lr_decay:  # learning rate Decay
            self.scheduler_actor.step() 
            self.scheduler_critic.step()
        
        #查看热力图
        heatmap(buffer, self.heatmapfilename,i_ep,device)

          
   