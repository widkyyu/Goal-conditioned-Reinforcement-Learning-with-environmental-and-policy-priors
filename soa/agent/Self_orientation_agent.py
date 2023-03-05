
from PIL import Image
from turtle import right
import gym
import sys, os
from numpy import savetxt
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
from agent.net.all_net import Net_Encoder, Net_Decoder,LSTM,Net_SoA_actor, Net_SoA_critic,Net_SoA_orient
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
import argparse
from img_proccess.heatmap import heatmap


#'s'为grid的矩阵表示，障碍物为-0.9，可机动区域为0.9，敌方移动单位为-0.5，agent为0.3
#'a'1维动作信息，上下左右移动以及不动；
#'p'为2维agent坐标状态向量；
#'g'为2维坐标信息
#'r'是1维奖励值信息
                  
class self_orinetation_agent():
    
    def __init__(self):
        self.traindate = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.filepath = Path('/datadisk/yuzhe/predictor/param/SoA/gridim_soa_'+self.traindate+'/')
        self.writer_dir = '/datadisk/yuzhe/tensor/SoA/logs_' + self.traindate
        self.writer = SummaryWriter(log_dir=self.writer_dir)
        self.actor  = Net_SoA_actor()
        self.critic = Net_SoA_critic()
        self.agent_position_preditor = Net_SoA_orient()
        self.encoder = Net_Encoder()
        self.decoder = Net_Decoder()
        self.predictor = LSTM()

        self.loss_func  = nn.SmoothL1Loss(reduction='mean')
        self.all_params = chain(self.actor.parameters(),self.critic.parameters())
        self.gamma = 0.99
        self.lr = 0.0001
        self.encoder_lr = 0.00001
        self.decoder_lr = 0.00001
        self.predictor_lr = 0.00001
        self.weight_decay = 0.0001
        self.lr_step_size = 200
        self.lr_gamma = 0.8
        self.lr_gamma_pre_agent = 0.8
        self.batch_size = 128
        self.batch_size_pre_agent= 128
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr, eps=1e-5)
        self.optimizer_agent_position_preditor = torch.optim.Adam(self.agent_position_preditor.parameters(), lr=self.lr, eps=1e-5)

        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(), lr=self.encoder_lr, betas=(0.9,0.98), eps=1e-09)
        self.optimizer_decoder = torch.optim.Adam(self.decoder.parameters(), lr=self.decoder_lr, betas=(0.9,0.98), eps=1e-09)
        self.optimizer_predictor = torch.optim.Adam(self.predictor.parameters(), lr=self.predictor_lr, betas=(0.9,0.98), eps=1e-09)


        self.scheduler_actor  = torch.optim.lr_scheduler.StepLR(self.optimizer_actor, step_size=self.lr_step_size, gamma=self.lr_gamma) #学习率衰减
        self.scheduler_critic  = torch.optim.lr_scheduler.StepLR(self.optimizer_critic, step_size=self.lr_step_size, gamma=self.lr_gamma) #学习率衰减
        self.scheduler_agent_position_preditor  = torch.optim.lr_scheduler.StepLR(self.optimizer_agent_position_preditor, step_size=5*self.lr_step_size, gamma=self.lr_gamma) #学习率衰减
        
        self.update_count = 0
        self.update_count_fp = 0
        self.net_copy_num = 1000
        self.clip_param = 0.1
        self.K_epochs = 10
        self.K_epochs_pre_agent_position = 50
        self.entropy_coef = 0.01
        self.use_grad_clip = False
        self.use_lr_decay = False
        self.max_steps = 0
        self.heatmapfilename = None
        self.future3positionfilename = None
       
        self.name = None
    
    def pred_states(self, state_matrix):
        states_pre = state_matrix.reshape(-1,1,289)
        self.encoder.eval()
        self.decoder.eval()
        self.predictor.eval()
        with torch.no_grad():
            z_c, z_c_upsample = self.encoder(states_pre)
            z_c = z_c.view(-1,4,64,4,4)
            z_theme_pred,_ = self.predictor(z_c) 
            z_theme_pred_4frame = z_theme_pred[:,3:7]
            states_head, states_head_pool = self.decoder(z_theme_pred_4frame)
        return  states_head, z_c_upsample, states_head_pool

    def select_action(self, state_matrix, states_stack, goal,device):
        current_states = states_stack[1:5] #选择新动作依据的是新状态，因此选后面四帧，最后一帧为当前环境状态
        current_states = np.array(current_states)
        current_state_matrix = state_matrix[1:5] #选择新动作依据的是新状态，因此选后面四帧，最后一帧为当前环境状态
        current_state_matrix = np.array(current_state_matrix)
        current_state_matrix   = torch.from_numpy(current_state_matrix).float().unsqueeze(0) #单样本在放进神经网络前需增维
        current_states = torch.from_numpy(current_states).float().unsqueeze(0)
        goal_tensor    = torch.from_numpy(goal).float().unsqueeze(0)
        self.actor.eval() #停止网络中的dropout等运算
        self.critic.eval() #停止网络中的dropout等运算
        self.agent_position_preditor.eval()
        
        with torch.no_grad():       #禁用梯度计算
            #数据类型转换
            current_state_matrix,current_states,goal_tensor = torch.FloatTensor(current_state_matrix).to(device),torch.FloatTensor(current_states).to(device),torch.FloatTensor(goal_tensor).to(device)
            #预测后4帧外界环境状态
            prediction_state_matrix, _, _ = self.pred_states(current_state_matrix)
            #当前及预测后的进行叠加
            cat_state_matrix = torch.cat([current_state_matrix, prediction_state_matrix.detach()], 1)

            #预测第3帧agent的位置
            Px_prob, Py_prob = self.agent_position_preditor(cat_state_matrix,current_states,goal_tensor)
            dist_x, dist_y = Categorical(probs=Px_prob), Categorical(probs=Py_prob)
            Px, Py = (dist_x.sample()-3).unsqueeze(-1), (dist_y.sample()-3).unsqueeze(-1)
            
            future_third_step_position = torch.cat([Px, Py], 1)
            cat_goal = torch.cat([goal_tensor, future_third_step_position.detach()], 1)
            #动作策略
            a_prob = self.actor(cat_state_matrix,current_states,cat_goal)  
            dist = Categorical(probs=a_prob)
            a = dist.sample()
            a_logprob = dist.log_prob(a)      
        
        action = a.item()
        a_logp = a_logprob.item()          #直接取数值
        future_x = Px.item() 
        future_y = Py.item() 
        return action, a_logp, future_x, future_y          

    def save_param(self, i_ep,running_score):
        state = {'model_actor':self.actor.state_dict(), 'model_critic':self.critic.state_dict(),'model_encoder':self.encoder.state_dict(),'model_decoder':self.decoder.state_dict(),'model_predictor':self.predictor.state_dict(),'model_agent_position_preditor':self.agent_position_preditor.state_dict(),'optimizer_actor':self.optimizer_actor.state_dict(), 'optimizer_critic':self.optimizer_critic.state_dict(),'optimizer_encoder':self.optimizer_encoder.state_dict(),'optimizer_decoder':self.optimizer_decoder.state_dict(),'optimizer_predictor':self.optimizer_predictor.state_dict(),'optimizer_agent_position_preditor':self.optimizer_agent_position_preditor.state_dict(),'epoch':i_ep}

        if self.filepath.exists():
            torch.save(state, str(self.filepath)+'/' + self.name + '_net_' + str(i_ep) + 'epoch_'+ str(running_score) + 'running_score'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'.pkl')
        else:
            os.makedirs(self.filepath)
            torch.save(state, str(self.filepath)+'/' + self.name + '_net_' + str(i_ep) + 'epoch_'+ str(running_score) + 'running_score'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'.pkl')
    
    def update_policy(self, buffer,device,i_ep):
        
        s = torch.tensor(buffer['s'], dtype=torch.float32).to(device)
        p = torch.tensor(buffer['p'], dtype=torch.float32).to(device)
        a = torch.tensor(buffer['a'], dtype=torch.int64).to(device)
        g = torch.tensor(buffer['g'], dtype=torch.float32).to(device)
        r = torch.tensor(buffer['r'], dtype=torch.float32).to(device)  
        old_a_logp = torch.tensor(buffer['a_logp'], dtype=torch.float32).to(device)
        f = torch.tensor(buffer['f'], dtype=torch.float32).to(device)
        with torch.no_grad(): 
            #环境状态预测
            prediction_state_matrix_, _, _ = self.pred_states(s[:,1:5])
            cat_state_matrix_ = torch.cat([s[:,1:5], prediction_state_matrix_.detach()], 1)
            #含预测第3帧agent的位置
            cat_goal_ = torch.cat([g, f[:,1]], 1)
            #计算目标状态值
            target_v = r[:,0] + self.gamma * self.critic(cat_state_matrix_,p[:,1:5],cat_goal_)

            #环境状态预测
            prediction_state_matrix, _, _ = self.pred_states(s[:,:4])
            cat_state_matrix = torch.cat([s[:,:4], prediction_state_matrix.detach()], 1)
            #含agent预测状态
            cat_goal = torch.cat([g, f[:,0]], 1)
            #优势函数值
            adv = target_v - self.critic(cat_state_matrix,p[:,:4],cat_goal)
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # #计算预测位置与实际位置的损失值
            # states_error = torch.sub(input=p[:,6], alpha=1, other=p[:,3])

        self.actor.train()
        self.critic.train()
        self.agent_position_preditor.train()
        action_loss_i_ep = 0
        value_loss_i_ep  = 0
        for _ in range(self.K_epochs):
            
            for sample_index in BatchSampler(SubsetRandomSampler(range(len(buffer))), batch_size = self.batch_size, drop_last=False):
                
                #计算action损失值
                dist = Categorical(probs=self.actor(cat_state_matrix[sample_index],p[sample_index][:,0:4],cat_goal[sample_index]))
                dist_entropy = dist.entropy().view(-1, 1)
                a_logp = dist.log_prob(a[sample_index][:,0].squeeze()).view(-1, 1)   #这里的log_prob是对策略概率密度函数求对数，由于概率密度函数是相乘，所以取对数的和
                ratio = torch.exp(a_logp - old_a_logp[sample_index][:,0])
                surr1 = ratio * adv[sample_index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[sample_index]
                action_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy   #取负号的目的是表示梯度上升
                action_loss = action_loss.mean()

                #计算value损失值
                value_loss = F.smooth_l1_loss(self.critic(cat_state_matrix[sample_index],p[sample_index][:,0:4],cat_goal[sample_index]), target_v[sample_index])
                
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                
                action_loss.backward()
                value_loss.backward()

                if self.use_grad_clip:  #  Gradient clip
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 0.5)
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 0.5)
                    
                self.optimizer_actor.step()
                self.optimizer_critic.step()
                # self.optimizer_encoder.step()
                
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
        
                 
    def update_orientation(self, buffer,device,i_ep):
        
        s = torch.tensor(buffer['s'], dtype=torch.float32).to(device)
        p = torch.tensor(buffer['p'], dtype=torch.float32).to(device)
        g = torch.tensor(buffer['g'], dtype=torch.float32).to(device)
        f = torch.tensor(buffer['f'], dtype=torch.float32).to(device)
        with torch.no_grad(): 

            #环境状态预测
            prediction_state_matrix, _, _ = self.pred_states(s[:,:4])
            cat_state_matrix = torch.cat([s[:,:4], prediction_state_matrix.detach()], 1)
            
            #计算预测位置与实际位置的差值
            states_error = torch.sub(input=p[:,6], alpha=1, other=p[:,3])
        
        self.agent_position_preditor.train()
        future_3steps_loss_i_ep = 0
        
        for _ in range(self.K_epochs_pre_agent_position):
            
            for sample_index in BatchSampler(SubsetRandomSampler(range(len(buffer))), batch_size = self.batch_size_pre_agent, drop_last=False):
                
                #计算预测位置与实际位置的损失值
                Px_prob, Py_prob = self.agent_position_preditor(cat_state_matrix[sample_index],p[sample_index][:,0:4],g[sample_index])
                dist_x, dist_y = Categorical(probs=Px_prob), Categorical(probs=Py_prob)

                Px_logpx_loss = dist_x.log_prob(states_error[sample_index][:,0]+3).view(-1, 1)
                Py_logpy_loss = dist_y.log_prob(states_error[sample_index][:,1]+3).view(-1, 1)
                future_3steps_loss = -Px_logpx_loss -Py_logpy_loss
                future_3steps_loss = future_3steps_loss.mean()
                
                self.optimizer_agent_position_preditor.zero_grad()
                               
                future_3steps_loss.backward()
                if self.use_grad_clip:  #  Gradient clip
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.agent_position_preditor.parameters(), 0.5)
                    
                self.optimizer_agent_position_preditor.step()
                
                self.writer.add_scalar('loss/future_3steps_loss_update', future_3steps_loss.item(), self.update_count_fp)
                self.update_count_fp +=1
                if self.update_count_fp % 100 == 0: 
                    print("future_3steps_loss=", future_3steps_loss.item(),"update_count_fp=",self.update_count_fp) 
                future_3steps_loss_i_ep = future_3steps_loss.item()
                
        self.writer.add_scalar('loss/future_3steps_loss_i_ep', future_3steps_loss_i_ep, i_ep)
        
        if self.use_lr_decay:  # learning rate Decay
            
            self.scheduler_agent_position_preditor.step()
        
        #查看数据结果
        result = torch.cat([p[:,3], g, f[:,0],p[:,6]], 1)
        result = np.array(result.cpu().detach(), dtype=int)
        savetxt(self.future3positionfilename + 'data.csv', result, delimiter=',')
