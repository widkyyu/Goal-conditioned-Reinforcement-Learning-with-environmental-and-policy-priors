

import sys, os
import numpy as np
from numpy import savetxt
import time
from tqdm import tqdm
import datetime
from datetime import datetime
from pathlib import Path
from itertools import chain
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from agent.net.all_net import Net_Encoder, Net_Decoder, LSTM

#'s'为grid的矩阵表示，障碍物为-0.9，可机动区域为0.9，敌方移动单位为-0.5，agent为0.3
#'a'1维动作信息，上下左右移动以及不动；
#'p'为2维agent坐标状态向量；
#'g'为2维坐标信息
#'r'是1维奖励值信息
                  
class encoder_lstm_decoder():
    
    def __init__(self):
        self.traindate = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.filepath = Path('/media/yuzhe/yuzhe/predictor/param/ppo_encoder_decoder_predictor/gridim_pre'+self.traindate+'/')
        self.writer_dir = '/media/yuzhe/yuzhe/tensor/predictor/logs_' + self.traindate
        self.writer = SummaryWriter(log_dir=self.writer_dir)

        self.en_de_filepath = Path('/media/yuzhe/yuzhe/predictor/param/ppo_encoder_decoder/gridim_encoder_decoder'+self.traindate+'/')
        self.en_de_writer_dir = '/media/yuzhe/yuzhe/tensor/encoder_decoder/logs_' + self.traindate
        self.en_de_writer = SummaryWriter(log_dir=self.en_de_writer_dir)
        
        self.encoder = Net_Encoder()
        self.decoder = Net_Decoder()
        self.predictor = LSTM()

        self.loss_func  = nn.MSELoss(reduction='none')  #nn.SmoothL1Loss(reduction='none')   nn.MSELoss(reduction='none')
        self.all_params = chain(self.encoder.parameters(),self.decoder.parameters(),self.predictor.parameters())
        self.gamma = 0.99
        self.lr = 0.0001
        self.encoder_lr = 1e-08
        self.decoder_lr = 1e-08
        self.predictor_lr = 1e-08
        self.weight_decay = 0.0001
        self.lr_step_size = 1
        self.lr_gamma = 0.9
        self.batch_size = 128
        self.num_workers = 15
        
        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(), lr=self.encoder_lr, betas=(0.9,0.98), eps=1e-09)
        self.optimizer_decoder = torch.optim.Adam(self.decoder.parameters(), lr=self.decoder_lr, betas=(0.9,0.98), eps=1e-09)
        self.optimizer_predictor = torch.optim.Adam(self.predictor.parameters(), lr=self.predictor_lr, betas=(0.9,0.98), eps=1e-09)

        self.scheduler_encoder = torch.optim.lr_scheduler.StepLR(self.optimizer_encoder, step_size=self.lr_step_size, gamma=self.lr_gamma)
        self.scheduler_decoder = torch.optim.lr_scheduler.StepLR(self.optimizer_decoder, step_size=self.lr_step_size, gamma=self.lr_gamma)
        self.scheduler_predictor = torch.optim.lr_scheduler.StepLR(self.optimizer_predictor, step_size=self.lr_step_size, gamma=self.lr_gamma)

        self.num_episodes_en_de = 5
        self.num_episodes_pre = 5
        self.use_grad_clip = False
        self.use_lr_decay = False
        self.max_steps = 0
        self.train_update_number_en_de = 0
        self.val_number_en_de = 0
        self.train_update_number_pre = 0
        self.val_number_pre = 0
        self.seed = 0
        self.en_de_average_score = 0
        self.pre_average_score = 0
       
        self.name = None
    
    def save_param(self, i_ep):
        state = {'model_encoder':self.encoder.state_dict(),'model_decoder':self.decoder.state_dict(),'model_predictor':self.predictor.state_dict(),'optimizer_encoder':self.encoder.state_dict(),'optimizer_decoder':self.decoder.state_dict(),'optimizer_predictor':self.predictor.state_dict(),'epoch':i_ep}
        if self.filepath.exists():
            torch.save(state, str(self.filepath)+'/' + self.name + '_pre_net_'+str(self.pre_average_score) +'average_score_' + str(i_ep) + 'epoch_'+ str(self.seed) + 'seed_'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'.pkl')
        else:
            os.makedirs(self.filepath)
            torch.save(state, str(self.filepath)+'/' + self.name + '_pre_net_'+str(self.pre_average_score) +'average_score_' + str(i_ep) + 'epoch_'+ str(self.seed) + 'seed_'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'.pkl')
    
    def save_param_encoder_decoder(self, i_ep):
        seed = self.seed
        state = {'model_encoder':self.encoder.state_dict(),'model_decoder':self.decoder.state_dict(),'optimizer_encoder':self.optimizer_encoder.state_dict(),'optimizer_decoder':self.optimizer_decoder.state_dict(),'epoch':i_ep}
        if self.en_de_filepath.exists():
            torch.save(state, str(self.en_de_filepath)+'/' + self.name + '_net_'+str(self.en_de_average_score) +'average_score_' + str(i_ep) + 'epoch_'+ str(seed) + 'seed_'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'.pkl')
        else:
            os.makedirs(self.en_de_filepath)
            torch.save(state, str(self.en_de_filepath)+'/' + self.name + '_net_'+str(self.en_de_average_score) +'average_score_' + str(i_ep) + 'epoch_'+ str(seed) + 'seed_'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'.pkl')
                

    def update_encoder_decoder(self, buffer,device):

        pre_buffer = buffer['s'][:,4].reshape(-1,1,289)

        s_train, s_val =  train_test_split(pre_buffer, test_size = 0.1, random_state = 1)
        s_train_tensor = torch.tensor(s_train)
        s_val_tensor = torch.tensor(s_val)
        train_dataset = TensorDataset(s_train_tensor)     #用于训练的数据集
        val_dataset   = TensorDataset(s_val_tensor)
        val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=self.batch_size, shuffle=True,
        num_workers=self.num_workers)
        print('batch_size',self.batch_size)
        train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=self.batch_size, shuffle=True,
        num_workers=self.num_workers)    #设置为True时会在每个epoch重新打乱数据(默认: False)

        #predictor更新时，做了梯度静止，这里重新恢复梯度
        for p in self.encoder.parameters():
            p.requires_grad = True
        for p in self.decoder.parameters():
            p.requires_grad = True
        
        train_loss = 0
        val_loss = 0
        average_loss = 0
        self.encoder.train()
        self.decoder.train()
        
        for i_ep in tqdm(range(self.num_episodes_en_de)):#range(args.num_episodes)
            train_loss_per_epoch = []
            idx = 0
            loss = 0
            for i, (states) in enumerate(train_loader): 
                states = states[0].to(device)
                z , z_upsample= self.encoder(states[:,:])
                states_head, states_head_pool = self.decoder(z)

                #查看数据结果
                row_result = np.array(states[1].view(17,17).cpu().detach(),dtype=float)
                encoder_decoder_result = np.array(states_head[1].view(17,17).cpu().detach(),dtype=float)
                # row_result = np.around(row_result, 1)
                # encoder_decoder_result = np.around(encoder_decoder_result, 1)
                savetxt('/home/yuzhe/gym-minigrid_1216/rrl_cnn/low_level/result/encoder_decoder/encoder_decoder_'+'data.csv', encoder_decoder_result, delimiter=',')
                savetxt('/home/yuzhe/gym-minigrid_1216/rrl_cnn/low_level/result/encoder_decoder/row_'+'data.csv', row_result, delimiter=',')
                
                z_t_recon_loss = self.loss_func(z_upsample, states_head_pool).mean((2,3,4)) #2,3维取均值
                loss =  z_t_recon_loss
                loss = loss.mean()
                loss = loss
                self.optimizer_encoder.zero_grad()
                self.optimizer_decoder.zero_grad()
                loss.backward()
                self.optimizer_encoder.step()
                self.optimizer_decoder.step()
                train_loss_per_epoch.append(loss.item())
                train_loss = loss.item()
                self.en_de_writer.add_scalar('loss/en_de_train_loss_update', train_loss, self.train_update_number_en_de)
                self.train_update_number_en_de += 1
                if i%100==99:
                    print('en_de_trani_loss100:',train_loss)
                    average_loss = average_loss * 0.99 + train_loss * 0.01 

            print('en_de_train_loss:',train_loss)
            
            if val_loader is not None:
                val_loss_per_epoch = 0
                self.encoder.eval()
                self.decoder.eval()
                val_loss_per_epoch = []
                logs = {}
                with torch.no_grad():
                    for idx, (states) in enumerate(val_loader):
                        states = states[0].to(device)
                        z , z_upsample= self.encoder(states[:,:])
                        states_head, states_head_pool = self.decoder(z)
                        z_t_recon_loss = self.loss_func(z_upsample, states_head_pool).mean((2,3,4)) #2,3维取均值
                        loss = z_t_recon_loss
                        loss = loss.mean()
                        val_loss_per_epoch.append(loss.item())
                        val_loss = loss.item()
                        self.en_de_writer.add_scalar('loss/en_de_value_loss_update', val_loss, self.val_number_en_de)
                        self.val_number_en_de += 1        
            print('en_de_val_loss:',val_loss)
            self.scheduler_encoder.step()
            self.scheduler_decoder.step()

            if (i_ep +1) % 2 == 0:
                print('save net')
                self.en_de_average_score = average_loss
                self.save_param_encoder_decoder(i_ep)

    def update_predictor(self, buffer,device):

        pre_buffer = buffer['s']

        s_train, s_val =  train_test_split(pre_buffer, test_size = 0.1, random_state = 1)
        s_train_tensor = torch.tensor(s_train)
        s_val_tensor = torch.tensor(s_val)
        train_dataset = TensorDataset(s_train_tensor)     #用于训练的数据集
        val_dataset   = TensorDataset(s_val_tensor)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers)    #设置为True时会在每个epoch重新打乱数据(默认: False)
        
        #Keep encoder and decoder static
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

        train_loss = 0
        val_loss = 0
        i_ep = 0
        grad_all = True #是否预测的四帧同时求梯度，否的话是每一帧做一次梯度更新
        for i_ep in tqdm(range(self.num_episodes_pre)):#range(args.num_episodes)
            train_loss_per_epoch = []
            loss = 0
            pre_average_loss = 0
            self.predictor.train()
            for i, (states) in enumerate(train_loader): 
                states = states[0].to(device)
                states_pre = states.view(-1,1,289)
                z_c, z_c_upsample = self.encoder(states_pre)
                z_c = z_c.view(-1,9,64,4,4) #每个样本中有9个steps
                z_theme_pred,_ = self.predictor(z_c[:,:4].detach()) 

                states_head, states_head_pool = self.decoder(z_theme_pred[:,3:7])

                #查看数据结果
                row_result = np.array(states[0,4].view(17,17).cpu().detach(),dtype=float)
                encoder_decoder_result = np.array(states_head[0,0].view(17,17).cpu().detach(),dtype=float)
                # row_result = np.around(row_result, 1)
                # encoder_decoder_result = np.around(encoder_decoder_result, 1)
                savetxt('/home/yuzhe/gym-minigrid_1216/rrl_cnn/low_level/result/predictor/pre_'+'data.csv', encoder_decoder_result, delimiter=',')
                savetxt('/home/yuzhe/gym-minigrid_1216/rrl_cnn/low_level/result/predictor/row_'+'data.csv', row_result, delimiter=',')

                if grad_all:
                    z_t_predictor_loss = self.loss_func(z_c_upsample.reshape(-1,9,1,68,68)[:,4:8], states_head_pool).mean((2,3,4)) #(2,4,1,68,68)
                    # z_t_predictor_loss = l1_loss(z_c[:,4:8], z_theme_pred[:,3:7]).mean((2,3,4)) #(2,4,1,68,68)
                    loss =  z_t_predictor_loss
                    loss = loss.mean()
                    self.optimizer_predictor.zero_grad()
                    loss.backward()
                    self.optimizer_predictor.step()
                else:
                    loss_4step = []
                    for i in range(4):
                        states_head_pool2 = torch.tensor(states_head_pool, requires_grad=True)
                        z_t_predictor_loss = self.loss_func(z_c_upsample.reshape(-1,9,1,68,68)[:,4+i:5+i], states_head_pool2[:,i:1+i]).mean((2,3,4)) #(2,4,1,68,68)
                        loss =  z_t_predictor_loss
                        loss = loss.mean()
                        self.optimizer_predictor.zero_grad()
                        loss.backward()
                        self.optimizer_predictor.step()
                        loss_4step.append([loss])
                    # loss = np.array(loss_4step).mean().item()
                    print('loss',loss)
                train_loss_per_epoch.append(loss.item())
                train_loss = loss.item()
                self.writer.add_scalar('loss/pre_train_loss_update', train_loss, self.train_update_number_pre)
                self.train_update_number_pre += 1
                if i %100 == 99:
                    print('pre_train_loss100:',train_loss)
                    pre_average_loss = pre_average_loss * 0.99 + train_loss * 0.01
            print('pre_train_loss:',train_loss)
            if val_loader is not None:
                val_loss_per_epoch = 0
                self.predictor.eval()
                val_loss_per_epoch = []
                logs = {}
                with torch.no_grad():
                    for idx, (states) in enumerate(val_loader):
                        states = states[0].to(device)
                        states_pre = states.view(-1,1,289)
                        z_c, z_c_upsample = self.encoder(states_pre)
                        z_c = z_c.view(-1,9,64,4,4)
                        z_theme_pred,_ = self.predictor(z_c[:,:4].detach())  
                        states_head, states_head_pool = self.decoder(z_theme_pred[:,3:7])
                        z_t_predictor_loss = self.loss_func(z_c_upsample.reshape(-1,9,1,68,68)[:,4:8], states_head_pool).mean((2,3,4)) #(2,4,1,68,68) 

                        loss =  z_t_predictor_loss
                        loss = loss.mean()
                        val_loss = loss.item()
                        self.writer.add_scalar('loss/pre_value_loss_update', val_loss, self.val_number_pre)
                        self.val_number_pre += 1
            print('pre_val_loss:',val_loss)
            if i_ep >1:
                self.scheduler_predictor.step()
                if i_ep  % 2 == 0:
                    print('save net')
                    self.pre_average_score = pre_average_loss
                    self.save_param(i_ep)
                
                
                   
        
                 
   