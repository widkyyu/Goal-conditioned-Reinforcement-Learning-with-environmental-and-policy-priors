#ppo+predictor_cat

from pathlib import Path
import gym
import sys, os
from gym_minigrid.window import Window
import numpy as np
import random
import time
import datetime
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torchvision

from agent.PPO_Predictor import ppo_predictor

from env_buffer import Buffer_gridworld, Env_transact
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="gym environment to load", default="MiniGrid-twoarmy-17x17-v4")
    parser.add_argument("--seed",type=int,help="random seed to generate the environment with",default=9981)
    parser.add_argument("--tile_size", type=int, help="size at which to render tiles", default=17)
    parser.add_argument("--batch_size", type=int, help="size at which to sample", default=128)
    parser.add_argument("--her",default=True,help="hindlight experience replay")
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay (default: 0.0001)')
    parser.add_argument('--lr_gamma', type=float, default=0.8, metavar='LG', help='learning rate discount factor (default: 0.8)')
    parser.add_argument('--lr_step_size', type=int, default=200, help='each step_size perform learning rate discount factor (default: 200)')
    parser.add_argument('--num_episodes', type=int, default=1000000, help='number all episodes (default: 100000)')
    parser.add_argument('--max_steps', type=int, default=50, help='number of max steps in environment (default: 100)')
    parser.add_argument("--predictor_net_file", help="net to load", default='/media/yuzhe/yuzhe/predictor/param/ppo_encoder_decoder_predictor/gridim_pre2023_02_07_17_51_33/MiniGrid-twoarmy-17x17-v2_predictor__pre_net_0.001866217129343992average_score_16epoch_6667seed_2023_02_07_19_25_30.pkl')
    parser.add_argument("--track_buffer_file", help="net to load", default='/datadisk/yuzhe/predictor/track_buffer/ppo_pre')
    parser.add_argument("--log_dir", help="model to load", default='/datadisk/yuzhe/predictor/param/')
    parser.add_argument("--cuda", help="buffer to load", default="cuda:7")
    parser.add_argument("--server", help="if running in the server", default=True)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    if args.server:
        device = torch.device(args.cuda if use_cuda else "cpu")
    else:
        device = torch.device("cuda:0" if use_cuda else "cpu")
    
    torch.set_num_threads(500)

    seed = None if args.seed == -1 else args.seed  
    random.seed(seed)     #
    np.random.seed(seed)  #
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)            #
    if use_cuda:
        torch.cuda.manual_seed(seed)       #
        torch.cuda.manual_seed_all(seed)                

    steps = 0
    action = None
    num_terminate = 0
    num_10epo = 0
    epochs = 0

    agent = ppo_predictor()
    agent.name = 'ppo_predictor'+'_'+args.env+'_'+str(seed) +'seed_' 
    agent.heatmapfilename = 'ppo_predictor'+'_'+ args.env +'_'+ str(seed) + 'seed_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    agent.gamma = args.gamma
    agent.lr = args.lr
    agent.weight_decay = args.weight_decay
    agent.lr_step_size = args.lr_step_size
    agent.lr_gamma = args.lr_gamma
    agent.actor.to(device)       
    agent.critic.to(device)
    agent.encoder.to(device)
    agent.encoder.device = device
    agent.decoder.to(device)
    agent.encoder.load_state_dict(torch.load(args.predictor_net_file,map_location=args.cuda)['model_encoder'])
    agent.decoder.load_state_dict(torch.load(args.predictor_net_file,map_location=args.cuda)['model_decoder'])
    agent.predictor.to(device)
    agent.predictor.device = device
    agent.predictor.load_state_dict(torch.load(args.predictor_net_file,map_location=args.cuda)['model_predictor'])
    agent.batch_size = args.batch_size
    agent.max_steps = args.max_steps
    #储存agent点轨迹的路径
    track_filepath = Path(args.track_buffer_file + agent.name + str(seed) +'seed'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'/')

    env = gym.make(
        args.env,
        seed=seed,
        new_step_api=True,
        tile_size=args.tile_size,
    )

    window = Window("gym_minigrid - " + args.env)

    buffer = Buffer_gridworld()
    envgrid = env.grid
    buffer.grid_size = envgrid.height
    
    buffer.buffer_pre_capacity = 2048
    buffer.pre_transition = np.dtype([ ('s', np.float64, (9,buffer.grid_size**2)),('a', np.int64, (5,1)), 
        ('p', np.float64, (9,2)),  ('g', np.float64, (2,)),  ('r', np.float64, (5,1)),  
        ('d', np.int64, (5,1)), ('a_logp', np.float64, (5,1))])
    buffer.pre_buffer = np.empty(buffer.buffer_pre_capacity, dtype=buffer.pre_transition)

    running_score = 0
    
    for i_ep in range(args.num_episodes):#range(args.num_episodes)
        if i_ep > 10000:
            agent.use_lr_decay = True
        
        ep_reward = 0
        env_transact = Env_transact()
        state_matrix_stack, states_stack, goal = env_transact.reset(env, window) 
        pre_state_matrix_stack, pre_states_stack = env_transact.predata_reset(env)
        pre_a, pre_r, pre_d, pre_a_logp = np.zeros((5,1)), np.zeros((5,1)), np.zeros((5,1)), np.zeros((5,1))

        buffer.epo_counter_start = buffer.pre_counter
        for t in range(10000):  #range(10000)
            # select action
            action_ind, a_logp = agent.select_action(state_matrix_stack, states_stack, goal,device)
            action = env_transact.env_action(env,action_ind)
            _, reward, terminated, truncated,done = env_transact.step(env, window, action,args)
            
            steps += 1

            state, goal = env_transact.data_env(env)
            states_stack = np.delete(states_stack, 0, 0)
            states_stack = np.append(states_stack,[state],0)
            state_matrix = env_transact.matrix_env(env)
            state_matrix_stack = np.delete(state_matrix_stack, 0, 0)
            state_matrix_stack = np.append(state_matrix_stack,[state_matrix],0)
    
            ep_reward += reward
            #专门生成每个样本9帧的样本数据
            pre_states_stack = np.delete(pre_states_stack, 0, 0)
            pre_states_stack = np.append(pre_states_stack,[state],0)
            pre_state_matrix_stack = np.delete(pre_state_matrix_stack, 0, 0)
            pre_state_matrix_stack = np.append(pre_state_matrix_stack,[state_matrix],0)
            pre_a = np.delete(pre_a, 0, 0)
            pre_a = np.append(pre_a,[[action_ind]],0)
            pre_r = np.delete(pre_r, 0, 0)
            pre_r = np.append(pre_r,[[reward]],0)
            pre_d = np.delete(pre_d, 0, 0)
            pre_d = np.append(pre_d,[[done]],0)
            pre_a_logp = np.delete(pre_a_logp, 0, 0)
            pre_a_logp = np.append(pre_a_logp,[[a_logp]],0)
            if t > 3: #从第5步开始存样本。
                buffer.pre_store((np.array(pre_state_matrix_stack, dtype = 'float32' ), np.array(pre_a, dtype = 'int64' ), np.array(pre_states_stack, dtype = 'float32' ), \
                          np.array(goal, dtype = 'float32' ),  np.array(pre_r, dtype = 'float32' ),  np.array(pre_d, dtype = 'int64' ), np.array(pre_a_logp, dtype = 'float32' )))
           
            if terminated or truncated:  #terminated为抵达目标，truncated为50步后未抵达目标
                for _ in range(4): #需要在最后结束时往前面移4次，保证每回合最后一个样本的第5个是终止状态，且之后4个都和终止状态一样。因为在做网络更新时当前状态就是第四个。
                    pre_states_stack = np.delete(pre_states_stack, 0, 0)
                    pre_states_stack = np.append(pre_states_stack,[state],0)
                    pre_state_matrix_stack = np.delete(pre_state_matrix_stack, 0, 0)
                    pre_state_matrix_stack = np.append(pre_state_matrix_stack,[state_matrix],0)
                    pre_a = np.delete(pre_a, 0, 0)
                    pre_a = np.append(pre_a,[[action_ind]],0)
                    pre_r = np.delete(pre_r, 0, 0)
                    pre_r = np.append(pre_r,[[reward]],0)
                    pre_d = np.delete(pre_d, 0, 0)
                    pre_d = np.append(pre_d,[[done]],0)
                    pre_a_logp = np.delete(pre_a_logp, 0, 0)
                    pre_a_logp = np.append(pre_a_logp,[[a_logp]],0)
                    buffer.pre_store((np.array(pre_state_matrix_stack, dtype = 'float32' ), np.array(pre_a, dtype = 'int64' ), np.array(pre_states_stack, dtype = 'float32' ), \
                          np.array(goal, dtype = 'float32' ),  np.array(pre_r, dtype = 'float32' ),  np.array(pre_d, dtype = 'int64' ), np.array(pre_a_logp, dtype = 'float32' )))

                print(env.agent_pos)
                if running_score > 0.1:
                    args.her = False
                elif running_score < 0.:
                    args.her = True
                if args.her: #是否调用her 函数
                    buffer.pre_her_func(max_steps = env.max_steps, newgoal_size_in = 4)  
                    print('her')
                       
                if terminated:
                    num_terminate += 1
                agent.writer.add_scalar('reward/ep_reward', ep_reward, i_ep)
                agent.writer.add_scalar('step/steps_epo', t, i_ep) 
                running_score = running_score * 0.99 + ep_reward * 0.01 
                agent.writer.add_scalar('score/score_epoch', running_score, epochs) 
                epochs += 1 
                print('running_score is',running_score)
                print("episodes {}, step is {},ep_reward is {},counter is {}".format(i_ep, t, ep_reward, buffer.pre_counter))
                if i_ep % 10 == 0:
                    
                    agent.writer.add_scalar('reward/num_terminate', num_terminate, num_10epo)
                    print("num_terminate=",num_terminate,"num_10epo=", num_10epo)
                    num_10epo += 1
                    num_terminate = 0 
                if (i_ep +1) % 50000 == 0:
                    print('net save')
                    agent.save_param(i_ep,running_score)
                break

            if buffer.pre_full:
                print('update')
                agent.update(buffer.pre_buffer,device, i_ep)
                buffer.pre_counter = 0
                buffer.pre_full = False
            

    
        
