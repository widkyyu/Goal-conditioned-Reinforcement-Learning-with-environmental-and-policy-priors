
import gym
import sys, os
from pathlib import Path
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

from agent.PPO import PPO
from env_buffer import Buffer_gridworld, Env_transact
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="gym environment to load", default="MiniGrid-twoarmy-17x17-v4")
    parser.add_argument("--seed",type=int,help="random seed to generate the environment with",default=9981)#6667,3344,
    parser.add_argument("--tile_size", type=int, help="size at which to render tiles", default=17)
    parser.add_argument("--batch_size", type=int, help="size at which to sample", default=128)
    parser.add_argument("--her",default=True,help="hindlight experience replay")
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay (default: 0.0001)')
    parser.add_argument('--lr_gamma', type=float, default=0.8, metavar='LG', help='learning rate discount factor (default: 0.8)')
    parser.add_argument('--lr_step_size', type=int, default=200, help='each step_size perform learning rate discount factor (default: 200)')
    parser.add_argument("--track_buffer_file", help="net to load", default='/datadisk/yuzhe/predictor/track_buffer/ppo_pure/')
    parser.add_argument('--num_episodes', type=int, default=1000000, help='number all episodes (default: 100000)')
    parser.add_argument('--max_steps', type=int, default=50, help='number of max steps in environment (default: 100)')
    parser.add_argument("--log_dir", help="model to load", default='/media/yuzhe/yuzhe/predictor/param/')
    parser.add_argument("--cuda", help="buffer to load", default="cuda:6")
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
    epochs = 0
    action = None
    num_terminate = 0
    num_10epo = 0

    agent = PPO()
    
    agent.name = 'ppo'+'_'+args.env+'_'+str(seed) +'seed_' 
    agent.heatmapfilename = 'ppo'+'_'+ args.env +'_'+ str(seed) + 'seed_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    agent.gamma = args.gamma
    agent.lr = args.lr
    agent.weight_decay = args.weight_decay
    agent.lr_step_size = args.lr_step_size
    agent.lr_gamma = args.lr_gamma
    agent.actor.to(device)
    agent.critic.to(device)
    agent.batch_size = args.batch_size
    agent.max_steps  = args.max_steps
    #储存agent点轨迹的路径
    track_filepath = Path(args.track_buffer_file + agent.name +datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'/')

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
    
    buffer.transition = np.dtype([ ('s', np.float32, (5,buffer.grid_size**2)),('a', np.int64, (1,)), 
        ('p', np.float32, (5,2)),  ('g', np.float32, (2,)),  ('r', np.float32, (1,)),  
        ('d', np.float32, (1,)), ('a_logp', np.float32, (1,))]) 
    buffer.buffer_capacity = 2048
    buffer.buffer = np.empty(buffer.buffer_capacity, dtype=buffer.transition)
    running_score = 0
    for i_ep in range(args.num_episodes):
        if i_ep > 10000:
            agent.use_lr_decay = True
        
        ep_reward = 0
        env_transact = Env_transact()
        state_matrix_stack, states_stack, goal = env_transact.reset(env, window) 
        buffer.epo_counter_start = buffer.counter

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
            buffer.store((np.array(state_matrix_stack, dtype = 'float32' ), np.array([action_ind], dtype = 'int64' ), np.array(states_stack, dtype = 'float32' ), \
                          np.array(goal, dtype = 'float32' ),  np.array([reward], dtype = 'float32' ),  np.array([done], dtype = 'int64' ), np.array([a_logp], dtype = 'float32' )))
            ep_reward += reward
   
            if terminated or truncated:  #terminated为抵达目标，truncated为50步后未抵达目标
                print(env.agent_pos)
                if running_score > 0.1:
                    args.her = False
                elif running_score < 0.:
                    args.her = True
                if args.her: #是否调用her 函数
                    buffer.her_func(max_steps = env.max_steps, newgoal_size_in = 4)  
                    print('her')
                       
                if terminated:
                    num_terminate += 1
                agent.writer.add_scalar('reward/ep_reward', ep_reward, i_ep)
                agent.writer.add_scalar('step/steps_epo', t, i_ep) 
                running_score = running_score * 0.99 + ep_reward * 0.01 
                agent.writer.add_scalar('score/score_epoch', running_score, epochs) 
                epochs += 1 
                print('running_score is',running_score)
                print("episodes {}, step is {},ep_reward is {},counter is {}".format(i_ep, t, ep_reward, buffer.counter))
                if i_ep % 10 == 0:
                    
                    agent.writer.add_scalar('reward/num_terminate', num_terminate, num_10epo)
                    print("num_terminate=",num_terminate,"num_10epo=", num_10epo)
                    num_10epo += 1
                    num_terminate = 0 
                if i_ep % 50000 == 0:
                    print('net save')
                    agent.save_param(i_ep,running_score)
                break

            if buffer.full:
                print('update')
                agent.update(buffer.buffer,device,i_ep)
                buffer.counter = 0
                buffer.full = False
            

    
        
