
from PIL import Image
from turtle import right
import gym
import sys, os
print(os.getcwd()) #获取当前路径
sys.path.append("../..")
sys.path.append("..")
from gym_minigrid.window import Window
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
import numpy as np
import random
import time
import datetime
from datetime import datetime
import matplotlib.pyplot as plt
from numpy import float64
from pathlib import Path
from itertools import chain
import torch
from env_buffer import Buffer_gridworld, Env_transact
import argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="gym environment to load", default="MiniGrid-twoarmy-17x17-v6")
    parser.add_argument("--act_net_file", help="act_net to load", default='/media/yuzhe/yuzhe/predictor/param/ppo/gridim2023_01_17_15_28_11/ppo_score0.5722677894104027_MiniGrid-twoarmy-17x17-v4__net2000epoch2023_01_17_16_15_39.pkl')
    parser.add_argument("--agent", help="agent to chose", default="random_")
    parser.add_argument("--seed",type=int,help="random seed to generate the environment with",default=2345)
    parser.add_argument("--tile_size", type=int, help="size at which to render tiles", default=32)
    parser.add_argument("--her",default=True,help="hindlight experience replay")
    parser.add_argument('--num_episodes', type=int, default=100000, help='number all episodes (default: 100000)')
    parser.add_argument('--max_steps', type=int, default=90, help='number all steps (default: 50)')
    parser.add_argument('--buffer_capacity', type=int, default=10000, help='number all buffer (default: 100000)')
    parser.add_argument('--buffer_pre_capacity', type=int, default=10000, help='number all buffer (default: 100000)')
    parser.add_argument("--log_dir", help="model to load", default='/media/yuzhe/yuzhe/predictor/data/')
    parser.add_argument("--cuda", help="buffer to load", default="cuda:1")
    parser.add_argument("--server", help="if running in the server", default=True)
    parser.add_argument("--random_action", help="if choose action randomly", default=False)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    if args.server:
        device = torch.device(args.cuda if use_cuda else "cpu")
    else:
        device = torch.device("cuda:0" if use_cuda else "cpu")
    device = torch.device(args.cuda if use_cuda else "cpu")
    
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
    epo_counter_start = 0

    env = gym.make(
        args.env,
        seed=seed,
        new_step_api=True,
        tile_size=args.tile_size,
    )
    window = Window("gym_minigrid - " + args.env)

    # agent.net.load_state_dict(torch.load(args.act_net_file,map_location=args.cuda))
    # agent.net.to(device) #double()是双精度浮点数

    buffer = Buffer_gridworld()
    envgrid = env.grid
    buffer.grid_size = envgrid.height
    buffer.buffer_pre_capacity = args.buffer_pre_capacity

    buffer.pre_transition = np.dtype([ ('s', np.float64, (9,buffer.grid_size**2)),('a', np.int64, (5,1)), 
        ('p', np.float64, (9,2)),  ('g', np.float64, (2,)),  ('r', np.float64, (5,1)),  
        ('d', np.int64, (5,1)), ('a_logp', np.float64, (5,1))])
    buffer.pre_buffer = np.empty(buffer.buffer_pre_capacity, dtype=buffer.pre_transition)
    buffer.name = args.env + '_'+ args.agent + str(buffer.buffer_pre_capacity) + '_prebuffer_'

    for i_ep in range(args.num_episodes):
        ep_reward = 0
        action_ind = 0
        random_1policy = False #在room1时是否采取随机策略
        random_2policy = False #在room2时是否采取随机策略
        env_transact = Env_transact()
        state_matrix_stack, states_stack, goal = env_transact.reset(env, window) 
        pre_state_matrix_stack, pre_states_stack = env_transact.predata_reset(env)
        pre_a, pre_r, pre_d, pre_a_logp = np.zeros((5,1)), np.zeros((5,1)), np.zeros((5,1)), np.zeros((5,1))
        buffer.epo_counter_start = buffer.counter
        # if np.random.choice(range(10), 1).item() == 1:
        #     random_1policy = True
        # if np.random.choice(range(2), 1).item() == 0:
        #     random_2policy = True
        for t in range(10000):
            # select action
            # if random_1policy:#args.random_action:#np.random.choice(range(10), 1).item() == 1:
            #采集样本时全部用随机动作
            action_ind = np.random.choice(range(5), 1).item()
            # else:
            #     action_ind, _ = agent.select_action(state_matrix_stack, states_stack, goal,device)
            # if env.agent_pos[1] < 8:
            #     if random_2policy:
            #         action_ind = np.random.choice(range(5), 1).item()
            action = env_transact.env_action(env,action_ind)
            _, reward, terminated, truncated, done = env_transact.step(env, window, action,args)

            state, goal = env_transact.data_env(env)
            states_stack = np.delete(states_stack, 0, 0)
            states_stack = np.append(states_stack,[state],0)
            state_matrix = env_transact.matrix_env(env)
            state_matrix_stack = np.delete(state_matrix_stack, 0, 0)
            state_matrix_stack = np.append(state_matrix_stack,[state_matrix],0)
            ep_reward += reward
            
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
            
            if t > 3:
                buffer.pre_store((np.array(pre_state_matrix_stack, dtype = 'float64' ), np.array(pre_a, dtype = 'int64' ), np.array(pre_states_stack, dtype = 'float64' ), \
                          np.array(goal, dtype = 'float64' ),  np.array(pre_r, dtype = 'float64' ),  np.array(pre_d, dtype = 'int64' ), np.array(pre_a_logp, dtype = 'float64' )))

            ep_reward += reward
            if buffer.pre_counter % 100 ==1:
                print(buffer.pre_counter)
                # print(state_laten)
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
                    
                    buffer.pre_store((np.array(pre_state_matrix_stack, dtype = 'float64' ), np.array(pre_a, dtype = 'int64' ), np.array(pre_states_stack, dtype = 'float64' ), \
                          np.array(goal, dtype = 'float64' ),  np.array(pre_r, dtype = 'float64' ),  np.array(pre_d, dtype = 'int64' ), np.array(pre_a_logp, dtype = 'float64' )))  

                print("episodes {}, step is {},ep_reward is {},counter is {}".format(i_ep, t, ep_reward,buffer.pre_counter))
                break
            if buffer.pre_full:
                np.save(args.log_dir + 'predictor_'+ buffer.name + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+".npy", buffer.pre_buffer)
                print("store successly")
                break
        if buffer.pre_full:
            break
                
            
            

    
        
