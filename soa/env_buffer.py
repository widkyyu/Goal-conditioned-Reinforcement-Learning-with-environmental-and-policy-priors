
from PIL import Image
from turtle import right
from pathlib import Path
import operator
import gym
import sys, os
from gym_minigrid.window import Window
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
import numpy as np
import time
import datetime
from datetime import datetime
import matplotlib.pyplot as plt
from gym_minigrid.minigrid import Goal
from numpy import float64

#'s'为17*17grid的环境状态向量
#'a'1维动作信息，0左转、1右转和2前进；
#'p'为3维状态向量，前2维为agent二维坐标，第3维为agent的方向；
#'g'为2维坐标信息
#'r'是1维奖励值信息
#'d'是1维达到终点信息

class Buffer_gridworld():
    buffer_capacity = 10000 
    def __init__(self):
        self.name = None
        self.grid_size = None
        self.transition = None
        self.pre_transition = None
        self.buffer_track_capacity = 1000
        self.buffer_capacity = 100000
        self.buffer_pre_capacity = 100000
        self.buffer = []
        self.pre_buffer = []
        self.fp_buffer = []
        self.track_buffer = []
        self.counter = 0
        self.pre_counter = 0
        self.track_counter = 0
        self.fp_counter = 0
        self.full = False #True代表用的是已生成好的Buffer,所以开始即更新
        self.pre_full = False
        self.fp_full = False
        self.epo_counter_start = 0
        self.epo_counter_end = 0

        self.track_transition = np.dtype([('p', np.float64, (9,2)), ('d', np.int64, (5,1))])
        
    def track_store(self, env, path, i_ep, track_store):
        
        (i, j) = env.agent_pos
        agent_place = np.array((j, i), dtype = float) #np.array(2,)
        self.track_buffer.append(agent_place)
        self.track_counter += 1
        if track_store:
            if path.exists():
                np.save(str(path)+'/' + str(i_ep)+'i_ep_' + 'track_buffer_'+str(self.track_counter+1) +'counter_' +datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+".npy", self.track_buffer)
            else:
                os.makedirs(path)
                np.save(str(path)+'/' + str(i_ep)+'i_ep_' + 'track_buffer_'+str(self.track_counter+1) +'counter_' +datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+".npy", self.track_buffer)
            print("store track_buffer")
            self.track_buffer = []
        
            

    def store(self, transition):
        if self.counter >= self.buffer_capacity:
            self.counter = 0
            self.full = True
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            self.full = True
        return self.full
    
    def future_3_position_store(self, transition):
        if self.fp_counter >= self.buffer_pre_capacity:
            self.fp_counter = 0
            self.fp_full = True
        self.fp_buffer[self.fp_counter] = transition
        self.fp_counter += 1
        if self.fp_counter == self.buffer_pre_capacity:
            self.fp_counter = 0
            self.fp_full = True
        return self.fp_full

    def pre_store(self, transition):
        if self.pre_counter >= self.buffer_pre_capacity:
            self.pre_counter = 0
            self.pre_full = True
        self.pre_buffer[self.pre_counter] = transition
        self.pre_counter += 1
        if self.pre_counter == self.buffer_pre_capacity:
            self.pre_counter = 0
            self.pre_full = True
        return self.pre_full
    
    def her_func(self, max_steps=50, newgoal_size_in = 4):
        self.epo_counter_end = self.counter - 1
        newgoal_size = newgoal_size_in

        #获取对应的rollout数据集
        buffer_her = self.buffer[self.epo_counter_start:self.epo_counter_end+1].copy()

        #返回每种元素的计数,indices对应states_mask中元素在旧列表中第一次出现的index
        states_mask, indices, counts = np.unique(buffer_her['p'][:,4,0:2],return_index=True,return_counts=True,axis=0)
        if indices.size < newgoal_size_in:
            newgoal_size = indices.size
            
        rollout_size = self.epo_counter_end - self.epo_counter_start + 1
        if rollout_size > 0:
            episode_idxs = np.random.choice(indices, size=newgoal_size, replace=False)
            
            # replace goal with achieved goal
            for index in episode_idxs:    #轨迹长度应该是index + 1
                
                buffer_new  = buffer_her[:index + 1].copy()
                future_goal = buffer_new['p'][index, 4, 0:2]

                if index > 0 and index < self.buffer_capacity:
                    #获取对应的rollout数据集
                    buffer_new['g'][0:index+1] = future_goal
                    (i, j) = list(map(int,future_goal.tolist()))
                    # old_goal = np.where(buffer_new['s'] == 0.)[2]
                    # buffer_new['s'][0:index+1,:,old_goal[0]] = 0.9 #0.9代表是可通行区域
                    # buffer_new['s'][0:index+1,:,i*self.grid_size + j] = 0. #0.代表是goal，goal在态势图里不显示
                    buffer_new['r'][index] = 0.9
                    buffer_new['d'][index] = 1
                    
                    if self.epo_counter_end+1+index+1 <= self.buffer_capacity:
                        self.buffer[self.epo_counter_end+1:self.epo_counter_end+1+index+1]=buffer_new[0:index+1]
                        self.epo_counter_end = self.epo_counter_end+1+index
                    else:
                        count_end = self.epo_counter_end+1+index+1 - self.buffer_capacity #count_end是计算超过buffer_capacity多少
                        self.buffer[self.epo_counter_end+1:self.buffer_capacity]=buffer_new[0:index+1-count_end]
                        self.buffer[:count_end]=buffer_new[index+1-count_end:index+1]
                        self.epo_counter_end = count_end - 1
                        self.full = True
                        
        self.counter = self.epo_counter_end + 1

    def pre_her_func(self, max_steps=50, newgoal_size_in = 4):
        self.epo_counter_end = self.pre_counter - 1
        newgoal_size = newgoal_size_in

        #获取对应的rollout数据集
        buffer_her = self.pre_buffer[self.epo_counter_start:self.epo_counter_end+1].copy()

        #返回每种元素的计数,indices对应states_mask中元素在旧列表中第一次出现的index
        states_mask, indices, counts = np.unique(buffer_her['p'][:,8,0:2],return_index=True,return_counts=True,axis=0)
        if indices.size < newgoal_size_in:
            newgoal_size = indices.size
            
        rollout_size = self.epo_counter_end - self.epo_counter_start + 1
        if rollout_size > 0:
            episode_idxs = np.random.choice(indices, size=newgoal_size, replace=False)
            
            # replace goal with achieved goal
            for index in episode_idxs:    #轨迹长度应该是index + 1
                
                buffer_new  = buffer_her[:index + 1].copy()
                future_goal = buffer_new['p'][index, 8, 0:2]

                if index > 0 and index < self.buffer_pre_capacity:
                    #获取对应的rollout数据集
                    buffer_new['g'][0:index+1] = future_goal
                    (i, j) = list(map(int,future_goal.tolist()))
                    # old_goal = np.where(buffer_new['s'] == 0.)[2]
                    # buffer_new['s'][0:index+1,:,old_goal[0]] = 0.9 #0.9代表是可通行区域
                    # buffer_new['s'][0:index+1,:,i*self.grid_size + j] = 0. #0.代表是goal，goal在态势图里不显示
                    buffer_new['r'][index,4] = 0.9
                    buffer_new['d'][index,4] = 1
                    for k in range(4): #需要在最后结束时往前面移4次，保证每回合最后一个样本的第5个是终止状态，且之后4个都和终止状态一样。因为在做网络更新时当前状态就是第四个。
                        buffer_new = np.append(buffer_new,[buffer_new[index]],0)
                        index += 1
                        
                        convey = np.delete(buffer_new['p'][index], 0, 0)
                        buffer_new['p'][index] = np.append(convey,[buffer_new['p'][index-1-k,8]],0)

                        convey = np.delete(buffer_new['s'][index], 0, 0)
                        buffer_new['s'][index] = np.append(convey,[buffer_new['s'][index-1-k,8]],0)

                        convey = np.delete(buffer_new['a'][index], 0, 0)
                        buffer_new['a'][index] = np.append(convey,[buffer_new['a'][index-1-k,4]],0)

                        convey = np.delete(buffer_new['r'][index], 0, 0)
                        buffer_new['r'][index] = np.append(convey,[buffer_new['r'][index-1-k,4]],0)

                        convey = np.delete(buffer_new['d'][index], 0, 0)
                        buffer_new['d'][index] = np.append(convey,[buffer_new['d'][index-1-k,4]],0)

                        convey = np.delete(buffer_new['a_logp'][index], 0, 0)
                        buffer_new['a_logp'][index] = np.append(convey,[buffer_new['a_logp'][index-1-k,4]],0)

                    
                    if self.epo_counter_end+1+index+1 <= self.buffer_pre_capacity:
                        self.pre_buffer[self.epo_counter_end+1:self.epo_counter_end+1+index+1]=buffer_new[0:index+1]
                        self.epo_counter_end = self.epo_counter_end+1+index
                    else:
                        count_end = self.epo_counter_end+1+index+1 - self.buffer_pre_capacity #count_end是计算超过buffer_pre_capacity多少
                        self.pre_buffer[self.epo_counter_end+1:self.buffer_pre_capacity]=buffer_new[0:index+1-count_end]
                        self.pre_buffer[:count_end]=buffer_new[index+1-count_end:index+1]
                        self.epo_counter_end = count_end - 1
                        self.pre_full = True
                        
        self.pre_counter = self.epo_counter_end + 1

    #用于策略网络输出未来第三步位置
    def pre_f_her_func(self, max_steps=50, newgoal_size_in = 4):
        self.epo_counter_end = self.pre_counter - 1
        newgoal_size = newgoal_size_in

        #获取对应的rollout数据集
        buffer_her = self.pre_buffer[self.epo_counter_start:self.epo_counter_end+1].copy()

        #返回每种元素的计数,indices对应states_mask中元素在旧列表中第一次出现的index
        states_mask, indices, counts = np.unique(buffer_her['p'][:,8,0:2],return_index=True,return_counts=True,axis=0)
        if indices.size < newgoal_size_in:
            newgoal_size = indices.size
            
        rollout_size = self.epo_counter_end - self.epo_counter_start + 1
        if rollout_size > 0:
            episode_idxs = np.random.choice(indices, size=newgoal_size, replace=False)
            
            # replace goal with achieved goal
            for index in episode_idxs:    #轨迹长度应该是index + 1
                
                buffer_new  = buffer_her[:index + 1].copy()
                future_goal = buffer_new['p'][index, 8, 0:2]

                if index > 0 and index < self.buffer_pre_capacity:
                    #获取对应的rollout数据集
                    buffer_new['g'][0:index+1] = future_goal
                    (i, j) = list(map(int,future_goal.tolist()))
                    # old_goal = np.where(buffer_new['s'] == 0.)[2]
                    # buffer_new['s'][0:index+1,:,old_goal[0]] = 0.9 #0.9代表是可通行区域
                    # buffer_new['s'][0:index+1,:,i*self.grid_size + j] = 0. #0.代表是goal，goal在态势图里不显示
                    buffer_new['r'][index,4] = 0.9
                    buffer_new['d'][index,4] = 1
                    for k in range(4): #需要在最后结束时往前面移4次，保证每回合最后一个样本的第5个是终止状态，且之后4个都和终止状态一样。因为在做网络更新时当前状态就是第四个。
                        buffer_new = np.append(buffer_new,[buffer_new[index]],0)
                        index += 1
                        
                        convey = np.delete(buffer_new['p'][index], 0, 0)
                        buffer_new['p'][index] = np.append(convey,[buffer_new['p'][index-1-k,8]],0)

                        convey = np.delete(buffer_new['s'][index], 0, 0)
                        buffer_new['s'][index] = np.append(convey,[buffer_new['s'][index-1-k,8]],0)

                        convey = np.delete(buffer_new['a'][index], 0, 0)
                        buffer_new['a'][index] = np.append(convey,[buffer_new['a'][index-1-k,4]],0)

                        convey = np.delete(buffer_new['r'][index], 0, 0)
                        buffer_new['r'][index] = np.append(convey,[buffer_new['r'][index-1-k,4]],0)

                        convey = np.delete(buffer_new['d'][index], 0, 0)
                        buffer_new['d'][index] = np.append(convey,[buffer_new['d'][index-1-k,4]],0)

                        convey = np.delete(buffer_new['a_logp'][index], 0, 0)
                        buffer_new['a_logp'][index] = np.append(convey,[buffer_new['a_logp'][index-1-k,4]],0)

                        convey = np.delete(buffer_new['f'][index], 0, 0)
                        buffer_new['f'][index] = np.append(convey,[buffer_new['f'][index-1-k,4]],0)

                    
                    if self.epo_counter_end+1+index+1 <= self.buffer_pre_capacity:
                        self.pre_buffer[self.epo_counter_end+1:self.epo_counter_end+1+index+1]=buffer_new[0:index+1]
                        self.epo_counter_end = self.epo_counter_end+1+index
                    else:
                        count_end = self.epo_counter_end+1+index+1 - self.buffer_pre_capacity #count_end是计算超过buffer_pre_capacity多少
                        self.pre_buffer[self.epo_counter_end+1:self.buffer_pre_capacity]=buffer_new[0:index+1-count_end]
                        self.pre_buffer[:count_end]=buffer_new[index+1-count_end:index+1]
                        self.epo_counter_end = count_end - 1
                        self.pre_full = True
                        
        self.pre_counter = self.epo_counter_end + 1
    

class Env_transact():
    def __init__(self):
        self.name = None
        self.grid = None
        self.size_agentob = 17**2
        self.state_matrix = np.zeros((self.size_agentob, )) + np.array([0.9])
        self.agent_placeini = (0,0)
        self.agent_place = (0,0)
        self.place_erro = (0,0)
        self.subgoal_place = (0,0)
        self.trans_ = False #是否重新计算agent的grid-base
        self.runstep = 0 
        self.riskcount = 0 #在巡逻区前徘徊步数

    def redraw(self,window, img):
        window.show_img(img)
        return 

    def matrix_env(self, env):

        self.grid = env.grid

        self.size_agentob = self.grid.height**2
        self.state_matrix = np.zeros((self.grid.height**2, )) + np.array([0.9])
        for i in range(self.size_agentob):
            if self.grid.grid[i] == None:
                self.state_matrix[i] = 0.9
            elif self.grid.grid[i].type == 'wall' :
                self.state_matrix[i] = -0.9
            elif self.grid.grid[i].type == 'ball' :
                self.state_matrix[i] = -0.5
            elif self.grid.grid[i].type == 'goal' :  #在态势图里不显示目标位置，目标位置直接通过坐标向量给予。原先是标为0.0
                self.state_matrix[i] = 0.9
        (i, j) = env.agent_pos
        self.state_matrix[self.grid.height*j + i] = 0.3
            
        return self.state_matrix

    def data_env(self,env): #加了方向项
        (i, j) = env.agent_pos
        agent_place = np.array((j, i), dtype = float) #np.array(2,)
        (i, j) = env.goal_pos
        goal = np.array((j, i), dtype = float)
        (i, j) = env.obstacles[0].cur_pos
        ball1 = np.array((j, i), dtype = float)
        (i, j) = env.obstacles[1].cur_pos
        ball2 = np.array((j, i), dtype = float)
        (i, j) = env.obstacles[2].cur_pos
        ball3 = np.array((j, i), dtype = float)
        
        # state = np.concatenate((agent_place, ball1, ball2, ball3),axis=0) #n
        
        return agent_place, goal

    def free_env(self,env): #根据第一个黄点的位置寻找可通行区域
        (i, j) = env.agent_pos
        agent_place = np.array((j, i), dtype = float) #np.array(2,)
        (i, j) = env.goal_pos
        goal = np.array((j, i), dtype = float)
        (i, j) = env.obstacles[0].cur_pos
        ball1 = np.array((j, i), dtype = float)
        (i, j) = env.obstacles[1].cur_pos
        ball2 = np.array((j, i), dtype = float)
        (i, j) = env.obstacles[2].cur_pos
        ball3 = np.array((j, i), dtype = float)
        free_region = np.array([8,6,8,7])
        if ball1[1] == 6:
            free_region = np.array([8,9,8,10])
        elif ball1[1] == 7:
            free_region = np.array([8,6,8,10])
        elif ball1[1] == 8:
            free_region = np.array([8,6,8,7])    
        state = np.concatenate((agent_place, free_region, goal),axis=0)
        states_stack = np.tile(state,(10,1))
        return state, states_stack
    
    def pre_col(self,env): #收集用于预测网络训练的空间数据
        state_matrix = self.matrix_env(env)
        state_matrix_stack = np.tile(state_matrix,(8,1))
        
        return state_matrix, state_matrix_stack

    def env_action(self,env,action_agent):
        action = None
        if action_agent == 0:
            action = env.actions.left
        elif action_agent == 1:
            action = env.actions.right
        elif action_agent == 2:
            action = env.actions.up
        elif action_agent == 3:
            action = env.actions.down
        elif action_agent == 4:
            action = env.actions.done
        return action
    
    # def env_trainsform_future_position(self,Px_ind,Py_ind):
    #     Px = 0
    #     if Px_ind == 0:
    #         Px = 0
    #     elif Px_ind == 1:
    #         Px = 1
    #     elif Px_ind == 2:
    #         Px = 2
    #     elif Px_ind == 3:
    #         Px = 3
    #     elif Px_ind == 4:
    #         Px = -1
    #     elif Px_ind == 5:
    #         Px = -2
    #     elif Px_ind == 6:
    #         Px = -3
        
    #     Py = 0
    #     if Py_ind == 0:
    #         Py = 0
    #     elif Py_ind == 1:
    #         Py = 1
    #     elif Py_ind == 2:
    #         Py = 2
    #     elif Py_ind == 3:
    #         Py = 3
    #     elif Py_ind == 4:
    #         Py = -1
    #     elif Py_ind == 5:
    #         Py = -2
    #     elif Py_ind == 6:
    #         Py = -3

    #     return Px, Py
    
    def reset(self,env, window):
        
        env.reset() #{'image','direction','mission'}
        if hasattr(env, "mission"):
            print("Mission: %s" % env.mission)
            window.set_caption(env.mission)

        state_matrix = self.matrix_env(env)
        state_matrix_stack = np.tile(state_matrix,(5,1))    
        state, goal = self.data_env(env)  #state and goal both are np type
        states_stack = np.tile(state,(5,1)) #将state复制5个

        img = env.get_full_render()
        self.redraw(window, img)

        return state_matrix_stack, states_stack, goal 

    def predata_reset(self, env) :

        state_matrix = self.matrix_env(env)
        state_matrix_stack = np.tile(state_matrix,(9,1))    
        state, goal = self.data_env(env)  #state and goal both are np type
        states_stack = np.tile(state,(9,1)) #将state复制9个

        return state_matrix_stack, states_stack
    
    def step(self,env, window,action,args):
        self.runstep += 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = 0
        
        if self.runstep > 49:
            truncated = True
        
        if terminated:
            done = 1
            reward = 0.9
            print("terminated!")
            print(f"step={env.step_count}, reward={reward:.2f}")  
        elif truncated:
            print("truncated!")
            print(f"step={env.step_count}, reward={reward:.2f}")
        img = env.get_full_render()

        if not args.server:
            self.redraw(window, img)
        
        return obs, reward, terminated, truncated, done

