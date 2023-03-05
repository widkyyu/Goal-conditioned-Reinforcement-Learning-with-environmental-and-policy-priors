#只有在room1有两个固定位置的障碍物，room2是空的
from gym_minigrid.minigrid import Goal, Ball,Grid, Wall,MiniGridEnv, MissionSpace
import numpy as np

class Twoarmy_v6(MiniGridEnv):
    """
    Twoarmy,  sparse reward
    """

    def __init__(self, size=8, agent_pos=(3,15), goal_pos=(14,2), **kwargs): #goal_pos=(2,3)
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self.obstacle_type = Wall()
        self.n_obstacles = 3
        self.step_move = 0
        self.pone = False
        self.pone1 = False
        self.pone2 = False
        self.Update_horizontal = False
        self.Update_longitudinal = True
        self.patrol = False
        self.up1 = False  #巡逻队1上移
        self.right2 = True #巡逻队1右移
        self.risk_count = 0
        self.first_to_room2 = True
        mission_space = MissionSpace(
            mission_func=lambda: "get to the green goal square"
        )

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=50,  #4 * size * size
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        for i in range(1,6):
            self.put_obj(self.obstacle_type, i, 8)
        for i in range(11,16):
            self.put_obj(self.obstacle_type, i, 8)
        
        # Place obstacles
        self.obstacles = []  #中间横排的三个黄球
        self.obstacles1 = [] #右上竖排的三个黄球
        self.obstacles2 = [] #左上四个方正排列的黄球
        
        for i_obst in range(self.n_obstacles):
            self.obstacles.append(Ball("yellow"))
            self.obstacles1.append(Ball("yellow"))
            self.put_obj(self.obstacles[i_obst], i_obst+7, 8)

        for i_obst in range(4):
            self.obstacles2.append(Ball("yellow"))

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            # assuming random start direction
            self.agent_dir = 3
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            self.goal_pos = self._goal_default_pos
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.goal_pos = self.place_obj(Goal())  #随机位置

        self.mission = "get to the green goal square"
    
    def step(self, action):
        # Invalid action
        if action >= self.action_space.n:
            action = 0
        
        self.step_move += 1

        # # Check if there is an obstacle in front of the agent
        # front_cell = self.grid.get(*self.front_pos)
        # not_clear = front_cell and front_cell.type != "goal"
        

        # Update obstacle positions
        old_pos_ = []
        for i_obst in range(len(self.obstacles)):
            old_pos = self.obstacles[i_obst].cur_pos
            self.grid.set(old_pos[0], old_pos[1], None)
            old_pos_.append((old_pos[0], old_pos[1]))
        for i_obst in range(len(self.obstacles)):
            old_pos = self.obstacles[i_obst].cur_pos
            try:
                if self.step_move % 6 ==1 or self.step_move % 6 ==0:
                    self.put_obj(self.obstacles[i_obst], old_pos_[i_obst][0]+1, 8)
                elif self.step_move % 6 ==2 or self.step_move % 6 ==3:
                    self.put_obj(self.obstacles[i_obst], old_pos_[i_obst][0]-1, 8)
                elif self.step_move % 6 ==4 or self.step_move % 6 ==5:
                    self.put_obj(self.obstacles[i_obst], old_pos_[i_obst][0], 8)

            except Exception:
                pass
        
        
        # #Update longitudinal obstacle positions
        # if self.Update_longitudinal:
        #     self.Update_horizontal = False
        #     if self.step_move % 4 ==2 or self.step_move % 6 ==3 or self.step_move % 6 ==0 or np.random.choice(range(10), 1).item()==6:
        #         if self.patrol:
        #             if self.up1:
        #                 old_pos_ = []
        #                 for i_obst in range(len(self.obstacles1)):
        #                     old_pos = self.obstacles1[i_obst].cur_pos
        #                     self.grid.set(old_pos[0], old_pos[1], None)
        #                     old_pos_.append((old_pos[0], old_pos[1]))
        #                 for i_obst in range(len(self.obstacles1)):
        #                     try:
        #                         self.put_obj(self.obstacles1[i_obst], old_pos_[i_obst][0], old_pos_[i_obst][1]-1)
        #                     except Exception:
        #                         pass
        #                 if self.obstacles1[0].cur_pos[1] == 3:
        #                     self.up1 = False
        #             else:
        #                 old_pos_ = []
        #                 for i_obst in range(len(self.obstacles1)):
        #                     old_pos = self.obstacles1[i_obst].cur_pos
        #                     self.grid.set(old_pos[0], old_pos[1], None)
        #                     old_pos_.append((old_pos[0], old_pos[1]))
        #                 for i_obst in range(len(self.obstacles1)):
        #                     try:
        #                         self.put_obj(self.obstacles1[i_obst], old_pos_[i_obst][0], old_pos_[i_obst][1]+1)
        #                     except Exception:
        #                         pass
        #                 if self.obstacles1[2].cur_pos[1] == 7:
        #                     self.up1 = True
        
        # #Update horizontal obstacle positions
        # if self.Update_horizontal:
        #     self.Update_longitudinal=False
        #     if self.step_move % 6 ==0 or self.step_move % 6 ==2 or self.step_move % 6 ==3 or self.step_move % 6 ==5 or self.step_move % 6 ==4 or np.random.choice(range(10), 1).item()==6:
        #         if self.patrol:
        #             if self.right2:
        #                 old_pos_ = []
        #                 for i_obst in range(len(self.obstacles2)):
        #                     old_pos = self.obstacles2[i_obst].cur_pos
        #                     self.grid.set(old_pos[0], old_pos[1], None)
        #                     old_pos_.append((old_pos[0], old_pos[1]))
        #                 for i_obst in range(len(self.obstacles2)):
        #                     try:
        #                         self.put_obj(self.obstacles2[i_obst], old_pos_[i_obst][0]+1, old_pos_[i_obst][1])
        #                     except Exception:
        #                         pass
        #                 if self.obstacles2[3].cur_pos[0] == 11:
        #                     self.right2 = False
        #             else:
        #                 old_pos_ = []
        #                 for i_obst in range(len(self.obstacles2)):
        #                     old_pos = self.obstacles2[i_obst].cur_pos
        #                     self.grid.set(old_pos[0], old_pos[1], None)
        #                     old_pos_.append((old_pos[0], old_pos[1]))
        #                 for i_obst in range(len(self.obstacles2)):
        #                     try:
        #                         self.put_obj(self.obstacles2[i_obst], old_pos_[i_obst][0]-1, old_pos_[i_obst][1])
        #                     except Exception:
        #                         pass
        #                 if self.obstacles2[0].cur_pos[0] == 5:
        #                     self.right2 = True
            
        # Update the agent's position/direction
        obs, reward, terminated, truncated, info = super().step(action)
        reward = -0.01
        if not self.pone: #room1中随机出现的方墙
            if self.agent_pos[0] > 3 or self.agent_pos[1] < 14:

                # i = np.random.choice(range(9,13), 1).item()
                i = 11
                self.put_obj(self.obstacle_type, 4, i)
                self.put_obj(self.obstacle_type, 5, i)
                self.put_obj(self.obstacle_type, 4, i+1)
                self.put_obj(self.obstacle_type, 5, i+1)

                # i = np.random.choice(range(6,10), 1).item()
                i = 8
                self.put_obj(self.obstacle_type, i, 11)
                self.put_obj(self.obstacle_type, i, 12)
                self.put_obj(self.obstacle_type, i+1, 11)
                self.put_obj(self.obstacle_type, i+1, 12)
                self.pone = True
        
        # if not self.pone1 and self.Update_horizontal: #可能出现的横墙
        #     if self.agent_pos[0] > 11 or self.agent_pos[1] < 7:

        #         for i in range(13,16):
        #             self.put_obj(self.obstacle_type, i, 3)
        #         self.pone1 = True
        
        # if not self.pone2 and self.Update_longitudinal: #可能出现的竖墙
        #     if self.agent_pos[0] > 11 or self.agent_pos[1] < 7:

        #         for i in range(1,5):
        #             self.put_obj(self.obstacle_type, 13, i)
        #         self.pone2 = True
    

        # if not self.patrol: #room2中黄球的起始位置
        #     if  self.agent_pos[1] == 8:
        #         i = np.random.choice(range(6,10), 1).item()
        #         self.put_obj(self.obstacles2[0], i, 4)
        #         self.put_obj(self.obstacles2[1], i+1, 4)
        #         self.put_obj(self.obstacles2[2], i, 5)
        #         self.put_obj(self.obstacles2[3], i+1, 5)
                

        #         i = np.random.choice(range(4,5), 1).item()
        #         for i_obst in range(self.n_obstacles):
        #             self.put_obj(self.obstacles1[i_obst], 12, i_obst+i)
                    
        #         self.patrol = True
        
        
        if self.agent_pos[0] == self.obstacles[1].cur_pos[0] and self.agent_pos[1] == self.obstacles[1].cur_pos[1]:
                reward = -0.9
                truncated = True
        if self.agent_pos[0] == self.obstacles[0].cur_pos[0] and self.agent_pos[1] == self.obstacles[0].cur_pos[1]:
            reward = -0.9
            truncated = True
        if self.agent_pos[0] == self.obstacles[2].cur_pos[0] and self.agent_pos[1] == self.obstacles[2].cur_pos[1]:
            reward = -0.9
            truncated = True
        
        if self.agent_pos[1] == self.obstacles[0].cur_pos[1]+1:
                if self.agent_pos[0]==self.obstacles[0].cur_pos[0]  or self.agent_pos[0]==self.obstacles[1].cur_pos[0]  or self.agent_pos[0]==self.obstacles[2].cur_pos[0]:
                    reward = -0.1

        if self.patrol:
            if self.agent_pos[1] == self.obstacles2[2].cur_pos[1]+1:
                if self.agent_pos[0]==self.obstacles2[2].cur_pos[0] or self.agent_pos[0]==self.obstacles2[3].cur_pos[0]:
                    reward = -0.1
            if self.agent_pos[0] == self.obstacles2[0].cur_pos[0]-1:
                if self.agent_pos[1]==self.obstacles2[0].cur_pos[1]  or self.agent_pos[1]==self.obstacles2[2].cur_pos[1]:
                    reward = -0.1
            if self.agent_pos[0] == self.obstacles2[1].cur_pos[0]+1:
                if self.agent_pos[1]==self.obstacles2[1].cur_pos[1]  or self.agent_pos[1]==self.obstacles2[3].cur_pos[1]:
                    reward = -0.1
            
            if self.agent_pos[0] == self.obstacles1[0].cur_pos[0]-1:
                if self.agent_pos[1]==self.obstacles1[0].cur_pos[1]  or self.agent_pos[1]==self.obstacles1[1].cur_pos[1]  or self.agent_pos[1]==self.obstacles1[2].cur_pos[1]:
                    reward = -0.1
        
            
            if self.agent_pos[0] == self.obstacles1[1].cur_pos[0] and self.agent_pos[1] == self.obstacles1[1].cur_pos[1]:
                reward = -0.9
                truncated = True
            if self.agent_pos[0] == self.obstacles1[0].cur_pos[0] and self.agent_pos[1] == self.obstacles1[0].cur_pos[1]:
                reward = -0.9
                truncated = True
            if self.agent_pos[0] == self.obstacles1[2].cur_pos[0] and self.agent_pos[1] == self.obstacles1[2].cur_pos[1]:
                reward = -0.9
                truncated = True


            if self.agent_pos[0] == self.obstacles2[1].cur_pos[0] and self.agent_pos[1] == self.obstacles2[1].cur_pos[1]:
                reward = -0.9
                truncated = True
            if self.agent_pos[0] == self.obstacles2[0].cur_pos[0] and self.agent_pos[1] == self.obstacles2[0].cur_pos[1]:
                reward = -0.9
                truncated = True
            if self.agent_pos[0] == self.obstacles2[2].cur_pos[0] and self.agent_pos[1] == self.obstacles2[2].cur_pos[1]:
                reward = -0.9
                truncated = True
            if self.agent_pos[0] == self.obstacles2[3].cur_pos[0] and self.agent_pos[1] == self.obstacles2[3].cur_pos[1]:
                reward = -0.9
                truncated = True
        
        if self.first_to_room2:
            if self.agent_pos[1] == 7:
                reward = 0.2
                self.first_to_room2 = False
        
        if reward == -0.1:
            self.risk_count += 1
            if self.risk_count > 5:
                truncated = True
                self.riskcount = 0

        if terminated or truncated:
            if terminated:
                reward = 0.9
            self.step_move = 0
            self.pone = False #是否有炸弹炸过道路
            self.patrol = False #是否有巡逻队
            self.first_to_room2 = True
            self.risk_count = 0
            self.pone1 = False
            self.pone2 = False
            if np.random.choice(range(2), 1).item()==1:
                self.up1 = False  #巡逻队1上移
                self.right2 = True #巡逻队2右移
            else:
                self.up1 = True  #巡逻队1上移
                self.right2 = False #巡逻队2右移

            if np.random.choice(range(2), 1).item()==1:
                self.Update_horizontal = False
                self.Update_longitudinal = True
            else:
                self.Update_horizontal = True
                self.Update_longitudinal = False

        # # If the agent tried to walk over an obstacle or wall
        # if action == self.actions.forward and not_clear:
        #     reward = -1
        #     terminated = True
        #     return obs, reward, terminated, truncated, info
        return obs, reward, terminated, truncated, info
