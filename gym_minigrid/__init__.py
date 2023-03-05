from gym.envs.registration import register

from gym_minigrid.minigrid import Wall


def register_minigrid_envs():
    
    
    #hard environment
    register(
        id="MiniGrid-twoarmy-17x17-v4",
        entry_point="gym_minigrid.envs:Twoarmy_v4",
        kwargs={"size": 17},
    )
    #easy environment
    register(
        id="MiniGrid-twoarmy-17x17-v6",
        entry_point="gym_minigrid.envs:Twoarmy_v6",
        kwargs={"size": 17},
    )

