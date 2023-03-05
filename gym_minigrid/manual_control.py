#!/usr/bin/env python3

import gym
import numpy as np
from gym_minigrid.window import Window
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper


def redraw(window, img):
    window.show_img(img)


def reset(env, window):
    _ = env.reset()

    if hasattr(env, "mission"):
        print("Mission: %s" % env.mission)
        window.set_caption(env.mission)
    env.agent_coordinate_display = env.agent_coordinate
    # env.put_obj(env.key,12, 5)
    img = env.get_render()

    redraw(window, img)


def step(env, window, action):
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"step={env.step_count}, reward={reward:.2f}")
    print(env.agent_pos) #0是向右，然后顺时针旋转
    
    if env.agent_pos[0] >10 and env.agent_pos[1] > 3:
        env.grid.set(12, 5, None)
        if env.agent_pos == (12,5):
            if env.agent_dir == 0:
                env.put_obj(env.key,15, 12)
                subgoal = np.array([15, 12])
            elif env.agent_dir == 1:
                env.put_obj(env.key,15, 12)
                subgoal = np.array([15, 12])
            elif env.agent_dir == 2:
                env.put_obj(env.key,15, 12)
                subgoal = np.array([15, 12])
            elif env.agent_dir == 3:
                env.put_obj(env.key,15, 12)
                subgoal = np.array([15, 12])
            img = env.get_full_render()     
    if env.agent_pos[0] >13 and env.agent_pos[1] > 10:
        env.grid.set(15, 12, None)
        if env.agent_pos == (15, 12):
            if env.agent_dir == 0:
                env.put_obj(env.key,20, 20)
                subgoal = np.array([20, 20])
            elif env.agent_dir == 1:
                env.put_obj(env.key,20, 20)
                subgoal = np.array([20, 20])
            elif env.agent_dir == 2:
                env.put_obj(env.key,20, 20)
                subgoal = np.array([20, 20])
            elif env.agent_dir == 3:
                env.put_obj(env.key,20, 20)
                subgoal = np.array([20, 20])
            img = env.get_full_render()

    if env.agent_pos[0] >18 and env.agent_pos[1] > 18:
        env.grid.set(20, 20, None)
        if env.agent_pos == (20, 20):
            subgoal = np.array(env.goal_pos)
            img = env.get_full_render()  

    if terminated:
        print("terminated!")
        reset(env, window)
    elif truncated:
        print("truncated!")
        reset(env, window)
    else:
        img = env.grid.render(env.tile_size,
            env.agent_pos,
            env.agent_dir,
            highlight_mask=env.agent_coordinate if True else None,)
        redraw(window, img)


def key_handler(env, window, event):
    print("pressed", event.key)
    
    if event.key == "escape":
        window.close()
        return

    if event.key == "backspace":
        reset(env, window)
        return

    if event.key == "left":
        step(env, window, env.actions.left)
        return
    if event.key == "right":
        step(env, window, env.actions.right)
        return
    if event.key == "up":
        step(env, window, env.actions.up)
        return
    if event.key == "down":
        step(env, window, env.actions.down)
        return

    # Spacebar
    if event.key == " ":
        step(env, window, env.actions.toggle)
        return
    if event.key == "pageup":
        step(env, window, env.actions.pickup)
        return
    if event.key == "pagedown":
        step(env, window, env.actions.drop)
        return

    if event.key == "enter":
        step(env, window, env.actions.done)
        return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", help="gym environment to load", default="MiniGrid-twoarmy-17x17-v0"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=-1,
    )
    parser.add_argument(
        "--tile_size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent_view",
        default=True,
        help="draw the agent sees (partially observable view)",
        action="store_true",
    )

    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed
    env = gym.make(
        args.env,
        seed=seed,
        new_step_api=True,
        render_mode="human",  # Note that we do not need to use "human", as Window handles human rendering.
        tile_size=args.tile_size,
    )

    if args.agent_view:
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)

    window = Window("gym_minigrid - " + args.env)
    window.reg_key_handler(lambda event: key_handler(env, window, event))

    reset(env, window)
    
    # Blocking event loop
    window.show(block=True)
