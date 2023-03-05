
Requirements:
- Python 3.7 to 3.10
- OpenAI Gym v0.22 to v0.25
- NumPy 1.18+
- Matplotlib (optional, only needed for display) - 3.0+

## Environment Reference

```
Chevalier-Boisvert, Maxime, Lucas Willems, and Suman Pal. "Minimalistic gridworld environment for openai gym." (2018): 13-20.
```


## Installation
After installing requirement:

```
cd gym-minigrid_1216
pip3 install -e .
```

then, you can control the agent by keyboard running "manual_control.py"

## Environment Version

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


