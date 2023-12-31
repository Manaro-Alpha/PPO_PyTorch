# PPO

## About

This repo contains an optimised version of PPO using tricks like Generalised Advantage Estimates, Entropy Regularisation etc. in an attempt to match the performance offered by StableBaselines3's PPO.

## Usage

- To train the agent, run `train.py`
- Run `tensorboard --logdir runs` to visualise the data in your browser
- To test the trained policy, run `test.py`

## Results

| PPO Continuous LunarLander-v2  | PPO Continuous LunarLander-v2 |
| :-------------------------:|:-------------------------: |
| ![](https://github.com/Manaro-Alpha/PPO_PyTorch/blob/main/GIFs/rl-video-LunarLanderContinuous-v2-episode-1000%20.gif) | ![](https://github.com/Manaro-Alpha/PPO_PyTorch/blob/main/Plot_Graphs/output.png) |

| PPO Continuous BipedalWalker-v3  | PPO Continuous BipedalWalker-v3 |
| :-------------------------:|:-------------------------: |
 ![](https://github.com/Manaro-Alpha/PPO_PyTorch/blob/main/GIFs/rl-video-episode-1000.gif) | ![](https://github.com/Manaro-Alpha/PPO_PyTorch/blob/main/Plot_Graphs/output_BipedalWalker-v3.png) |

