import time
import gymnasium as gym
import torch
from PPO import PPO

def train(env_id,total_timesteps):
    env = gym.make(env_id,render_mode = "rgb_array")
    model = PPO(env)
    model.env_id = env_id
    model.learn(total_timesteps)
    path = 'PPO_' + env_id + "_model_"
    model.save(path)

if __name__ == '__main__':
    train("BipedalWalker-v3",1405000)