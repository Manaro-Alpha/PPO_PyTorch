import gymnasium as gym
import torch
import numpy as np
from PPO import PPO,ActorCritic

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_test_episodes = 10
    max_ep_length = 500
    env = gym.make("BipedalWalker-v3",render_mode= 'human')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    test_agent = PPO(env)
    test_agent.load("PPO_BipedalWalker-v3_model_1694988276")

    print("##############################")
    test_running_award = 0
    for step in range(1,total_test_episodes+1):
        ep_reward = 0
        obs,_ = env.reset() 
        for t in range(1,max_ep_length):
            action,_ = test_agent.agent.select_action(torch.tensor(obs,dtype=torch.float,device=device))
            obs,reward,done,_,_ = env.step(action)
            ep_reward += reward
            if done:
                break
        
        test_running_award += ep_reward
        print(f"episode: {step} \n Reward: {round(ep_reward,2)}")
        ep_reward = 0
    env.close()

    avg_test_reward = test_running_award/total_test_episodes
    print(f"avg_test_rew: {avg_test_reward}")

if __name__ == "__main__":
    test()