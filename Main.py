from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.registry import default_registry

import random
import numpy as np
import os
import network
from PPO import PPO
from network import FeedForwardNN
if __name__ == "__main__":
    try:
        if not os.path.isdir("results"):
            os.mkdir("results")

        # This is a non-blocking call that only loads the environment.
        print ("Script started. Please start Unity environment to start training proccess.")
        engine_channel = EngineConfigurationChannel()
        
        # don't forget to comment the following line if using environment from defaul_registry
        env = UnityEnvironment( side_channels=[engine_channel])


        # uncomment the folowing line if you want to load the environment without having to open unity
        # env = default_registry['CrawlerStaticTarget'].make(side_channels=[engine_channel])

        engine_channel.set_configuration_parameters(time_scale = 5, width=1920, height=1080) # control time scale 0.5 - half speed, 10. - 10x time
        
        #Start interacting with the environment.
        env.reset()


        # set up PPO and start training
        ppo = PPO(FeedForwardNN, env)
        try:
            ppo.load_networks("results/ppo_actor.pth", "results/ppo_critic.pth")
        except FileNotFoundError:
            print("failed to find parameter files, will create new networks...")
        #ppo.just_roll()
        ppo.learn(300000)
    finally:
        env.close()