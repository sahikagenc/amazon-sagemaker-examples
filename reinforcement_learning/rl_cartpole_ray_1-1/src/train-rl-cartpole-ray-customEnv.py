import json
import os

import gym
import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env

from sagemaker_rl.ray_launcher import SageMakerRayLauncher


class MyCartPole(gym.Env):
    def __init__(self, env_config):
        # pick actual env based on worker and env indexes
        self.env = gym.make('CartPole-v1')
        self.env.render(mode='rgb_array')
        # record videos
        self.env = gym.wrappers.Monitor(env=env, directory="./videos", force=True)
        # action and observation
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)
    def render(self,mode='rgb_array'):
        return self.env.render(mode='rgb_array')


class MyLauncher(SageMakerRayLauncher):

    def register_env_creator(self):
        register_env("mycartpole", lambda config: MyCartPole(config))

    def get_experiment_config(self):
        return {
          "training": {
            "env": "mycartpole",
            "run": "PPO",
            "stop": {
                "training_iteration": 10
            },
            "config": {
              "framework": "tf", # default framework is tensorflow
              "gamma": 0.99,
              "kl_coeff": 1.0,
              "num_sgd_iter": 20,
              "lr": 0.0001,
              "sgd_minibatch_size": 1000,
              "train_batch_size": 25000,
              "monitor": True,  # Record videos.
              "model": {
                 "free_log_std": True
              },
              "num_workers": (self.num_cpus-1),
              "num_gpus": self.num_gpus,
              "batch_mode": "truncate_episodes"
            }
          }
        }

if __name__ == "__main__":
    MyLauncher().train_main()
