# main.py
import yaml
import gym
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO


print("Samuel - Check if cuda is avaible to train on:", torch.cuda.is_available())
torch.cuda.set_device(1)

with open('scripts/config.yml', 'r') as f:
    env_config = yaml.safe_load(f)

env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "scripts:airsim-env-v0", 
                ip_address="127.0.0.1",
                image_shape = (480, 640, 3),
                env_config=env_config["TrainEnv"],
                step_length=4,
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# LOAD the model instead of initializing a new one
loaded_model = PPO.load("best_model.zip") 
loaded_model.set_env(env)


# Evaluate the policy
mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=10)


# Log evaluation results
print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")