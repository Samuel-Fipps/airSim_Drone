import yaml
import gym
import time
import torch
import wandb
import numpy as np


from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback



class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super(CustomEvalCallback, self).__init__(*args, **kwargs)

    def _on_step(self) -> bool:
        super()._on_step()
        if self.n_calls % self.eval_freq == 0:
            # Access the first environment in the vector (assuming you have a vector environment)
            eval_env = self.eval_env.envs[0]

            # Check if the environment is a Monitor wrapper and retrieve the statistics
            if hasattr(eval_env, 'get_episode_rewards'):
                latest_results = eval_env.get_episode_rewards()
                latest_lengths = eval_env.get_episode_lengths()

                # Compute the mean and standard deviation
                mean_reward = np.mean(latest_results)
                std_reward = np.std(latest_results)
                mean_length = np.mean(latest_lengths)
                std_length = np.std(latest_lengths)

                # Log to wandb
                wandb.log({
                    "eval/mean_reward": mean_reward,
                    "eval/std_reward": std_reward,
                    "eval/mean_length": mean_length,
                    "eval/std_length": std_length
                })
            else:
                print("Monitor wrapper not found in evaluation environment.")
        return True





print("Samuel - Check if cuda is avaible to train on:", torch.cuda.is_available())
torch.cuda.set_device(0)

with open('scripts/config.yml', 'r') as f:
    env_config = yaml.safe_load(f)

env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "scripts:airsim-env-v0", 
                ip_address="127.0.0.1",
                image_shape = (720, 1280, 3),
                env_config=env_config["TrainEnv"],
                step_length=4,
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# LOAD the model instead of initializing a new one
loaded_model = PPO.load("F:/Everything/PPO-based-Autonomous-Navigation-for-Quadcopters/best_model_2") # first: 5, second: 5, third: 0
#loaded_model = PPO.load("D:/Everything/PPO-based-Autonomous-Navigation-for-Quadcopters/best_model_speed_3") # first: 5, second: 5, third: 0
#loaded_model = PPO.load("D:/Everything/PPO-based-Autonomous-Navigation-for-Quadcopters/newbestmodel/best_model_2") # first: 5, second: 5, third: 0
loaded_model.set_env(env)

# Start logging
wandb.init(project='Drone',
            config={
            "speed": 3,
            }
)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = CustomEvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=25,
    best_model_save_path="newbestmodel/",
    log_path=".",
    eval_freq=10000,
)
callbacks.append(eval_callback)



kwargs = {}
kwargs["callback"] = callbacks


# Continue training the loaded model
# 5e5 = 500,000
# Results on the PyBullet benchmark (2M steps) using 6 seeds.
# Train for a certain number of timesteps
loaded_model.learn(
    total_timesteps=4e6,
    tb_log_name="ppo_airsim_drone_run_" + str(time.time()),
    **kwargs
)

# Save the further trained model
loaded_model.save("ppo_airsim_drone_policy_continued_bad")
wandb.finish()