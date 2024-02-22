import yaml
import gym
import time
import torch

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO

print("Samuel - Check if cuda is avaible to train on:", torch.cuda.is_available())
torch.cuda.set_device(1)

with open('scripts/config.yml', 'r') as f:
    env_config = yaml.safe_load(f)

env = DummyVecEnv (
    [
        lambda: Monitor(
            gym.make(
                "scripts:airsim-env-v0", 
                ip_address="127.0.0.1",
                image_shape = (480, 640, 3),
                env_config=env_config["TrainEnv"],
                #step_length=0.25,
                step_length=4,
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# Initialize RL algorithm type and parameters
model = PPO(
    policy="CnnPolicy",
    env=env,
    learning_rate=0.0003,
    n_steps=1024, # to train
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    device="cuda:1",
    tensorboard_log="C:/Users/14055/Documents/AirSim/tf",
)

#print(model.policy)
#exit()

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=10,
    best_model_save_path="C:/Users/14055/Documents/AirSim/",
    log_path="C:/Users/14055/Documents/AirSim/tf",
    eval_freq=1025,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# 5e5 = 500,000
# Results on the PyBullet benchmark (2M steps) using 6 seeds.
# Train for a certain number of timesteps
model.learn(
    total_timesteps=5e5,
    tb_log_name="ppo_airsim_drone_run_" + str(time.time()),
    **kwargs,
    #progress_bar=True
)


# Save policy weights
model.save("ppo_airsim_drone_policy")

