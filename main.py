import yaml
import gym
import time
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, SubprocVecEnv 
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO

print("Samuel - Check if cuda is avaible to train on:", torch.cuda.is_available())
torch.cuda.set_device(0)

with open('scripts/config.yml', 'r') as f:
    env_config = yaml.safe_load(f)

env = DummyVecEnv (
    [
        lambda: Monitor(
            gym.make(
                "scripts:airsim-env-v0", 
                ip_address="127.0.0.1",
                image_shape = (720, 1280, 3),
                env_config=env_config["TrainEnv"]
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

"""
# Initialize RL algorithm type and parameters
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=0.00025,
    verbose=1,
    batch_size=32,
    train_freq=4,
    target_update_interval=10000,
    learning_starts=10000,
    #buffer_size=500000,
    buffer_size=5000, 
    #buffer_size=90000, not 3d 
    max_grad_norm=10,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    device="cpu",
    tensorboard_log="./tb_logs/",
)
"""


# Initialize RL algorithm type and parameters
model = PPO(
    policy="CnnPolicy",
    #policy_kwargs={
    #'n_lstm': 128, 
    #'layers': [64, 64],
    #'act_fun': tf.nn.relu,  # Assuming you imported TensorFlow as tf
    #'feature_extraction': 'mlp'
    #}, 
    env=env,
    learning_rate=0.0003,
    n_steps=2048, # to train
    #n_steps=8, # to train
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    device="cuda:1",
    tensorboard_log="./tb_logs/",
)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=100,
    best_model_save_path=".",
    log_path=".",
    eval_freq=10000,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# 5e5 = 500,000
# Results on the PyBullet benchmark (2M steps) using 6 seeds.
# Train for a certain number of timesteps
model.learn(
    total_timesteps=5e5,
    tb_log_name="dqn_airsim_drone_run_" + str(time.time()),
    **kwargs,
    #progress_bar=True
)


# Save policy weights
model.save("dqn_airsim_drone_policy")