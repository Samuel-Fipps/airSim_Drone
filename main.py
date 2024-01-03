import yaml
import gym
import time
import torch

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

# Define a custom CNN feature extractor
class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(CustomCNNFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Define custom convolutional layers
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # First Conv layer
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Second Conv layer
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # Third Conv layer
            nn.ReLU(),
            nn.Flatten(),  # Flatten the output
        )

        # Assume the input features to the linear layer based on the Flatten layer output
        self.linear = nn.Sequential(
            nn.Linear(858624, features_dim),  # Adjust the input dimension as needed
            nn.ReLU()
        )

    def forward(self, observations):
        features = self.cnn(observations)
        return self.linear(features)

# CustomCnnPolicy with the same MLP structure as the original
class CustomCnnPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.ReLU, *args, **kwargs):
        super(CustomCnnPolicy, self).__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, *args, **kwargs, 
                                              features_extractor_class=CustomCNNFeatureExtractor,
                                              features_extractor_kwargs=dict(features_dim=512))


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
                #step_length=0.25,
                step_length=4,
            )
        )
    ]
)

env = VecTransposeImage(env)

# Initialize RL algorithm type and parameters
model = PPO(
    policy=CustomCnnPolicy,  # Using the custom policy
    env=env,
    learning_rate=0.0003,
    n_steps=2048,  # to train
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    device="cuda:1",
    tensorboard_log="./tb_logs/",
)

#print(model.policy)
#exit()

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=25,
    best_model_save_path=".",
    log_path=".",
    eval_freq=10000,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=5e6,
    tb_log_name="ppo_airsim_drone_run_" + str(time.time()),
    **kwargs,
    #progress_bar=True
)

# Save policy weights
model.save("ppo_airsim_drone_policy")





# 5e5 = 500,000
# 5e6 = 5,000,000
# Results on the PyBullet benchmark (2M steps) using 6 seeds.
# Train for a certain number of timesteps