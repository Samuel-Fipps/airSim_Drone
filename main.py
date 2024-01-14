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

from torch.utils.checkpoint import checkpoint
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np

# Define a custom CNN feature extractor
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(CustomCNNFeatureExtractor, self).__init__(observation_space, features_dim)

        # Enhanced CNN layers with Group Normalization
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        # Dynamically calculate the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_space.shape)  # Create a dummy input
            dummy_output = self.cnn(dummy_input)  # Pass the dummy input through the CNN
            flattened_size = dummy_output.shape[1]  # Get the output size

        # Linear layer before Multi-Head Attention
        self.pre_attn_linear = nn.Sequential(
            nn.Linear(flattened_size, features_dim),
            nn.ReLU(),
            nn.Dropout(0.10)
        )

        # Multi-Head Attention Layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=features_dim, num_heads=4)

        # Linear layer before LSTM
        self.linear_before_lstm = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(0.10)
        )

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=features_dim, hidden_size=features_dim, batch_first=True)

        # Final Linear Layer
        self.final_linear = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU()  # No dropout in the final layer
        )

    def forward(self, observations):

        debug = False
        if debug:
            print("hello")
            # Convert the tensor to an image format for visualization
            # Assuming observations is a 4D Tensor of shape (batch_size, channels, height, width)
            img = observations[0]  # Take the first image in the batch
            img = img.cpu().numpy()  # Convert to numpy array
            img = np.transpose(img, (1, 2, 0))  # Change the channel order for visualization

            # Normalize the image for better visualization
            img = (img - img.min()) / (img.max() - img.min())

            # Plot the image
            plt.imshow(img)
            plt.show()


        cnn_features = self.cnn(observations)
        pre_attn_linear = self.pre_attn_linear(cnn_features)
        attn_output, _ = self.multihead_attn(pre_attn_linear.unsqueeze(0), pre_attn_linear.unsqueeze(0), pre_attn_linear.unsqueeze(0))
        linear_before_lstm = self.linear_before_lstm(attn_output)
        lstm_out, _ = self.lstm(linear_before_lstm)
        return self.final_linear(lstm_out.squeeze(0))



# CustomCnnPolicy with the same MLP structure as the original
class CustomCnnPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.ReLU, *args, **kwargs):
        super(CustomCnnPolicy, self).__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, *args, **kwargs, 
                                              features_extractor_class=CustomCNNFeatureExtractor,
                                              features_extractor_kwargs=dict(features_dim=512))



print("Samuel - Check if cuda is avaible to train on:", torch.cuda.is_available())
#torch.cuda.set_device(0)

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
    n_steps=1024,  # to train
    batch_size=32,
    n_epochs=50,
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