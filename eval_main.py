# main.py
import yaml
import gym
import time
import torch
import wandb

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
torch.cuda.set_device(0)

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
#loaded_model = PPO.load("best_hoovering_model_v1_0_") 
loaded_model = PPO.load("newbestmodel/best_model3") 
loaded_model.set_env(env)

# Evaluate the policy
mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=10)


# Log evaluation results
print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")