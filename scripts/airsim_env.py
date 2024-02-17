from . import airsim
import gym
import numpy as np
import airsim
import math
import time
import gym
import cv2

first_pass = 1
counter = 0
start_time = 0
frame_time = 0
bus_finder_model = None
device = None
processor_Depth_Finder_Model = None
Depth_Finder_Model = None
prevous_depth = 0
current_degree = -5
frame_count = 0
current_yaw = 0 
how_many_in_a_row = 0
image_counter = 0
start_time_1 = 0
prevous_distance_from_bus = 100000
elapsed_time_2 =0
step_counter = 0
continues_to_be_bad = 1
within_distance_counter = 0 
pic_counter = 0
debug_counter = 0
end_of_maze_counter = 0
win_counter = 0

def interpolate_velocity(current_v, target_v, steps=10):
    return np.linspace(current_v, target_v, steps)

class AirSimDroneEnv(gym.Env):
    def __init__(self, ip_address, image_shape, env_config, step_length):
        self.image_shape = image_shape
        self.sections = env_config["sections"]

        self.step_length = step_length

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(9)
        #self.action_space = gym.spaces.Discrete(7)

        self.info = {"collision": False}

        self.last_position = None
        self.last_time = None


        self.collision_time = 0
        self.random_start = True
        self.current_vy = 0  # Initialize current vertical velocity
        self.current_vz = 0 
        self.current_speed = 0
        self.setup_flight()

    def step(self, action):
        #self.do_action(action)
        self.do_action(action)
        obs, info = self.get_obs()
        reward, done = self.compute_reward()
        return obs, reward, done, info

    def reset(self):
        self.setup_flight()
        obs, _ = self.get_obs()
        return obs

    def render(self):
        return self.get_obs()

    def setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Prevent drone from falling after reset
        #self.drone.moveToZAsync(-1, 1)

        #self.drone.moveToPositionAsync(-20.55265, 0.9786, -10.0225, 5).join()
        #self.drone.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians( current_degree-5 ), 0, 0)))
        

        self.drone.moveByRollPitchYawrateZAsync(0, 0, 0, self.drone.getMultirotorState().kinematics_estimated.position.z_val, 1).join()
        #self.drone.moveToPositionAsync(-0.55265, 0.9786, -1.0225, 5).join()
        time.sleep(1)
        # Get collision time stamp
        self.collision_time = self.drone.simGetCollisionInfo().time_stamp



    def do_action(self, select_action):
        """was speed = 1.0 and worked"""
        #speed = 3.0
        speed = 3.0
        if select_action == 0:
            vy, vz = (-speed, -speed)
        elif select_action == 1:
            vy, vz = (0, -speed)
        elif select_action == 2:
            vy, vz = (speed, -speed)
        elif select_action == 3:
            vy, vz = (-speed, 0)
        elif select_action == 4:
            vy, vz = (0, 0)
        elif select_action == 5:
            vy, vz = (speed, 0)
        elif select_action == 6:
            vy, vz = (-speed, speed)
        elif select_action == 7:
            vy, vz = (0, speed)
        else:
            vy, vz = (speed, speed)

        
        self.drone.moveByVelocityBodyFrameAsync(speed, vy, vz, duration=0.1).join()

    def get_obs(self):
        self.info["collision"] = self.is_collision()
        obs = self.get_rgb_image()
        return obs, self.info


    def compute_reward(self):
        global previous_x
        global end_of_maze_counter
        global win_counter

        end_of_maze_counter+=1

        # Get current x position
        pose = self.drone.simGetVehiclePose().position
        x1 = pose.x_val

        # Compute x difference traveled
        if 'previous_x' not in globals():
            previous_x = x1  # Initialize if not present
        x_difference = previous_x - x1
        previous_x = x1  # Update previous_x for the next step

        # Use x_difference as the reward
        reward = x_difference 
        
        #keeps wanting to just hover in place, counter this:
        if reward < 0.1 and reward >= 0: reward = -0.1

        done = False
        if end_of_maze_counter > 350:
            done = True
            reward = 100
            end_of_maze_counter = 0
            win_counter +=1
            print("Win Number: ", win_counter)


        if self.is_collision():
            end_of_maze_counter = 0
            done = True
            reward = -10  # Penalty for collision
            previous_x = 0

        return reward, done



    def is_collision(self):
        current_collision_time = self.drone.simGetCollisionInfo().time_stamp
        return True if current_collision_time != self.collision_time else False
    
    def get_rgb_image(self):
        #global pic_counter 
        rgb_image_request = airsim.ImageRequest(1, airsim.ImageType.Scene, False, False)
        responses = self.drone.simGetImages([rgb_image_request])
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3)) 

        #pic_counter += 1
        # Save the image using OpenCV
        #save_path = "D:\\Home\\Images\\"+str(pic_counter)+".png"
        #cv2.imwrite(save_path, img2d)

        # Sometimes no image returns from api
        try:
            return img2d.reshape(self.image_shape)
        except:
            print("No picuture")
            return np.zeros((self.image_shape))

    def get_depth_image(self, thresh = 2.0):
        depth_image_request = airsim.ImageRequest(1, airsim.ImageType.DepthPerspective, True, False)
        responses = self.drone.simGetImages([depth_image_request])
        depth_image = np.array(responses[0].image_data_float, dtype=np.float32)
        depth_image = np.reshape(depth_image, (responses[0].height, responses[0].width))
        depth_image[depth_image>thresh]=thresh
        return depth_image


class TestEnv(AirSimDroneEnv):
    def __init__(self, ip_address, image_shape, env_config):
        self.eps_n = 0
        super(TestEnv, self).__init__(ip_address, image_shape, env_config)
        self.agent_traveled = []
        self.random_start = False

    def setup_flight(self):
        super(TestEnv, self).setup_flight()
        self.eps_n += 1

        # Start the agent at a random yz position
        y_pos, z_pos = (0,0)
        pose = airsim.Pose(airsim.Vector3r(self.agent_start_pos,y_pos,z_pos))
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)
        
    def compute_reward(self):
        reward = 0
        done = 0

        x,_,_ = self.drone.simGetVehiclePose().position

        if self.is_collision():
            done = 1
            self.agent_traveled.append(x)
    
        if done and self.eps_n % 5 == 0:
            print("---------------------------------")
            print("> Total episodes:", self.eps_n)
            print("> Flight distance (mean): %.2f" % (np.mean(self.agent_traveled)))
            print("> Holes reached (max):", int(np.max(self.agent_traveled)//4))
            print("> Holes reached (mean):", int(np.mean(self.agent_traveled)//4))
            print("---------------------------------\n")
        
        return reward, done
