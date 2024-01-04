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
target_yaw = 0  

def interpolate_velocity(current_v, target_v, steps=10):
    return np.linspace(current_v, target_v, steps)

class AirSimDroneEnv(gym.Env):
    def __init__(self, ip_address, image_shape, env_config, step_length):
        self.image_shape = image_shape
        self.sections = env_config["sections"]

        self.step_length = step_length

        self.drone = airsim.MultirotorClient(ip=ip_address)
        # Define the range for each parameter
        max_linear_speed = 1.0  # Example maximum linear speed
        max_angular_speed = 1.0  # Example maximum angular speed

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=image_shape, dtype=np.uint8
                )


        # Update action space to a Box space
        self.action_space = gym.spaces.Box(
            low=np.array([-max_linear_speed, -max_linear_speed, -max_linear_speed, -max_angular_speed, -max_angular_speed, -max_angular_speed]),
            high=np.array([max_linear_speed, max_linear_speed, max_linear_speed, max_angular_speed, max_angular_speed, max_angular_speed]),
            dtype=np.float32
        )

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
        

        self.drone.moveByRollPitchYawrateZAsync(0, 0, 6.3, self.drone.getMultirotorState().kinematics_estimated.position.z_val, 1).join()
        self.drone.moveToPositionAsync(-0.55265, 0.9786, -1.0225, 5).join()
        time.sleep(1)
        # Get collision time stamp
        self.collision_time = self.drone.simGetCollisionInfo().time_stamp



    def do_action(self, action):
        vx, vy, vz, roll, pitch, yaw = action

        # Convert numpy.float32 to native Python floats
        vx = float(vx)
        vy = float(vy)
        vz = float(vz)
        roll = float(roll)
        pitch = float(pitch)
        yaw = float(yaw)


        # Directly use the action values for drone control
        # Note: Ensure that these values are within the acceptable range for your drone's API
        #self.drone.moveByVelocityAsync(vx, vy, vz, duration=0.3).join()
        #self.drone.moveByRollPitchYawZAsync(roll, pitch, yaw, vz, duration=0.3).join()
        self.drone.moveByVelocityAsync(vx, vy, vz, duration=0.1).join()
        self.drone.moveByRollPitchYawZAsync(roll, pitch, yaw, vz, duration=0.1).join()


    def get_obs(self):
        self.info["collision"] = self.is_collision()
        obs = self.get_rgb_image()
        return obs, self.info



    def convert_to_euler_angles(self, quaternion):
        """
        Convert a quaternion to Euler angles (roll, pitch, yaw).
        Quaternion is assumed to be in the format of [w, x, y, z].
        """

        w, x, y, z = quaternion.w_val, quaternion.x_val, quaternion.y_val, quaternion.z_val

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
    


    def compute_reward(self):
        global starting_pose
        global target_yaw 


        # Get current position
        pose = self.drone.simGetVehiclePose().position
        x1, y1, z1 = pose.x_val, pose.y_val, pose.z_val

        # Initialize starting_pose if not present
        if 'starting_pose' not in globals():
            starting_pose = (x1, y1, z1)

        # Get current position and orientation
        pose = self.drone.simGetVehiclePose()
        position = pose.position
        orientation_q = pose.orientation
        roll, pitch, yaw = self.convert_to_euler_angles(orientation_q)  # Convert quaternion to Euler angles

        # Desired orientation - assuming forward facing is represented by yaw = 0
        if target_yaw == 0:
            target_yaw = yaw
        yaw_deviation_threshold = 0.45  # Acceptable yaw deviation threshold

        # Calculate distance from the starting position
        distance_from_start = np.sqrt((position.x_val - starting_pose[0])**2 + 
                                    (position.y_val - starting_pose[1])**2 + 
                                    (position.z_val - starting_pose[2])**2)

        # Calculate yaw deviation from target
        yaw_deviation = abs(yaw - target_yaw)

        # Compute position-based reward
        if distance_from_start < 0.5:
            position_reward = 100
        else:
            position_reward = -distance_from_start * 3

        # Compute orientation-based reward
        if yaw_deviation < yaw_deviation_threshold:
            orientation_reward = 10
        else:
            orientation_reward = -yaw_deviation * 3  # Scale as needed

        #print("position_reward:", position_reward)
        #print("orientation_reward:", orientation_reward)

        # Combine rewards
        reward = position_reward + orientation_reward

        # Collision check
        done = False
        if self.is_collision():
            done = True
            reward = -200  # Large penalty for collision

        return reward, done


    import math



    def compute_reward_bac_v2(self):
        # Hovering
        global starting_pose

        # Get current position
        pose = self.drone.simGetVehiclePose().position
        x1, y1, z1 = pose.x_val, pose.y_val, pose.z_val

        # Initialize starting_pose if not present
        if 'starting_pose' not in globals():
            starting_pose = (x1, y1, z1)

        # Calculate distance from the starting position
        distance_from_start = ((x1 - starting_pose[0])**2 + (y1 - starting_pose[1])**2 + (z1 - starting_pose[2])**2) ** 0.5

        # Reward for hovering (small or no movement from the starting position)
        if distance_from_start < 0.5:  # Threshold for considering as hovering
            reward = 1
        else:
            reward = -distance_from_start*5  # Penalize for moving away from the start

        # Check for collision
        done = False
        if self.is_collision():
            done = True
            reward = -100  # Penalty for collision

        #print(reward)
        return reward, done


    def compute_reward_bac(self):
        #Hmoving forward
        global previous_x

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
        if self.is_collision():
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
