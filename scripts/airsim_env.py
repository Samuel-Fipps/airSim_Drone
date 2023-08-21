from . import airsim
import gym
import numpy as np
import airsim
import math
import time
import gym


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


class AirSimDroneEnv(gym.Env):
    def __init__(self, ip_address, image_shape, env_config):
        self.image_shape = image_shape
        self.sections = env_config["sections"]

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(9)

        self.info = {"collision": False}

        self.collision_time = 0
        self.random_start = True
        self.setup_flight()

    def step(self, action):
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

        self.drone.moveToPositionAsync(-20.55265, 0.9786, -10.0225, 5).join()
        self.drone.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians( current_degree-5 ), 0, 0)))
        

        self.drone.moveByRollPitchYawrateZAsync(0, 0, 2.85, self.drone.getMultirotorState().kinematics_estimated.position.z_val, 1).join()
        self.drone.moveToPositionAsync(-20.55265, 0.9786, -10.0225, 5).join()

        # Get collision time stamp
        self.collision_time = self.drone.simGetCollisionInfo().time_stamp

    def do_action_new_with_rotate(self, select_action):
        speed = 1.5
        yaw_rate = 45  # This is an example value, representing 45 degrees per second. Adjust as needed.
        vy, vz, yaw = 0, 0, 0  # Initialize velocities and yaw rate

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
        elif select_action == 8:
            vy, vz = (speed, speed)
        elif select_action == 9:
            yaw = yaw_rate  # Rotate clockwise
        elif select_action == 10:
            yaw = -yaw_rate  # Rotate counter-clockwise

        # Execute action
        # Assuming the drone API has a method to control yaw. If not, you'll need to find the appropriate method.
        self.drone.moveByVelocityBodyFrameAsync(speed, vy, vz, yaw_rate=yaw, duration=1).join()

        # Prevent swaying and rotation
        self.drone.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1)
        # Assuming the drone API has a method to stop yaw rotation. If not, you'll need to find the appropriate method.
        self.drone.stopYaw()

    def do_action(self, select_action):
        speed = 1.5
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

        # Execute action
        self.drone.moveByVelocityBodyFrameAsync(speed, vy, vz, duration=1).join()

        # # Prevent swaying
        self.drone.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1)

    def get_obs(self):
        self.info["collision"] = self.is_collision()
        obs = self.get_rgb_image()
        return obs, self.info


    def compute_reward(self):
        global first_pass
        global counter
        global frame_time
        global start_time
        thresh_dist = 2
        beta = 1
        global prevous_depth
        global current_degree
        global frame_count 
        global current_yaw, start_time_1
        global prevous_distance_from_bus, elapsed_time_2, continues_to_be_bad, step_counter
        global within_distance_counter 

        
        # big code change time ===================================================================

        if(first_pass):
            #self.real_first_pass()
            first_pass = 0
            start_time = time.time()

        #depth, closest_edge = self.first_pass()
        depth = None
        closest_edge = None

        #print(closest_edge)
        if closest_edge == "right": # turn right 25 degrees
            current_yaw += 10
            self.drone.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians(current_degree), 0, math.radians(current_yaw))))
            #time.sleep(0.2)

        if closest_edge == "Skip_right": # turn right 25 degrees
            current_yaw += 20
            self.drone.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians(current_degree), 0, math.radians(current_yaw))))
            #time.sleep(0.2)

        if closest_edge == "left": # turn left 25 degrees
            current_yaw -= 10
            self.drone.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians(current_degree), 0, math.radians(current_yaw))))
            #time.sleep(0.2)

        if closest_edge == "bottom":
            current_degree -= 10
            self.drone.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians(current_degree), 0, math.radians(current_yaw))))
            #time.sleep(0.2)

        if closest_edge == "top":
            current_degree += 10
            self.drone.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians(current_degree), 0, math.radians(current_yaw))))
            #time.sleep(0.2)

        #exit()

        """
        # frame counter
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1: 
            #print(f"FPS: {frame_count}")
            frame_count = 0
            start_time = time.time()
        """

        """
        #60 second resest timer:
        done_here = False
        elapsed_time_2 = time.time() - start_time_1
        if elapsed_time_2 >= 120:
            #print("=====================================================")
            #print("=======================Reseting======================")
            #print("=====================================================")
            #self.drone.reset()
            done_here = True
            start_time_1 = time.time()
        #print("Time: ", elapsed_time_2)
        """
        #step_counter += 1
        #if step_counter % 2000 == 0: print("Current Step:", step_counter, "Out of 500,000")
        
        # stop at 200 steps
        done_here = False
        elapsed_time_2 += 1
        if elapsed_time_2 >= 120:
            done_here = True
            elapsed_time_2 = 0

        # coord reward
        pose = self.drone.simGetVehiclePose().position
        x1 = pose.x_val
        y1 = pose.y_val
        z1 = pose.z_val
        #print(pose)

        # bus point
        x2, y2, z2 = -110, 0, 0

        # Calculate the distance using Euclidean distance formula
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        if distance < prevous_distance_from_bus: 
            reward_for_distance_to_bus = prevous_distance_from_bus - distance
            continues_to_be_bad = 1
        else: 
            reward_for_distance_to_bus = (prevous_distance_from_bus - distance)
            if reward_for_distance_to_bus > -2:
                reward_for_distance_to_bus = -2 
            continues_to_be_bad += 1

        reward_for_distance_to_bus = reward_for_distance_to_bus * continues_to_be_bad  # new
        prevous_distance_from_bus = distance


        # if distance is less then 10 for 40 steps then reward 500 and done 
        # and take away negtives 
        if distance < 10:
            within_distance_counter += 1
            if reward_for_distance_to_bus < 0: 
                reward_for_distance_to_bus = 0
        else:
            within_distance_counter = 0
        
        reward_for_goal = 0
        if within_distance_counter >= 40:
            reward_for_goal = 10000


        # TODO if distance  is to close + reward and reset

        """
        # depth reward
        if depth != None:
            if depth < prevous_depth: reward_for_depth = 10
            elif depth > prevous_depth: reward_for_depth = -10
            else: reward_for_depth = -1
            prevous_depth = depth
        else: reward_for_depth = 0
        if depth == None:
            reward_for_depth = -10
        #print("depth: ", depth, "reward_for_depth: ", reward_for_depth)
        """

        # collision reward ===================================================================
        if self.is_collision():
            #reward = -5000
            collision = True
            elapsed_time_2 = 0

        
        else:
            
            collision = False
            '''
            dist = 10000000
            for i in range(0, len(pts) - 1):
                dist = min(dist, np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i + 1])))
                    / np.linalg.norm(pts[i] - pts[i + 1]),
                )

            #print(dist)
            if dist < thresh_dist:
                reward = -10

            else:
                
                reward_dist = math.exp(-beta * dist) - 0.5
                reward_dist = 0 # toookout out distance reawrd doesn't work!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                reward_speed = (
                    np.linalg.norm(
                        [
                            self.state["velocity"].x_val,
                            self.state["velocity"].y_val,
                            self.state["velocity"].z_val,
                        ]
                    )
                    - 0.5
                )
                #print(reward_speed)
                if (reward_speed < .5): reward_speed = -2
                elif(reward_speed>10):reward_speed = 10
                reward = reward_dist + reward_speed
                if(reward > 20): reward = 10
                reward += reward_for_depth
                '''
            
        reward = reward_for_distance_to_bus

        done = 0
        if collision: 
            done = True
            reward = -300
        if done_here: 
            done = True

        if  reward_for_goal == 10000:
            reward = 10000
            done = True

        #print("Final reward: ", reward)
        return reward, done

    def compute_reward_his(self):
        reward = 0
        done = 0

        # Target distance based reward
        x,y,z = self.drone.simGetVehiclePose().position
        target_dist_curr = np.linalg.norm(np.array([y,-z]) - self.target_pos)
        reward += (self.target_dist_prev - target_dist_curr)*20

        self.target_dist_prev = target_dist_curr

        # Get meters agent traveled
        agent_traveled_x = np.abs(self.agent_start_pos - x)

        # Alignment reward
        if target_dist_curr < 0.30:
            reward += 12
            # Alignment becomes more important when agent is close to the hole 
            if agent_traveled_x > 2.9:
                reward += 7

        elif target_dist_curr < 0.45:
            reward += 7

        # Collision penalty
        if self.is_collision():
            reward = -100
            done = 1

        # Check if agent passed through the hole
        elif agent_traveled_x > 3.7:
            reward += 10
            done = 1

        # Check if the hole disappeared from camera frame
        # (target_dist_curr-0.3) : distance between agent and hole's end point
        # (3.7-agent_traveled_x) : distance between agent and wall
        # (3.7-agent_traveled_x)*sin(60) : end points that camera can capture
        # FOV : 120 deg, sin(60) ~ 1.732 
        elif (target_dist_curr-0.3) > (3.7-agent_traveled_x)*1.732:
            reward = -100
            done = 1

        return reward, done

    def is_collision(self):
        current_collision_time = self.drone.simGetCollisionInfo().time_stamp
        return True if current_collision_time != self.collision_time else False
    
    def get_rgb_image(self):
        rgb_image_request = airsim.ImageRequest(1, airsim.ImageType.Scene, False, False)
        responses = self.drone.simGetImages([rgb_image_request])
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3)) 

        # Sometimes no image returns from api
        try:
            return img2d.reshape(self.image_shape)
        except:
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
