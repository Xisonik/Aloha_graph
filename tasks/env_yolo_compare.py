import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import carb
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import clip
import torchvision.transforms as T
from typing import Optional
from scipy.special import expit
from pprint import pprint 
from dataclasses import asdict, dataclass
from configs.main_config import MainConfig

from ultralytics import YOLO
import cv2

from embed_nn import SceneEmbeddingNetwork
import torch.optim as optim

import os
import gzip
import pickle
import argparse
import json

sim_config = {
    "renderer": "RayTracedLighting",
    "headless": True,
    #"headless": False,
    "multi_gpu": False, 
    #"active_gpu": gpu_to_use,
    "enable":"omni.kit.livestream.native"
}

GET_DIR = False

def euler_from_quaternion(vec):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x, y, z, w = vec[0], vec[1], vec[2], vec[3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def get_quaternion_from_euler(roll,yaw=0, pitch=0):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return np.array([qx, qy, qz, qw])


class AlphaBaseEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        config = MainConfig(),
        skip_frame=4,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=1200,
        seed=10,
        MAX_SR=50,
        test=False,
        reward_mode=0
    ) -> None:
        from omni.isaac.kit import SimulationApp
        self.config = config
        sim_config["headless"] = asdict(config).get('headless', None)
        self._simulation_app = SimulationApp(sim_config)
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)
        from omni.isaac.core import World
        from .wheeled_robot import WheeledRobot
        from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
        from omni.isaac.core.objects import VisualCuboid, FixedCuboid
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.core.utils.prims import create_prim, define_prim, delete_prim

        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
        self._my_world.scene.add_default_ground_plane()
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
        jetbot_asset_path = asdict(config).get('jetbot_asset_path', None)
        room_usd_path = asdict(config).get('room_usd_path', None)
        create_prim(
                    prim_path=f"/room",
                    translation=(0, 0, 0),
                    usd_path=room_usd_path,
                )

        self.jetbot = self._my_world.scene.add(
            WheeledRobot(
                prim_path="/jetbot",
                name="my_jetbot",
                wheel_dof_names=["left_wheel", "right_wheel"],
                create_robot=True,
                usd_path=jetbot_asset_path,
                position=np.array([7, 0, 0.0]),
                orientation=get_quaternion_from_euler(0),
            )
        )
        from pxr import PhysicsSchemaTools, UsdUtils, PhysxSchema, UsdPhysics
        from pxr import Usd
        from omni.physx import get_physx_simulation_interface
        import omni.usd
        self.my_stage = omni.usd.get_context().get_stage()
        self.my_prim = self.my_stage.GetPrimAtPath("/jetbot")

        contactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(self.my_prim)
        contact_report_sub = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact_report_event)
        create_prim(
                    prim_path=f"/cup",
                    #translation=(0, 0.22, 0),
                    position=np.array([10.0,0.0,0.0]),
                    usd_path=asdict(config).get('cup_usd_path', None),
                )
        self.jetbot_controller = DifferentialController(name="simple_control", wheel_radius=0.068, wheel_base=0.34)
        #if GET_DIR:
        self.goal_cube = self._my_world.scene.add(
            VisualCuboid(
                prim_path="/new_cube_1",
                name="visual_cube",
                position=np.array([15.0,0.0,0.0]),
                size=0.3,
                color=np.array([0, 1.0, 0]),
            )
        )
        self.goal_position = np.array([0,0,0])
        self.render_products = []
        from omni.replicator.isaac.scripts.writers.pytorch_writer import PytorchWriter
        from omni.replicator.isaac.scripts.writers.pytorch_listener import PytorchListener
        import omni.replicator.core as rep
        self.image_resolution = 400
        self.camera_width = self.image_resolution
        self.camera_height = self.image_resolution
        camera_paths = room_usd_path = asdict(config).get('camera_paths', None)

        render_product = rep.create.render_product(camera_paths, resolution=(self.camera_width, self.camera_height))
        self.render_products.append(render_product)

        # initialize pytorch writer for vectorized collection
        self.pytorch_listener = PytorchListener()
        self.pytorch_writer = rep.WriterRegistry.get("PytorchWriter")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#"cpu")
        print("device = ", self.device)
        self.pytorch_writer.initialize(listener=self.pytorch_listener, device=self.device)
        self.pytorch_writer.attach(self.render_products)

        self.seed(seed)
        self.reward_range = (-10000, 10000)
        
        gym.Env.__init__(self)
        self.action_space = spaces.Box(low=-1, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1000000000, high=1000000000, shape=(1542,), dtype=np.float32)

        self.max_velocity = 1.2
        self.max_angular_velocity = math.pi*0.4
        self.events = [0, 1]#[0, 1, 2]
        self.event = 0

        convert_tensor = transforms.ToTensor()

        clip_model, clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess

        goal_path = asdict(config).get('goal_image_path', None)

        img_goal = clip_preprocess(Image.open(goal_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.img_goal_emb = self.clip_model.encode_image(img_goal)
            self.start_emb = self.img_goal_emb

        self.model = YOLO("yolov8m-seg.pt")
        #self.model.to('cuda')
        self.stept = 0
        
        self.collision = False
        self.start_step = True
        self.MAX_SR = MAX_SR
        self.num_of_step = 0
        self.steps_array = []
        self.reward_modes = ["move", "rotation"]#, "Nan"]
        self.reward_mode = asdict(config).get('reward_mode', None)
        self.local_reward_mode = 0
        self.delay_change_RM = 0
        self.prev_SR = {}
        self._test = test
        self.log_path = asdict(config).get('log_path', None)

        self.training_mode = asdict(config).get('training_mode', None)
        self.local_training_mode = 0
        self.traning_radius = 0
        self.trining_delta_angle = 0
        self.max_traning_radius = 4
        self.max_trining_angle = np.pi/6
        self.amount_angle_change = 0
        self.amount_radius_change = 0
        self.max_amount_angle_change = 4
        self.max_amount_radius_change = 60
        self.repeat = 5
        self.change_line = 0
        self.num_of_envs = 0
        torch.save(torch.tensor([0]), '/home/kit/Downloads/loss.pt')

        import omni.isaac.core.utils.prims as prim_utils

        light_1 = prim_utils.create_prim(
            "/World/Light_1",
            "SphereLight",
            position=np.array([2.5, 5.0, 20.0]),
            attributes={
                "inputs:radius": 0.1,
                "inputs:intensity": 5e7,
                "inputs:color": (1.0, 1.0, 1.0)
            }
)
        self.init_embedding_nn()

        return
    def init_embedding_nn(self):
        device  = self.device
        self.embedding_net = SceneEmbeddingNetwork(object_feature_dim=518).to(device)
        self.embedding_optimizer = optim.Adam(self.embedding_net.parameters(), lr=0.001)

    
    def get_success_rate(self, observation, terminated, sources, source="Nan"):
        #cut_observation = list(observation.items())[0:6]
        self._insert_step(self.steps_array, self.num_of_step, self.event, observation, terminated, source)
        pprint(self.steps_array)
        print("summary")
        pprint(self._calculate_SR(self.steps_array, self.events, sources))
        
    def _insert_step(self, steps_array, i, event, observation, terminated, source):
         steps_array.append({
            "i": i,
            "event": event,
            "terminated": terminated,
            "source": source,
            "observation": observation,
            })
         if len(steps_array) > self.MAX_SR:
            steps_array.pop(0)

    def _calculate_SR(self, steps_array, events, sources):
        SR = 0
        SR_distribution = dict.fromkeys(events,0)
        step_distribution = dict.fromkeys(events,0)
        FR_distribution = dict.fromkeys(sources, 0)
        FR_len = 0
        for step in steps_array:
            step_distribution[step["event"]] += 1
            if step["terminated"] is True:
                SR += 1
                SR_distribution[step["event"]] += 1
            else:
                FR_distribution[step["source"]] += 1
                FR_len += 1

        for source in sources:
            if FR_len > 0:
                FR_distribution[source] = FR_distribution[source]/FR_len
        for event in events:
            if step_distribution[event] > 0:
                SR_distribution[event] = SR_distribution[event]/step_distribution[event]

        SR = SR/len(steps_array)
        self.prev_SR = SR_distribution
        return  SR, SR_distribution, FR_distribution
    
    def _get_dt(self):
        return self._dt

    def _is_collision(self):
        if self.collision:
            print("collision error!")
            self.collision = False
            return True 
        return False

    def _get_current_time(self):
        return self._my_world.current_time_step_index - self._steps_after_reset

    def _is_timeout(self):
        if self._get_current_time() >= self._max_episode_length:
            print("time out")
            return True
        return False

    def get_quadrant(self, nx, ny, vector):
        LR = vector[0]*nx[1] - vector[1]*nx[0]
        mult = 1
        if LR < 0:
            mult = -1
        return mult

    def get_gt_observations(self, previous_jetbot_position, previous_jetbot_orientation):
        goal_world_position = self.goal_position
        current_jetbot_position, current_jetbot_orientation = self.jetbot.get_world_pose()
        jetbot_linear_velocity = self.jetbot.get_linear_velocity()
        jetbot_angular_velocity = self.jetbot.get_angular_velocity()
        entrance_world_position = np.array([0.0, 0.0])

        if self.event == 0:
            dif = 0.9
            entrance_world_position[0] = goal_world_position[0] - dif
            entrance_world_position[1] = goal_world_position[1] - dif
        elif self.event == 1:
            entrance_world_position[0] = goal_world_position[0] + 1
            entrance_world_position[1] = goal_world_position[1]
        else:
            entrance_world_position[0] = goal_world_position[0]
            entrance_world_position[1] = goal_world_position[1] - 1
        goal_world_position[2] = 0

        current_dist_to_goal = np.linalg.norm(goal_world_position[0:2] - current_jetbot_position[0:2])

        nx = np.array([-1,0])
        ny = np.array([0,1])
        to_goal_vec = (goal_world_position - current_jetbot_position)[0:2]
        quadrant = self.get_quadrant(nx, ny, to_goal_vec)
        cos_angle = np.dot(to_goal_vec, nx) / np.linalg.norm(to_goal_vec) / np.linalg.norm(nx)
        delta_angle = math.degrees(abs(euler_from_quaternion(current_jetbot_orientation)[0] - quadrant*np.arccos(cos_angle)))
        orientation_error = delta_angle if delta_angle < 180 else 360 - delta_angle

        observation = {
            "entrance_world_position": entrance_world_position, 
            "goal_world_position": goal_world_position, 
            "current_jetbot_position": current_jetbot_position, 
            "current_jetbot_orientation":math.degrees(euler_from_quaternion(current_jetbot_orientation)[0]),
            "jetbot_to_goal_orientation":math.degrees(quadrant*np.arccos(cos_angle)),
            "jetbot_linear_velocity": jetbot_linear_velocity,
            "jetbot_angular_velocity": jetbot_angular_velocity,
            "delta_angle": delta_angle,
            "current_dist_to_goal": current_dist_to_goal,
            "orientation_error": orientation_error,
        }
        return observation

    def change_reward_mode(self):
        if self.start_step:
            self.start_step = False
            if self.delay_change_RM < self.MAX_SR:
                self.delay_change_RM += 1
            else:
                print("distrib SR", list(self.prev_SR.values()))
                self.log(str(list(self.prev_SR.values())) + str(self.num_of_step))
                if all(np.array(list(self.prev_SR.values())) > 0.85):
                    if not self.amount_angle_change >= self.max_amount_angle_change:
                        self.amount_angle_change += 1
                    elif not self.amount_radius_change >= self.max_amount_radius_change:
                        self.amount_radius_change += 1
                        self.amount_angle_change = 0
                    self.log("training mode up to " + str(self.training_mode) + " step: " + str(self.num_of_step) + " radius " + str(self.traning_radius))
                    self.delay_change_RM = 0

    def _get_terminated(self, observation, RM):
        achievements = dict.fromkeys(self.reward_modes, False)
        if observation["current_dist_to_goal"] < 1:
            achievements["move"] = True
        if RM > 0 and achievements["move"] and abs(observation["orientation_error"]) < 15:
            achievements["rotation"] = True

        return achievements

    def get_reward(self, obs):
        achievements = self._get_terminated(obs, self.reward_mode)
        print(achievements)
        terminated = False
        truncated = False
        punish_time = self._get_punish_time()

        if not achievements[self.reward_modes[0]]:
            reward = -2/self._max_episode_length
        else:
            if not achievements[self.reward_modes[1]]:
                reward = -1/self._max_episode_length
            else:
                if self.reward_mode == 1:
                    terminated = True
                    reward = 3
                    return reward, terminated, truncated
                else:
                    print("error in get_reward function!")
                
        return reward, terminated, truncated
    
    def _get_punish_time(self):
        return 5*float(self._get_current_time())/float(self._max_episode_length)

    def move(self, action):
        raw_forward = action[0]
        raw_angular = action[1]

        forward = (raw_forward + 1.0) / 2.0
        forward_velocity = forward * self.max_velocity

        angular_velocity = raw_angular * self.max_angular_velocity

        for i in range(self._skip_frame):
            self.jetbot.apply_wheel_actions(
                self.jetbot_controller.forward(command=[forward_velocity, angular_velocity])
            )
            self._my_world.step(render=False)

        return

    def step(self, action):
        if not self._test:
            observations = self.get_observations()
            print("self.traning_radius",  self.traning_radius)
            print("self.traning_angle", self.traning_angle)
            print(str(list(self.prev_SR.values())))
            info = {}
            truncated = False
            terminated = False

            previous_jetbot_position, previous_jetbot_orientation = self.jetbot.get_world_pose()
            self.move(action)

            gt_observations = self.get_gt_observations(previous_jetbot_position, previous_jetbot_orientation)
            reward, terminated, truncated = self.get_reward(gt_observations)
            sources = ["time_out", "collision", "Nan"]
            source = "Nan"

            if not terminated:
                if self._is_timeout():
                    truncated = False
                    reward = reward - 4
                    source = sources[0]
                if self._is_collision() and self._get_current_time() > 2*self._skip_frame:
                    truncated = True
                    reward = reward - 5
                    source = sources[1]
            
            if terminated or truncated:
                self.get_success_rate(gt_observations, terminated, sources, source)
                self.start_step = True
                reward -= self._get_punish_time()

            return observations, reward, terminated, truncated, info
        else:
            return self.test(action)


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        from omni.isaac.core.utils.prims import create_prim, define_prim, delete_prim
        self._my_world.reset()
        #torch.cuda.empty_cache()
        info = {}
        self.event = np.random.choice(self.events)
        self.num_of_step = self.num_of_step + 1
        poses_bowl = [np.array([2.7843892574310303,-2.334873676300049,0.7]),
                      np.array([2.6932122707366943,-2.096846103668213,0.7]),
                      np.array([2.6258580684661865,-1.9797149896621704,0.7]),
                      np.array([2.2554433345794678,-1.4549099206924438,0.7]),
                      np.array([2.0932204723358154,-0.9349952340126038,0.7]),
                      np.array([1.7827470302581787,-0.5929904580116272,0.7]),
                      np.array([2.1453826427459717,1.099216103553772,0.7]),
                      np.array([2.213555097579956,1.3844422101974487,0.7]),
                      np.array([2.3652637004852295,1.6736856698989868,0.7]),
                      np.array([2.525071859359741,2.019548177719116,0.7]),
                      np.array([2.6450564861297607,2.3489251136779785,0.7]),]
        if self.event == 0:
            self.num_of_envs = np.random.choice([0,1,2,3,4,5])
        elif self.event == 1:
            self.num_of_envs = np.random.choice([6,8,9,10])
        # self.num_of_envs
        self.goal_position = poses_bowl[self.num_of_envs]
        if 0:
            self.goal_cube.set_world_pose(self.goal_position)
        else:
            delete_prim(f"/cup")
            create_prim(
                    prim_path=f"/cup",
                    position=self.goal_position,
                    usd_path=asdict(self.config).get('cup_usd_path', None),
                )

        self.traning_radius = self.amount_radius_change*self.max_traning_radius/self.max_amount_radius_change
        self.traning_angle = self.amount_angle_change*self.max_trining_angle/self.max_amount_angle_change
        print("self.traning_radius",  self.traning_radius)
        print("self.traning_angle", self.traning_angle)
        if 0:
            self.traning_angle = np.pi/8
            self.traning_radius = 0.4
        if self.num_of_step > 0:
            self.change_reward_mode()

        new_pos, new_angle = self.get_position(self.goal_position[0], self.goal_position[1])
        self.jetbot.set_world_pose(new_pos ,get_quaternion_from_euler(new_angle))
        observations = self.get_observations()
        return observations, info
    
    def get_position(self, x_goal, y_goal):
        k = 0
        self.change_line += 1
        reduce_r = 1
        reduce_phi = 1
        if self.change_line >= self.repeat:
            reduce_r = np.random.rand()
            reduce_phi = np.random.rand()
            self.change_line=0
        print("reduce", reduce_r)
        while 1:
            k += 1
            target_pos = np.array([x_goal, y_goal, 0.1])
            if self.event == 0:
                target_pos += np.array([0.85,0.65,0])
            elif self.event == 1:
                target_pos += np.array([0.85,-0.65,0])
            
            if self.traning_radius > 4:
                target_pos += np.array([2,-y_goal,0])

            alpha = np.random.rand()*2*np.pi
            target_pos += reduce_r*self.traning_radius*np.array([np.cos(alpha), np.sin(alpha), 0])

            goal_world_position = np.array([x_goal, y_goal])
            nx = np.array([-1,0])
            ny = np.array([0,1])
            to_goal_vec = goal_world_position - target_pos[0:2]
            
            cos_angle = np.dot(to_goal_vec, nx) / np.linalg.norm(to_goal_vec) / np.linalg.norm(nx)
            
            quadrant = self.get_quadrant(nx, ny, to_goal_vec)
            if target_pos[1]>=-2.4 and target_pos[1]<=2.4 and target_pos[0]>=1.8 and ((target_pos[1]<=-0.6 and target_pos[0]>-2*target_pos[1]-3) or (target_pos[1]>=0.6 and target_pos[0]>2*target_pos[1]-3) or target_pos[0]>=1.8):
                n = np.random.randint(2)
                return target_pos, quadrant*np.arccos(cos_angle) + ((-1)**n)*reduce_phi*self.traning_angle
            elif k >= 50:
                print("can't get correct robot position: ", target_pos, quadrant*np.arccos(cos_angle) + reduce_phi*self.traning_angle, reduce_r)
            

    def get_observations(self):
        self._my_world.render()
        jetbot_linear_velocity = self.jetbot.get_linear_velocity()
        jetbot_angular_velocity = self.jetbot.get_angular_velocity()
        images = self.pytorch_listener.get_rgb_data()
        if images is not None:
            from torchvision.utils import save_image, make_grid
            img = images/255
            # if self._get_current_time() < 20:
            #     save_image(make_grid(img, nrows = 2), '/home/kit/.local/share/ov/pkg/isaac-sim-2023.1.1/standalone_examples/base_aloha_env/Aloha/img/memory.png')
        else:
            print("Image tensor is NONE!")
        transform = T.ToPILImage()

        yimg = images[0].cpu().numpy().transpose(1, 2, 0) 
        yolo_classes = list(self.model.names.values())
        classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
        conf = 0.3
        results = self.model.predict(yimg, classes=45, conf=conf)
        if results[0].masks is not None:
            colors = 255#[random.choices(range(256), k=1) for _ in classes_ids]
            for result in results:
                for mask, box in zip(result.masks.xy, result.boxes):
                    points = np.int32([mask])
                    color_number = classes_ids.index(int(box.cls[0]))
                    cv2.fillPoly(yimg, points, [0,256,0])
            # cv2.imwrite("/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha/img/yolotest.png", yimg)
            img_current = self.clip_preprocess(transform(yimg)).unsqueeze(0).to(self.device)
        else:
            print("can't detect")
            # cv2.imwrite("/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha/img/yolotest.png", yimg)
            img_current = self.clip_preprocess(transform(img[0])).unsqueeze(0).to(self.device)
        #save_image(make_grid(yimg, nrows = 2), '/home/kit/.local/share/ov/pkg/isaac-sim-2023.1.1/standalone_examples/base_aloha_env/Aloha/img/yolotest.png')
        with torch.no_grad():
            img_current_emb = self.clip_model.encode_image(img_current)
        event = self.event,
        if event == 1:
            s = "target on right table"
        else:
            s = "target on left table"

        text = clip.tokenize([s]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text)
        graph_embedding = self.get_graph_embedding()
        # with torch.no_grad():
        #     graph_embedding = graph
        print("embedding ", type(graph_embedding))

        return np.concatenate(
            [
                jetbot_linear_velocity,
                jetbot_angular_velocity,
                self.img_goal_emb[0].cpu(),
                img_current_emb[0].cpu(),
                text_features[0].cpu(),
                # graph_embedding.cpu().detach().numpy(),
            ]
        )
    def prepare_input_data(self, objects, bbox_pose):
        object_ids = [0, 1, 2, 3]  # assume we have 4 objects in one env
        features = []

        for obj_id in object_ids:
            # extract descriptor (1, 384) -> (384,)
            descriptor = np.array(objects["objects"][obj_id]["descriptor"]).flatten()

            # extract bbox info
            bbox_extent = np.array(bbox_pose[obj_id]["bbox_extent"])  # (3,)
            bbox_center = np.array(bbox_pose[obj_id]["bbox_center"])  # (3,)

            # (384 + 3 + 3 = 390)
            object_feature = np.concatenate([descriptor, bbox_extent, bbox_center])  # (390,)
            features.append(object_feature)

        # to tensor
        return torch.tensor(features, dtype=torch.float32).to(self.device)  # Shape: (4, 390)

    def get_graph_embedding(self):
        self.stept += 1
        if self.stept % 5000 == 0:
            save_dir = "/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha"
            checkpoint_path = os.path.join(save_dir, f"scene_embedding_epoch_{self.stept}.pth")
            torch.save(self.embedding_net.state_dict(), checkpoint_path)
        scene_path = asdict(self.config).get('scene_file', None)
        scene_file = os.path.join(scene_path, f"scene_{self.num_of_envs}.pkl")  # 替换为实际路径前缀

        with open(scene_file, "rb") as f:
            objects = pickle.load(f)

        # input feature (7, 518)
        object_features = []
        for obj_id, data in objects.items():
            # get bbox_center and bbox_size
            bbox_center = data["bbox_center"]  # (3,)
            bbox_size = data["bbox_size"]      # (3,)
            clip_embd = data["clip_embd"]      # (512,)

            # -->518
            feature = np.concatenate([bbox_center, bbox_size, clip_embd])  # (3 + 3 + 512 = 518,)
            object_features.append(feature)

        # to tensor (7, 518)
        object_features = torch.tensor(object_features, dtype=torch.float32).to(self.device)  # (7, 518)

        # object_features = self.prepare_input_data(objects, bbox_pose)

        rl_loss = torch.load('/home/kit/Downloads/loss.pt')

        # forward
        predicted_scene_embedding = self.embedding_net(object_features)

        # define loss
        if rl_loss.ndim == 0:  
            rl_loss_value = rl_loss.to(self.device)
        else:  
            rl_loss_value = rl_loss[-1].to(self.device)

        # rl_loss_value = torch.tensor(rl_loss[-1], dtype=torch.float32).to(self.device)
        self.embedding_loss = torch.abs(torch.mean(predicted_scene_embedding) * rl_loss_value)

        # gradient update
        self.embedding_optimizer.zero_grad()
        self.embedding_loss.backward()
        self.embedding_optimizer.step()

        return predicted_scene_embedding #32

    def render(self, mode="human"):
        return

    def close(self):
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]
    
    def log(self, message):
        f = open(self.log_path, "a+")
        f.write(message + "\n")
        f.close()
        return

    def _on_contact_report_event(self, contact_headers, contact_data):
        from pxr import PhysicsSchemaTools

        for contact_header in contact_headers:
            # instigator
            act0_path = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0))
            # recipient
            act1_path = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1))
            # the specific collision mesh that belongs to the Rigid Body
            cur_collider = str(PhysicsSchemaTools.intToSdfPath(contact_header.collider0))

            # iterate over all contacts
            contact_data_offset = contact_header.contact_data_offset
            num_contact_data = contact_header.num_contact_data
            for index in range(contact_data_offset, contact_data_offset + num_contact_data, 1):
                cur_contact = contact_data[index]

                # find the magnitude of the impulse
                cur_impulse =  cur_contact.impulse[0] * cur_contact.impulse[0]
                cur_impulse += cur_contact.impulse[1] * cur_contact.impulse[1]
                cur_impulse += cur_contact.impulse[2] * cur_contact.impulse[2]
                cur_impulse = math.sqrt(cur_impulse)

            if num_contact_data > 1: #1 contact with flore here yet
                self.collision = True