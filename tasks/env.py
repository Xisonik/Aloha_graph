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
from scipy.optimize import minimize
import psutil
from ultralytics import YOLO
import cv2

from .control_manager import PID_controller, Control_module
from .scene_manager import Scene_controller
from .memory_manager import ImageMemoryManager
from .graph_manager import Graph_manager
from .localization import LocalizationModule
import random

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

import time
from embed_nn import SceneEmbeddingNetwork
import torch.optim as optim
from collections import deque
import os

# В начале файла добавим импорт
from .graph_data_collector import GraphDataCollector

class CLGRCENV(gym.Env):
    """
        key = numner of obstacles
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        config = MainConfig(),
        skip_frame=4,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=4096,
        seed=10,
        MAX_SR=100,
        test=False,
        reward_mode=0
    ) -> None:
        self.init = True
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
        from pxr import Semantics
        self.scene_controller = Scene_controller()
        self.controle_module = Control_module()
        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
        self._my_world.scene.add_default_ground_plane()
        # Для пола (в __init__)
        import omni.usd
        self.my_stage = omni.usd.get_context().get_stage()
        ground_prim = self.my_stage.GetPrimAtPath("/World/defaultGroundPlane")
        semantics_api = Semantics.SemanticsAPI.Apply(ground_prim, "Semantics")
        semantics_api.CreateSemanticTypeAttr().Set("class")
        semantics_api.CreateSemanticDataAttr().Set("floor")
        print(f"Semantic tag for /World/defaultGroundPlane: {semantics_api.GetSemanticDataAttr().Get()}")
        # ground_prim = self.my_stage.GetPrimAtPath("/World/defaultGroundPlane")
        # ground_prim.SetCustomDataByKey("semantic:tag", "floor")
        # Словарь для отслеживания созданных конфигураций
        self.scene_configs = {}  # {scene_id: config_dict}
        self.current_scene_id = None

        # Базовая директория для вывода
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_data")

        # Остальная инициализация (без робота пока)
        room_usd_path = asdict(config).get('room_usd_path', None)
        create_prim(prim_path="/room", translation=(0, 0, 0), usd_path=room_usd_path)
        jetbot_asset_path = asdict(config).get('jetbot_asset_path', None)
        self.jetbot = self._my_world.scene.add(
            WheeledRobot(
                prim_path="/jetbot",
                name="my_jetbot",
                wheel_dof_names=["left_wheel", "right_wheel"],
                create_robot=True,
                usd_path=jetbot_asset_path,
                position=np.array([-10, -10, 0.0]),
                orientation=get_quaternion_from_euler(-np.pi/2),
            )
        )
        # Отложим создание робота до reset
        # self.jetbot = None
        # self.jetbot_controller = None

        # Остальная часть __init__ остаётся без изменений
        from pxr import PhysicsSchemaTools, UsdUtils, PhysxSchema, UsdPhysics
        from pxr import Usd
        from omni.physx import get_physx_simulation_interface
        self.my_stage = omni.usd.get_context().get_stage()
        self.my_prim = self.my_stage.GetPrimAtPath("/jetbot")

        contactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(self.my_prim)
        contact_report_sub = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact_report_event)
        # self.set_env(config)
        self.jetbot_controller = DifferentialController(name="simple_control", wheel_radius=0.068, wheel_base=0.34)

        self.goal_position = np.array([0,0,0])
        self.render_products = []
        from omni.replicator.isaac.scripts.writers.pytorch_writer import PytorchWriter
        from omni.replicator.isaac.scripts.writers.pytorch_listener import PytorchListener
        import omni.replicator.core as rep
        self.image_resolution = 400
        self.camera_width = self.image_resolution
        self.camera_height = self.image_resolution

        camera_path_1 = "/jetbot/fl_link4/visuals/realsense/husky_rear_left"#asdict(config).get('camera_paths', None)
        camera_path_2 = "/jetbot/fr_link4/visuals/realsense/husky_rear_left"#asdict(config).get('camera_paths', None)
        render_product_rgb = rep.create.render_product(camera_path_1, resolution=(self.camera_width, self.camera_height))
        render_product_depth = rep.create.render_product(camera_path_1, resolution=(self.camera_width, self.camera_height))
        self.render_products = [render_product_rgb, render_product_depth]
        self.key = 0

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
        shape_size = 1542
        self.use_memory = asdict(config).get('memory', None)
        print("use memory", self.use_memory)
        if self.use_memory:
            self.memory = ImageMemoryManager()
        if self.use_memory:
            shape_size += 390

        self.use_graph = asdict(config).get('graph', None)
        if self.use_graph:
            self.graph_module = Graph_manager()
        if self.use_graph:
            shape_size += 390

        self.observation_space = spaces.Box(low=-1000000000, high=1000000000, shape=(2060,), dtype=np.float32) #

        self.max_velocity = 1.5
        self.max_angular_velocity = math.pi*0.4
        self.events = [1]#[0, 1, 2]
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

        # Load YOLO model once
        # self.model = YOLO("yolov8m-seg.pt")
        # self.model.to(self.device)
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
        self.log_path = asdict(config).get('log_path', None)

        self.training_mode = asdict(config).get('training_mode', None)
        self.local_training_mode = 0
        self.traning_radius = 0
        self.trining_delta_angle = 0
        self.max_traning_radius = 4
        self.max_trining_angle = np.pi/3
        self.amount_angle_change = 0
        self.amount_radius_change = 0
        self.max_amount_angle_change = 3
        self.max_amount_radius_change = 10
        self.num_of_envs = 0
        self.eval = asdict(config).get('eval', None)
        torch.save(torch.tensor([0]), asdict(config).get('loss_path', None))
        self.t = 0
        import omni.isaac.core.utils.prims as prim_utils
        self.learn_emb = 0
        self.current_jetbot_position = np.array([0,0])
        self.current_jetbot_orientation = 0

        self.evalp = asdict(config).get('eval_print', None)
        self.eval_log_path = asdict(config).get('eval_log_path', None)
        self.eval_r = 0.2
        self.eval_angle = 0
        self.eval_dt = 0.2
        self.eval_dangle = np.pi/18
        self.eval_step = 0
        self.eval_step_angle = 0
        self.eval_sr = []
        self.eval_write = 0
        self.SR_len = 10
        self.SR_t = 0
        self.demonstrate = False
        self.target_positions = []
        self.level = 0
        self.cache = None

        if self.use_graph:
            self.graph_embedding = self.graph_module.get_graph_embedding()
        self.imitation_part = 1
        self.init = False
        from omni.isaac.sensor import Camera
        from omni.isaac.core.utils.stage import get_current_stage
        from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats
        import omni.replicator.core as rep

        self.camera_resolution = (1280, 720)  # Match CLGRCENV's image_resolution
        self.camera_path = "/jetbot/fl_link4/visuals/realsense/husky_rear_left"
        self.camera = Camera(prim_path=self.camera_path)
        self.camera.initialize()
        self.camera.add_distance_to_camera_to_frame()  # Enable depth output

        self.camera.set_local_pose(
            np.array([0.2, 0, 0.1]),  # Relative to jetbot prim
            euler_angles_to_quats(np.array([-90, 0, 0]), degrees=True),
            camera_axes="world"
        )
        camera_prim = get_current_stage().GetPrimAtPath(self.camera_path)
        camera_prim.GetAttribute("horizontalAperture").Set(80)  # Match GraphDataCollector
        camera_prim.GetAttribute("focalLength").Set(35.0)  # Увеличиваем фокусное расстояние (мм)
        camera_prim.GetAttribute("horizontalAperture").Set(80)  # Уменьшаем горизонтальную апертуру (мм)
        camera_prim.GetAttribute("verticalAperture").Set(24.0)

        # Create a single render product
        self.render_product = rep.create.render_product(self.camera_path, resolution=self.camera_resolution)

        # Initialize annotators
        self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("LdrColor")  # Use LdrColor like GraphDataCollector
        self.depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")  # Use distance_to_image_plane
        self.rgb_annotator.attach([self.render_product])
        self.depth_annotator.attach([self.render_product])
        self.annotators = [self.rgb_annotator, self.depth_annotator]

        # Remove old render products and PytorchWriter (no longer needed)
        self.render_products = [self.render_product]  # For compatibility with existing code
        self.pytorch_listener = None
        self.pytorch_writer = None

        # Initialize localization module with the same YOLO model
        # self.localization_module = LocalizationModule(model=self.model)
        
        # Cache for scene graphs
        self.obstacles_prop = None
        self.scene_graphs_cache = {}
        self.max_number_of_obstacles = None
        self.SR = 0

        return

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
        self.SR = SR
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
        # print("time: ", self._get_current_time())
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
        self.current_jetbot_position = current_jetbot_position[0:2]
        self.current_jetbot_orientation = euler_from_quaternion(current_jetbot_orientation)[0]
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
                    self.SR_t += 1
                if self.SR_t >= self.SR_len: 
                    if not self.amount_angle_change >= self.max_amount_angle_change:
                        self.amount_angle_change += 1
                    elif not self.amount_radius_change >= self.max_amount_radius_change:
                        self.amount_radius_change += 1
                        self.amount_angle_change = 0
                    self.log("training mode up to " + str(self.training_mode) + " step: " + str(self.num_of_step) + " radius " + str(self.traning_radius) + "level " + str(self.level))
                    self.delay_change_RM = 0
                    self.SR_t = 0

    def _get_terminated(self, observation, RM):
        achievements = dict.fromkeys(self.reward_modes, False)
        if observation["current_dist_to_goal"] < 1.3:
            achievements["move"] = True
        if RM > 0 and achievements["move"] and abs(observation["orientation_error"]) < 15:
            achievements["rotation"] = True

        return achievements

    def get_reward(self, obs):
        achievements = self._get_terminated(obs, self.reward_mode)
        # print(achievements)
        terminated = False
        truncated = False
        punish_time = self._get_punish_time()

        if not achievements[self.reward_modes[0]]:
            reward = 2*punish_time
        else:
            if not achievements[self.reward_modes[1]]:
                reward = punish_time
            else:
                if self.reward_mode == 1:
                    terminated = True
                    reward = 5
                    return reward, terminated, truncated
                else:
                    print("error in get_reward function!")
                
        return reward, terminated, truncated
    
    def _get_punish_time(self):
        return -0.2#self._max_episode_length

    def move(self, action):
        # print("action type", type(action), action.shape, action)
        if not self.demonstrate:
            raw_forward = action[0]
            raw_angular = action[1]

            forward = (raw_forward + 1.0) / 2.0
            forward_velocity = forward * self.max_velocity

            angular_velocity = raw_angular * self.max_angular_velocity
        else:
            current_jetbot_position, current_jetbot_orientation = self.jetbot.get_world_pose()
            forward_velocity, angular_velocity = self.controle_module.pure_pursuit_controller(current_jetbot_position[0:2], self.current_jetbot_orientation,)
            # forward_velocity, angular_velocity = self.controle_module.get_PID_controle_velocity(current_jetbot_position[0:2], self.current_jetbot_orientation, self.goal_position)
            # print("Agent get controle: ", forward_velocity, angular_velocity)
        for i in range(self._skip_frame):
            self.jetbot.apply_wheel_actions(
                self.jetbot_controller.forward(command=[forward_velocity, angular_velocity])
            )
            self._my_world.step(render=False)
        
        return self.write_action_from_velocity(forward_velocity, angular_velocity)
    
    def write_action_from_velocity(self, forward_velocity, angular_velocity):
        action = None
        forward = forward_velocity / self.max_velocity
        raw_angular = angular_velocity / self.max_angular_velocity
        raw_forward = forward * 2.0 - 1.0
        action = np.array([[raw_forward, raw_angular]])
        #write action and bool in file
        # print("write action", action)
        return action
    
    def step(self, action):
        observations = self.get_observations()
        info = {}
        truncated = False
        terminated = False

        previous_jetbot_position, previous_jetbot_orientation = self.jetbot.get_world_pose()
        new_action = self.move(action)

        gt_observations = self.get_gt_observations(previous_jetbot_position, previous_jetbot_orientation)
        reward, terminated, truncated = self.get_reward(gt_observations)
        sources = ["time_out", "collision", "Nan"]
        source = "Nan"
        if not terminated:
            if self._is_timeout():
                truncated = True #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!здесь почему-то был false
                reward = -5
                source = sources[0]
            if self._is_collision() and self._get_current_time() > 2*self._skip_frame:
                truncated = True
                reward = -6
                source = sources[1]
        
        if terminated or truncated:
            if not self.demonstrate:
                self.get_success_rate(gt_observations, terminated, sources, source)
            self.start_step = True
            reward -= self._get_punish_time()
            if self.evalp and not self.demonstrate:
                s = 0
                if terminated:
                    s = 1
                self.eval_sr.append(s)
                if self.eval_write:
                    self.eval_write = 0
                    sr = sum(self.eval_sr) / len(self.eval_sr)
                    message = "r: " + str(round(self.traning_radius, 2)) + " a: " + str(round(self.traning_angle,2)) + " s: " + str(round(sr,2))
                    self.eval_sr = []
                    f = open(self.eval_log_path, "a+")
                    f.write(message + "\n")
                    f.close()

        message_to_collback = [self.demonstrate, new_action]
        # info["message_to_collback", message_to_collback]
        return observations, reward, terminated, truncated, info, message_to_collback
    
    def generate_random_key(self):
        """Генерирует случайный key в формате [k0, k1], где k0 и k1 — числа от 0 до 7."""
        k0 = 0
        k1 = 0
        r = r = asdict(self.config).get('eval_radius', None)
        test_level = False# asdict(self.config).get('test_level', None)
        level = self.level if not test_level else test_level
        print("current level is ", level)
        if self.max_number_of_obstacles is None:
            _, obstacles, _ = self.scene_controller.get_obstacles(-1)
            self.max_number_of_obstacles = len(obstacles)
        k = random.randint(5, self.max_number_of_obstacles)
        return k

    def set_env(self, config, key, obstacles_prop,  goal_position=np.array([1, 1, 0.0])):
        from omni.isaac.core.objects import VisualCuboid, FixedCuboid
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.core.utils.prims import create_prim, define_prim, delete_prim
        from omni.isaac.core import World
        from .wheeled_robot import WheeledRobot
        from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
        from omni.isaac.core.objects import VisualCuboid, FixedCuboid, VisualCylinder
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.core.utils.prims import create_prim, define_prim, delete_prim
        import omni.isaac.core.utils.prims as prim_utils
        import omni.usd
        from pxr import UsdGeom, Gf, Semantics
        import random
        import json
        import os

        # Базовый путь к USD-файлам
        base_path = "/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/assets/scenes/obstacles"
        # Генерация конфигурации сцены
        self.key = key
        self.obstacles_prop = obstacles_prop
        scene_config = {
            "goal_position": goal_position.tolist(),
            "obstacles": obstacles_prop
        }

        # Сбрасываем позицию робота
        self.jetbot.set_world_pose([-10, -10, 0], get_quaternion_from_euler(0))
        self._my_world.reset()

        # Удаляем старые объекты
        stage = omni.usd.get_context().get_stage()
        # Удаляем старые препятствия
        i = 0
        while True:
            prim_path = f"/obstacle_{i}"
            if stage.GetPrimAtPath(prim_path).IsValid():
                delete_prim(prim_path)
                print(f"Deleted old obstacle: {prim_path}")
            else:
                break
            i += 1
        # Удаляем старую цель
        delete_prim("/cup")
        print("Deleted old cup")

        # Добавляем освещение только при инициализации
        if self.init:
            light_1 = prim_utils.create_prim(
                "/World/Light_1", "SphereLight", position=np.array([2.5, 5.0, 20.0]),
                attributes={"inputs:radius": 0.1, "inputs:intensity": 5e7, "inputs:color": (1.0, 1.0, 1.0)}
            )
            print("Added light at initialization")

        # Вызываем функцию для генерации JSON
        # self.generate_obstacles_json(obstacles_prop, goal_position, self.key)

        # Создаём новые препятствия
        for i, obstacle_prop in enumerate(obstacles_prop):
            prim_path = f"/obstacle_{i}"
            if obstacle_prop["shape"] == "table":
                usd_path = f"{base_path}/table.usd"
                prim = stage.DefinePrim(prim_path, "Xform")
                prim.GetReferences().AddReference(usd_path)

                xform = UsdGeom.Xformable(prim)
                xform.ClearXformOpOrder()
                translate_op = xform.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(obstacle_prop["position"]))
                rotate_op = xform.AddRotateXYZOp()
                rotate_op.Set(Gf.Vec3d(0, 0, 90))

                semantics_api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
                semantics_api.CreateSemanticTypeAttr().Set("class")
                semantics_api.CreateSemanticDataAttr().Set("table")
                print(f"Semantic tag for {prim_path}: {semantics_api.GetSemanticDataAttr().Get()}")

            elif obstacle_prop["shape"] == "chair":
                usd_path = random.choice([f"{base_path}/chair1.usd", f"{base_path}/chair2.usd"])
                prim = stage.DefinePrim(prim_path, "Xform")
                prim.GetReferences().AddReference(usd_path)

                xform = UsdGeom.Xformable(prim)
                xform.ClearXformOpOrder()
                translate_op = xform.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(obstacle_prop["position"]))
                rotate_op = xform.AddRotateXYZOp()
                rotate_op.Set(Gf.Vec3d(0, 0, 0))

                semantics_api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
                semantics_api.CreateSemanticTypeAttr().Set("class")
                semantics_api.CreateSemanticDataAttr().Set("chair")
                print(f"Semantic tag for {prim_path}: {semantics_api.GetSemanticDataAttr().Get()}")

            elif obstacle_prop["shape"] == "trashcan":
                usd_path = f"{base_path}/TrashCan.usd"
                prim = stage.DefinePrim(prim_path, "Xform")
                prim.GetReferences().AddReference(usd_path)

                xform = UsdGeom.Xformable(prim)
                xform.ClearXformOpOrder()
                translate_op = xform.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(obstacle_prop["position"]))
                rotate_op = xform.AddRotateXYZOp()
                rotate_op.Set(Gf.Vec3d(0, 0, 0))

                semantics_api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
                semantics_api.CreateSemanticTypeAttr().Set("class")
                semantics_api.CreateSemanticDataAttr().Set("trashcan")
                print(f"Semantic tag for {prim_path}: {semantics_api.GetSemanticDataAttr().Get()}")

            elif obstacle_prop["shape"] == "vase":
                usd_path = f"{base_path}/vase.usd"
                prim = stage.DefinePrim(prim_path, "Xform")
                prim.GetReferences().AddReference(usd_path)

                xform = UsdGeom.Xformable(prim)
                xform.ClearXformOpOrder()
                translate_op = xform.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(obstacle_prop["position"]))
                rotate_op = xform.AddRotateXYZOp()
                rotate_op.Set(Gf.Vec3d(0, 0, 0))

                semantics_api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
                semantics_api.CreateSemanticTypeAttr().Set("class")
                semantics_api.CreateSemanticDataAttr().Set("vase")
                print(f"Semantic tag for {prim_path}: {semantics_api.GetSemanticDataAttr().Get()}")

        # Добавляем цель (чашу)
        create_prim(prim_path="/cup", position=goal_position, usd_path=asdict(config).get('cup_usd_path', None))
        cup_prim = stage.GetPrimAtPath("/cup")
        semantics_api = Semantics.SemanticsAPI.Apply(cup_prim, "Semantics")
        semantics_api.CreateSemanticTypeAttr().Set("class")
        semantics_api.CreateSemanticDataAttr().Set("bowl")
        print(f"Semantic tag for /cup: {semantics_api.GetSemanticDataAttr().Get()}")
        if self.use_graph:
            # Convert key list to tuple for dictionary key
            if len(self.scene_graphs_cache) is 0:
                import json
                import os
                import re

                # Путь к папке с JSON-файлами
                folder_path = '/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/gjst'

                # Регулярное выражение для извлечения ключей из имени файла
                pattern = re.compile(r'obstacles_and_goal_(\d+)_(\d+)\.json')

                # Кэш для хранения данных: {(goal_key, obstacle_key): data}
                cache = {}

                for filename in os.listdir(folder_path):
                    match = pattern.match(filename)
                    if match:
                        goal_key = int(match.group(1))
                        obstacle_key = str(match.group(2))
                        filepath = os.path.join(folder_path, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            self.scene_graphs_cache[(goal_key, obstacle_key)] = data
            self.graph_embedding = self.graph_module.get_graph_embedding()
        # Запускаем сбор данных для графа (если нужно)
        if False:
            scene_id = self.get_scene_id(scene_config)
            collector = GraphDataCollector(self._my_world, scene_config, self.output_dir, scene_name=f"scene_{scene_id}")
            collector.collect_data()
            
    def get_scene_id(self, scene_config):
        import json
        # Генерация уникального ID для конфигурации сцены
        import hashlib
        config_str = json.dumps(scene_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]  # Короткий хэш для читаемости
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if self.num_of_step > 0:
            self.change_reward_mode()

        from pxr import UsdShade
        import omni

        stage = omni.usd.get_context().get_stage()
        material_path = "/room/Looks/OmniPBR"  # Замените на актуальный путь
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        shader = UsdShade.Shader(material.GetPrim().GetChild("Shader"))
        # Основной цвет (albedo)
        shader.GetInput("diffuse_color_constant").Set((0.0, 0.0, 1.0))  # Красный[1][6]

        self._my_world.reset()
        self.event = np.random.choice(self.events)
        self.num_of_step += 1

        self.stept = self.stept + 1
        
        tuning = asdict(self.config).get('tuning', None)
        r = asdict(self.config).get('eval_radius', None)
        if self.traning_radius > 2.5 + self.level*2:
            self.level += 1
        correct_position = False
        second_try = False
        reset_obstacles = False
        while not correct_position:
            if self.stept % 10 == 0 or self.stept==0 or second_try:
                key = self.generate_random_key()
                self.key = key
                obstacles_prop = self.scene_controller.generate_obstacles(key=self.key)
                self.obstacles_prop = obstacles_prop
                self.goal_position, self.target_positions, self.num_of_envs = self.scene_controller.get_target_position()
                self.stept = 1
                reset_obstacles = True
            print("radius is ", self.traning_radius)
            eval = 1 if self.eval else 0
            add_r = 1 if tuning else 0
            if eval == add_r and eval == 1:
                print("u use eval and tuning")
            random_angle = 0
            self.traning_radius = add_r * r + r * eval + self.amount_radius_change * self.max_traning_radius / self.max_amount_radius_change
            self.traning_angle = eval * random_angle + self.amount_angle_change * self.max_trining_angle / self.max_amount_angle_change
            new_pos, new_angle, correct_position = self.scene_controller.get_robot_position(self.goal_position[0], self.goal_position[1], self.traning_radius, self.traning_angle)
            second_try = True
            if reset_obstacles and correct_position:
                self.set_env(config=self.config, obstacles_prop=obstacles_prop, key=key, goal_position=self.goal_position)

        self.jetbot.set_world_pose(new_pos, get_quaternion_from_euler(new_angle))

        get_prob_true = lambda x: np.random.rand() <= x
        use_control = 1 if asdict(self.config).get('control', None) else 0
        self.imitation_part = 1 if asdict(self.config).get('imitation', None) else 0
        if (self.eval or self.evalp) and not self.imitation_part:
            self.demonstrate = get_prob_true(-1)
        else:
            self.demonstrate = use_control * get_prob_true(min(max(1-self.SR, 0.6), 0.1) + self.imitation_part)
        print("demonstrate: ", self.demonstrate)
        if self.demonstrate:
            self.controle_module.update(new_pos, self.goal_position, self.target_positions, scene_controller=self.scene_controller, key=self.key)

        observations = self.get_observations()
        return observations, {}

    def tensor_to_pil(self, image_tensor):
        # Убедимся, что тензор находится в диапазоне [0, 255]
        image_tensor = image_tensor.clamp(0, 255)
        
        # Преобразуем (C, H, W) → (H, W, C) и приводим к типу uint8
        image_np = image_tensor.permute(1, 2, 0).byte().cpu().numpy()
        
        # Создаем изображение
        return Image.fromarray(image_np, mode="RGB")
    
    def list_to_graph(self, objects_list):
        """Convert a list of objects into a NetworkX graph."""
        import networkx as nx
        G = nx.Graph()
        
        # Add nodes with their properties
        for obj in objects_list:
            node_id = obj['id']
            # Extract position from bbox_center (using x, y, and z coordinates)
            pos = [obj['bbox_center'][0], obj['bbox_center'][1], obj['bbox_center'][2]]
            
            # Get object name, handling both string and list formats
            if isinstance(obj['name'], list):
                name = obj['name'][0]  # Take first name if it's a list
            else:
                name = obj['name']
                
            # Store all data as node attributes
            G.add_node(node_id, 
                      pos=pos,  # Add position attribute with x, y, z coordinates
                      bbox_center=obj['bbox_center'],
                      bbox_extent=obj['bbox_extent'],
                      name=name,
                      description=obj['description'])
            
        # Add edges between all nodes (you might want to modify this based on your needs)
        for i in range(len(objects_list)):
            for j in range(i + 1, len(objects_list)):
                G.add_edge(objects_list[i]['id'], objects_list[j]['id'])
                
        return G

    def get_observations(self):
        self._my_world.render()
        jetbot_linear_velocity = self.jetbot.get_linear_velocity()
        jetbot_angular_velocity = self.jetbot.get_angular_velocity()
        
        # Get RGB and depth data from annotators
        rgba_data = self.rgb_annotator.get_data()  # LdrColor returns RGBA
        depth_data = self.depth_annotator.get_data()  # distance_to_image_plane
    
        # # Debug prints for raw data
        # print("\n=== Raw Data Debug ===")
        # print("RGBA data type:", type(rgba_data), "shape:", rgba_data.shape if hasattr(rgba_data, 'shape') else "no shape")
        # print("Depth data type:", type(depth_data), "shape:", depth_data.shape if hasattr(depth_data, 'shape') else "no shape")
        # print("Depth data min:", np.min(depth_data) if depth_data is not None else "N/A", "max:", np.max(depth_data) if depth_data is not None else "N/A")
        
        # Save raw images for debugging
        # import os
        # debug_dir = "debug_images"
        # os.makedirs(debug_dir, exist_ok=True)
        
        # # Save raw RGBA image
        # if rgba_data is not None:
        #     import cv2
        #     cv2.imwrite(, rgba_data)
        #     print(f"Saved raw RGBA image to {debug_dir}/raw_rgba_{self.num_of_step}.png")
        min_val, max_val = 0.01, 10.0
        if rgba_data.size != 0:
            # Сохранение глубины
            clipped_depth = np.clip(rgba_data, min_val, max_val)
            normalized_depth = ((clipped_depth - min_val) / (max_val - min_val)) * 65535
            depth_image_uint16 = normalized_depth.astype("uint16")
            # cv2.imwrite(os.path.join(debug_dir, f"raw_rgba_{self.num_of_step}.png"), depth_image_uint16)
            # # print(f"Saved Depth data to {rgba_data}")
        
        # Save raw depth image
        if depth_data is not None:
            depth_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
            # cv2.imwrite(os.path.join(debug_dir, f"raw_depth_{self.num_of_step}.png"), depth_normalized)
            # print(f"Saved raw depth image to {debug_dir}/raw_depth_{self.num_of_step}.png")
    
        # Process RGBA to RGB
        rgba_data = np.asarray(rgba_data)  # Ensure NumPy array
        rgb_data = rgba_data[:, :, :3]  # Extract RGB channels (shape: [height, width, 3])
        rgb_img = rgb_data.astype(np.float32) / 255.0  # Normalize to [0, 1]
        rgb_img = torch.from_numpy(rgb_img).to(self.device)  # Shape: [height, width, 3]
        rgb_img = rgb_img.permute(2, 0, 1)  # Convert to [C, H, W]
        
        # Debug prints for processed RGB
        # print("\n=== Processed RGB Debug ===")
        # print("RGB data shape:", rgb_data.shape)
        # print("RGB img tensor shape:", rgb_img.shape)
        # print("RGB img tensor device:", rgb_img.device)
        # print("RGB img tensor dtype:", rgb_img.dtype)
        # print("RGB img value range:", rgb_img.min().item(), "to", rgb_img.max().item())
                
        # if rgb_data is None or depth_data is None:
        #     print("Warning: RGB or depth data is None. Returning default observation.")    
        # Process depth
        depth_img = np.asarray(depth_data)  # Ensure NumPy array
        depth_img = torch.from_numpy(depth_img).to(self.device)  # Convert to PyTorch tensor (shape: [height, width])
        
        # # Debug prints for processed depth
        # print("\n=== Processed Depth Debug ===")
        # print("Depth data shape:", depth_data.shape)
        # print("Depth img tensor shape:", depth_img.shape)
        # print("Depth img tensor device:", depth_img.device)
        # print("Depth img tensor dtype:", depth_img.dtype)
        # print("Depth img value range:", depth_img.min().item(), "to", depth_img.max().item())
        
        # Save processed images
        # if rgb_data is not None:
        #     cv2.imwrite(os.path.join(debug_dir, f"processed_rgb_{self.num_of_step}.png"), rgb_data)
        #     print(f"Saved processed RGB image to {debug_dir}/processed_rgb_{self.num_of_step}.png")
        
        # if depth_data is not None:
        #     depth_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
        #     cv2.imwrite(os.path.join(debug_dir, f"processed_depth_{self.num_of_step}.png"), depth_normalized)
        #     print(f"Saved processed depth image to {debug_dir}/processed_depth_{self.num_of_step}.png")
        
        transform = T.ToPILImage()
        
        # Process RGB for CLIP (handle single camera view for now)
        img_current = self.clip_preprocess(transform(rgb_img)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            img_current_emb = self.clip_model.encode_image(img_current)
        
        event = self.event
        if event == 1:
            s = "go to the bowl wall with 1 color"
        else:
            s = "go to the bowl wall with 2 color"

        # Process text on GPU
        text = clip.tokenize([s]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text)
        
        # Get graph embedding with localization (if enabled)
        if self.use_graph:
                # Get obstacles properties and generate scene graph
            #     obstacles_prop = self.obstacles_prop
            #     objects_list = self.generate_obstacles_json(obstacles_prop, self.goal_position, self.key)
            #     # Convert list to NetworkX graph
            #     self.scene_graphs_cache[self.key] = self.list_to_graph(objects_list)
            # scene_graph = self.scene_graphs_cache[self.key]
            
            # Prepare images for localization
            # Convert RGB to [H, W, 3] format and ensure values are in [0, 255]
            rgb_for_localize = (rgb_img * 255).clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            depth_for_localize = depth_img.cpu().numpy()
            
            # # Debug prints for localization input
            # print("\n=== Localization Input Debug ===")
            # print("RGB for localize shape:", rgb_for_localize.shape)
            # print("RGB for localize dtype:", rgb_for_localize.dtype)
            # print("RGB for localize value range:", rgb_for_localize.min(), "to", rgb_for_localize.max())
            # print("Depth for localize shape:", depth_for_localize.shape)
            # print("Depth for localize dtype:", depth_for_localize.dtype)
            # print("Depth for localize value range:", depth_for_localize.min(), "to", depth_for_localize.max())
            
            # # Save localization input images
            # cv2.imwrite(os.path.join(debug_dir, f"localize_rgb_{self.num_of_step}.png"), rgb_for_localize)
            # depth_normalized = cv2.normalize(depth_for_localize, None, 0, 255, cv2.NORM_MINMAX)
            # cv2.imwrite(os.path.join(debug_dir, f"localize_depth_{self.num_of_step}.png"), depth_for_localize)
            # print(f"Saved localization input images to {debug_dir}/localize_*_{self.num_of_step}.png")
            
            # Get robot position using localization
            current_jetbot_position, current_jetbot_orientation = self.jetbot.get_world_pose()
            # robot_position = self.localization_module.localize(
            #     rgb_for_localize,
            #     depth_for_localize,
            #     scene_graph,
            #     current_jetbot_position,
            #     current_jetbot_orientation
            # )
            
            # print("\n=== Localization Result ===")
            # print("Robot position:", robot_position)
            
            # # Transform scene graph to robot's local coordinate system
            # transformed_graph = self.localization_module.transform_graph_to_local(
            #     scene_graph,
            #     robot_position
            # )
            
            # Get embedding using the key
            # graph_embedding = self.get_graph_embedding()
            
            # Ensure all tensors are on GPU and have correct dimensions
            jetbot_linear_velocity_tensor = torch.tensor(jetbot_linear_velocity, device=self.device).flatten()
            jetbot_angular_velocity_tensor = torch.tensor(jetbot_angular_velocity, device=self.device).flatten()
            
            # Debug prints for final tensors
            # print("\n=== Final Tensors Debug ===")
            # print("Linear velocity tensor shape:", jetbot_linear_velocity_tensor.shape)
            # print("Angular velocity tensor shape:", jetbot_angular_velocity_tensor.shape)
            # print("Image embedding shape:", img_current_emb[0].shape)
            # print("Text features shape:", text_features[0].shape)
            # print("Graph embedding shape:", graph_embedding[0].shape)
            
            # Concatenate all features on GPU first
            features = torch.cat([
                jetbot_linear_velocity_tensor,
                jetbot_angular_velocity_tensor,
                img_current_emb[0],
                img_current_emb[0],
                text_features[0],
                self.graph_embedding.to("cuda"),
            ])
            # Move to CPU only once at the end
            return features.cpu().numpy()
                
        # Default case
        jetbot_linear_velocity_tensor = torch.tensor(jetbot_linear_velocity, device=self.device).flatten()
        jetbot_angular_velocity_tensor = torch.tensor(jetbot_angular_velocity, device=self.device).flatten()
        
        features = torch.cat([
            jetbot_linear_velocity_tensor,
            jetbot_angular_velocity_tensor,
            img_current_emb[0],
            img_current_emb[0],
            text_features[0],
        ])
        return features.cpu().numpy()

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
            if num_contact_data > 1: # or not self.scene_controller.no_intersect_with_obstacles(self.current_jetbot_position): #1 contact with flore here yet
                self.collision = True

    def get_env_info(self):  # <-- Этот метод нужен rl_games!
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "agents": 1  # Количество агентов
        }
    
    def get_graph_embedding(self):
        """
        Возвращает эмбеддинг сцены для заданного key, используя кэш последнего эмбеддинга.
        
        Args:
            key (list): Ключ в формате [k0, k1], например [3, 1].
        
        Returns:
            torch.Tensor: Тензор формы (max_objects, 390) с эмбеддингом сцены.
        """
        _, _, obstacles_id = self.scene_controller.get_obstacles()
        # _, _, obstacles = self.scene_controller.get_obstacles()
        _, _ , goal_position = self.scene_controller.get_target_position(not_change=True)
        key = ''.join(str(x) for x in sorted(obstacles_id))
        gr = self.scene_graphs_cache[(goal_position, key)]

        # Проверяем, есть ли кэш и совпадает ли текущий key с последним использованным
        if hasattr(self, 'cache') and self.cache is not None:
            cached_gr, cached_key, cached_embedding = self.cache
            if cached_key == key and cached_gr == gr:
                return cached_embedding.to(self.device)  # Убедимся, что тензор на GPU

        scene_embeddings = get_simple_embedding_from_json(gr, max_objects=5, device=self.device)
        self.cache = (gr, key, scene_embeddings)
        print(f"Generated and cached embedding for key {gr}_{key}")
        return scene_embeddings.to(self.device)  # Убедимся, что тензор на GPU


        # Формируем путь к JSON-файлу на основе key
        key_str = f"{key}"  # Например, "3_1"
        json_file = os.path.join("/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/gjst/", f"obstacles_and_goal_{key_str}.json")

        # Проверяем, существует ли JSON-файл
        if not os.path.exists(json_file):
            print(f"JSON file for key {key} not found at {json_file}!")
            return torch.zeros((8, 390), dtype=torch.float32, device=self.device)

        # Генерируем эмбеддинг с помощью get_simple_embedding_from_json
        try:
            scene_embeddings = get_simple_embedding_from_json(
                json_file, max_objects=8, device=self.device
            )
            # Сохраняем в кэш
            self.cache = (key, scene_embeddings)
            print(f"Generated and cached embedding for key {key}")
            return scene_embeddings.to(self.device)  # Убедимся, что тензор на GPU
        except Exception as e:
            print(f"Error generating embedding for key {key}: {e}")
            return torch.zeros((8, 390), dtype=torch.float32, device=self.device)
    



    #part with graph, should to move it

import json
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

def get_simple_embedding_from_json(data, model_name='all-MiniLM-L6-v2', max_objects=8, device=None):
    """
    Генерирует эмбеддинги для всех объектов из JSON-файла с фиксированной размерностью.
    
    Args:
        json_file (str): Путь к JSON-файлу с описанием сцены.
        model_name (str): Название модели sentence-transformers для генерации текстовых эмбеддингов.
        max_objects (int): Максимальное количество объектов, которое будет включено в итоговый тензор.
        device (str or torch.device): Устройство для вычислений (например, 'cuda:0' или 'cpu').
    
    Returns:
        torch.Tensor: Тензор формы (max_objects, 390), где каждый объект представлен отдельно.
                     Если объектов меньше max_objects, оставшиеся места заполняются нулями.
                     Если объектов больше max_objects, лишние обрезаются.
    """
    # Определяем устройство
    print("make graph embedding")
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Загружаем модель sentence-transformers
    model = SentenceTransformer(model_name, device=device)

    # # Читаем JSON-файл
    # with open(json_file, "r") as f:
    #     data = json.load(f)

    object_features = []

    # Обрабатываем каждый объект в JSON
    for obj_data in data:
        # Извлекаем числовые характеристики и перемещаем на нужное устройство
        bbox_extent = torch.tensor(obj_data["bbox_extent"], dtype=torch.float32, device=device)  # 3 dimensions
        bbox_center = torch.tensor(obj_data["bbox_center"], dtype=torch.float32, device=device)  # 3 dimensions

        # Извлекаем текстовое описание и генерируем эмбеддинг
        description = obj_data["description"]
        text_embedding = model.encode(description, convert_to_tensor=True, device=device)  # 384 dimensions (для all-MiniLM-L6-v2)

        # Объединяем числовые характеристики и текстовый эмбеддинг
        features = torch.cat([bbox_extent, bbox_center, text_embedding], dim=0)  # 3 + 3 + 384 = 390 dimensions
        object_features.append(features)

    # Если объектов нет, возвращаем тензор с нулями
    if not object_features:
        return torch.zeros((max_objects, 390), dtype=torch.float32, device=device)

    # Преобразуем список в тензор
    object_tensor = torch.stack(object_features)  # Shape: (num_objects, 390)

    # Фиксируем размерность
    num_objects = object_tensor.shape[0]
    feature_dim = object_tensor.shape[1]  # 390

    # Создаем итоговый тензор фиксированной формы (max_objects, 390)
    fixed_tensor = torch.zeros((max_objects, feature_dim), dtype=torch.float32, device=device)

    if num_objects <= max_objects:
        # Если объектов меньше или равно max_objects, заполняем тензор и оставляем нули в конце
        fixed_tensor[:num_objects, :] = object_tensor
    else:
        # Если объектов больше max_objects, обрезаем лишние
        fixed_tensor = object_tensor[:max_objects, :]

    return fixed_tensor

# # Пример использования
# json_file = "/path/to/your/scene_graph_2_26/1.json"  # Укажите путь к вашему JSON-файлу
# scene_embeddings = get_simple_embedding_from_json(json_file, max_objects=10)
# print("Scene embeddings shape:", scene_embeddings.shape)
# print("Scene embeddings:", scene_embeddings)

class LocalizationModule:
    def __init__(self, model=None, save_debug=True):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model  # Use the passed model instead of loading a new one
        self.save_debug = save_debug
        self.grid_size = 18
        self.inner_grid_size = 16
        self.objects = []
        self.unique_id_map = {}
        self.distance_ratios = []
        self.depths = []
        self.local_graph = []
        self.seq_iobj = []
        self.closest_object = None
        self.closest_detected_set = None
        self.robot_position = None
        self.robot_orientation = None
        # Store previous optimization result
        self.prev_optimization_result = None
        # Store previous robot position
        self.prev_robot_position = np.array([0.0, 0.0, 0.0])

    def detect_objects(self, rgb_image, depth_image):
        """Detect objects and compute depths using RGB and depth images."""
        results = self.model(rgb_image)
        detections = results[0].boxes
        centers = []
        names = []
        masks = []
        self.depths = []
        
        if self.save_debug:
            print("\n=== YOLO Detection Results ===")
            print("Number of detections:", len(detections))
            print("\nDetected objects:")
            for box in detections:
                name = self.model.names[int(box.cls)]
                confidence = float(box.conf.cpu().numpy())  # Convert tensor to float
                print(f"- {name} (confidence: {confidence:.2f})")

        for i, box in enumerate(detections):
            name = self.model.names[int(box.cls)]
            xyxy = box.xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = map(int, xyxy)
            center_x = (x1 + x2) / 2
            centers.append(center_x)
            names.append(name)
            
            if hasattr(results[0], 'masks') and results[0].masks is not None:
                mask = results[0].masks[i].data.cpu().numpy().squeeze()
                mask = cv2.resize(mask, (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = mask.astype(bool)
                depth_values = depth_image[mask]
                if depth_values.size > 0:
                    depth_min = depth_values.min()
                    self.depths.append(depth_min)
                else:
                    self.depths.append(0)
                    if self.save_debug:
                        print(f"No depth values for object {name} (index {i})")
            else:
                self.depths.append(0)
                if self.save_debug:
                    print(f"No mask available for object {name} (index {i})")

        scene_types = {obj['name'] for obj in self.objects}
        filtered_indices = [i for i, name in enumerate(names) if name in scene_types]
        if not filtered_indices:
            if self.save_debug:
                print("\nNo detected objects match the scene graph")
                print("Available scene objects:", scene_types)
                print("Detected objects:", set(names))
            return False

        filtered_names = [names[i] for i in filtered_indices]
        filtered_centers = [centers[i] for i in filtered_indices]
        self.depths = [self.depths[i] for i in filtered_indices]
        
        if self.save_debug:
            print("\nFiltered objects (matching scene graph):")
            for name, depth in zip(filtered_names, self.depths):
                print(f"- {name} (depth: {depth:.2f})")
        
        sorted_indices = np.argsort(filtered_centers)
        self.seq_iobj = [filtered_names[i] for i in sorted_indices]
        sorted_centers = [filtered_centers[i] for i in sorted_indices]
        self.depths = [self.depths[i] for i in sorted_indices]

        self.distance_ratios = []
        for i in range(len(sorted_centers) - 1):
            distance = abs(sorted_centers[i + 1] - sorted_centers[i])
            self.distance_ratios.append(distance)
        if self.distance_ratios and sum(self.distance_ratios) > 0:
            total_distance = sum(self.distance_ratios)
            self.distance_ratios = [d / total_distance for d in self.distance_ratios]
        else:
            self.distance_ratios = [1.0] * (len(sorted_centers) - 1)

        if self.save_debug:
            print("\nFinal sequence of objects (left to right):")
            for name in self.seq_iobj:
                print(f"- {name}")

        return len(self.seq_iobj) >= 2

    def build_local_graph(self, fov=60, image_width=640):
        """Build local graph from detected objects using pixel positions and depths."""
        self.local_graph = []
        fov_rad = np.deg2rad(fov)
        focal_length = (image_width / 2) / np.tan(fov_rad / 2)

        for i, (name, depth) in enumerate(zip(self.seq_iobj, self.depths)):
            if depth == 0:
                if self.save_debug:
                    print(f"Invalid depth for object {name}, skipping")
                continue
            pixel_x = (i + 0.5) * (image_width / len(self.seq_iobj)) - (image_width / 2)
            angle = np.arctan2(pixel_x, focal_length)
            x = depth * np.sin(angle)
            y = depth * np.cos(angle)
            import uuid
            obj_id = str(uuid.uuid4())
            self.local_graph.append({
                'id': obj_id,
                'name': name,
                'bbox_center': [float(x), float(y), 0.0],
                'description': 'obstacle'
            })
        return len(self.local_graph) >= 2

    def align_graphs(self):
        """Align local graph with scene graph using ICP."""
        detected_names = set(obj['name'] for obj in self.local_graph)
        graph_objects = [obj for obj in self.objects if obj['name'] in detected_names]
        if len(graph_objects) < len(self.local_graph):
            if self.save_debug:
                print("Not enough matching objects in scene graph")
            return None

        local_points = np.array([obj['bbox_center'][:2] for obj in self.local_graph])
        graph_points = np.array([obj['bbox_center'][:2] for obj in graph_objects])
        local_names = [obj['name'] for obj in self.local_graph]
        graph_names = [obj['name'] for obj in graph_objects]
        graph_ids = [str(obj['id']) for obj in graph_objects]

        def compute_error(params, src, dst, src_names, dst_names):
            tx, ty, theta = params
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]])
            transformed = (R @ src.T).T + np.array([tx, ty])
            error = 0
            for i, (p1, n1) in enumerate(zip(transformed, src_names)):
                min_dist = float('inf')
                for p2, n2 in zip(dst, dst_names):
                    if n1 == n2:
                        dist = np.linalg.norm(p1 - p2)
                        min_dist = min(min_dist, dist)
                error += min_dist
            return error

        # Use previous result as initial guess if available
        if self.prev_optimization_result is not None:
            initial_guess = self.prev_optimization_result
            if self.save_debug:
                print("Using previous optimization result as initial guess:", initial_guess)
        else:
            # First time initialization
            tx_init = np.random.uniform(0, 8)
            ty_init = np.random.uniform(0, 6)
            theta_init = np.random.uniform(-np.pi, np.pi)
            initial_guess = [tx_init, ty_init, theta_init]
            if self.save_debug:
                print("Using random initial guess:", initial_guess)

        result = minimize(
            compute_error,
            initial_guess,
            args=(local_points, graph_points, local_names, graph_names),
            method='Powell'
        )

        if not result.success:
            if self.save_debug:
                print("Graph alignment failed")
            return None

        # Store successful result for next time
        self.prev_optimization_result = result.x

        tx, ty, theta = result.x
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        matched_set = []
        transformed_points = (R @ local_points.T).T + np.array([tx, ty])

        for i, (p, n) in enumerate(zip(transformed_points, local_names)):
            min_dist = float('inf')
            best_id = None
            for j, (p2, n2, obj_id) in enumerate(zip(graph_points, graph_names, graph_ids)):
                if n == n2:
                    dist = np.linalg.norm(p - p2)
                    if dist < min_dist:
                        min_dist = dist
                        best_id = obj_id
            if best_id:
                matched_set.append(best_id)

        if len(matched_set) < len(self.local_graph):
            if self.save_debug:
                print("Incomplete graph matching")
            return None

        robot_pos_local = np.array([0, 0])
        robot_pos_global = (R @ robot_pos_local) + np.array([tx, ty])
        self.robot_position = robot_pos_global
        self.closest_detected_set = matched_set
        if self.save_debug:
            print(f"Aligned graph, robot position: {self.robot_position}, matched set: {matched_set}")
        return matched_set

    def localize(self, rgb_image, depth_image, scene_graph, current_jetbot_position, current_jetbot_orientation):
        """
        Определяет позицию робота в глобальных координатах на основе RGB и depth изображений.
        
        Args:
            rgb_image (numpy.ndarray): RGB изображение с камеры робота
            depth_image (numpy.ndarray): Изображение глубины с камеры робота
            scene_graph (networkx.Graph): Граф сцены с глобальными координатами объектов
            
        Returns:
            numpy.ndarray: Позиция робота в глобальных координатах [x, y, theta]
        """
        # Получаем реальную позицию робота
        real_position = np.array([
            current_jetbot_position[0],
            current_jetbot_position[1],
            euler_from_quaternion(current_jetbot_orientation)[0]
        ])
        
        if self.save_debug:
            print("\n=== Robot Position Comparison ===")
            print("Real robot position:", real_position)

        # Сохраняем объекты из графа сцены
        self.objects = []
        for node_id, node_data in scene_graph.nodes(data=True):
            self.objects.append({
                'id': node_id,
                'name': node_data['name'],
                'bbox_center': node_data['pos'],
                'description': node_data.get('description', 'obstacle')
            })

        # Детекция объектов
        if not self.detect_objects(rgb_image, depth_image):
            if self.save_debug:
                print("Less than 2 objects detected, using previous position")
                print("Estimated position:", self.prev_robot_position)
                print("Position error:", np.linalg.norm(real_position[:2] - self.prev_robot_position[:2]))
            return self.prev_robot_position

        # Построение локального графа
        if not self.build_local_graph():
            if self.save_debug:
                print("Failed to build local graph, using previous position")
                print("Estimated position:", self.prev_robot_position)
                print("Position error:", np.linalg.norm(real_position[:2] - self.prev_robot_position[:2]))
            return self.prev_robot_position

        # Выравнивание графов
        matched_set = self.align_graphs()
        if matched_set is None:
            if self.save_debug:
                print("Graph alignment failed, using previous position")
                print("Estimated position:", self.prev_robot_position)
                print("Position error:", np.linalg.norm(real_position[:2] - self.prev_robot_position[:2]))
            return self.prev_robot_position

        # Проверяем, что позиция робота не нулевая
        if np.all(self.robot_position == 0):
            if self.save_debug:
                print("Robot position is zero, using previous position")
                print("Estimated position:", self.prev_robot_position)
                print("Position error:", np.linalg.norm(real_position[:2] - self.prev_robot_position[:2]))
            return self.prev_robot_position

        # Сохраняем текущую позицию как предыдущую для следующего вызова
        self.prev_robot_position = np.array([self.robot_position[0], self.robot_position[1], 0.0])
        
        if self.save_debug:
            print("Estimated position:", self.prev_robot_position)
            print("Position error:", np.linalg.norm(real_position[:2] - self.prev_robot_position[:2]))
        
        # Возвращаем текущую позицию робота
        return self.prev_robot_position

    def transform_graph_to_local(self, scene_graph, robot_position):
        """
        Преобразует граф сцены из глобальной системы координат в локальную систему координат робота.
        
        Args:
            scene_graph: Граф сцены в глобальной системе координат
            robot_position: Позиция робота (x, y, theta) в глобальной системе координат
            
        Returns:
            nx.Graph: Граф сцены в локальной системе координат робота
        """
        local_graph = scene_graph.copy()
        robot_x, robot_y, robot_theta = robot_position
        
        rotation_matrix = np.array([
            [np.cos(robot_theta), -np.sin(robot_theta)],
            [np.sin(robot_theta), np.cos(robot_theta)]
        ])
        
        for node in local_graph.nodes():
            current_pos = local_graph.nodes[node]['pos']
            shifted_pos = np.array(current_pos[:2]) - np.array([robot_x, robot_y])
            rotated_pos = rotation_matrix @ shifted_pos
            local_graph.nodes[node]['pos'] = [rotated_pos[0], rotated_pos[1], current_pos[2]]
            
            if 'orientation' in local_graph.nodes[node]:
                local_graph.nodes[node]['orientation'] -= robot_theta
        
        return local_graph
