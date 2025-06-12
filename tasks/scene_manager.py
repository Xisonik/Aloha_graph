import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path

from configs.main_config import MainConfig

d = Path().resolve()#.parent
general_path = str(d) + "/standalone_examples/Aloha_graph/Aloha"
log = general_path + "/logs/"

import random

class Scene_controller:
    def __init__(self):
        self.config = MainConfig()
        self.targets = []
        self.obstacles = None
        self.change_line = 0
        self.repeat = 5
        self.obstacles_shape = ["table", "chair", "human"]
        self.generate_positions_for_contole_module()
        self._init_scene()
        self.robot_r = 0.34
        self.external_management = False
        self.possible_positions_for_obstacles = None
        self.id_obstacles = None

    def generate_positions_for_contole_module(self, key=0, positions_for_obstacles=None):
        self.obstacles = []
        possible_positions_for_obstacles  = [
            [2, 1, 0.0],
            [2.8, 5, 0.0],
            # [3.4, 5, 0.0],
            [2.8, 3.5, 0.0],
            # [3.4, 3.5, 0.0],
            [2.8, 2, 0.0],
            [4, 0.6, 0.0],
            # [1.5, -0.5, 0.0],
        ]
        self.possible_positions_for_obstacles = possible_positions_for_obstacles
        if key == -1:
            key = len(possible_positions_for_obstacles)
        if key > 0:
            if positions_for_obstacles is None:
                positions_for_obstacles = random.sample(range(len(possible_positions_for_obstacles)), key)
                self.id_obstacles = positions_for_obstacles
            for position in positions_for_obstacles:
                self.obstacles.append(self._set_obstacle(
                    shape="chair",
                    position=possible_positions_for_obstacles[position]))

        return # Задаём позиции и типы препятствий
        
    def _init_scene(self):
        pass
    
    def generate_obstacles(self,key=0, positions_for_obstacles=None):
        self.generate_positions_for_contole_module(key,positions_for_obstacles=positions_for_obstacles)

        return self.obstacles
    
    def get_obstacles(self,key=0, positions_for_obstacles=None):
        if key == -1:
            self.generate_obstacles(key=-1,positions_for_obstacles=positions_for_obstacles)
        return self.obstacles, self.possible_positions_for_obstacles, self.id_obstacles

    def _set_obstacle(self, shape, position, usd_path=None, radius=0, len_x=0.5, len_y=0.5, height=0.5):
        return {
            "shape": shape,
            "position": position,
            "radius": radius,
            "len_x": len_x,
            "len_y": len_y,
            "height": height
        }

    def intersect_with_obstacles(self, robot_pos, add_r=0):
        intersect = False
        
        for obstacle in self.obstacles:
            if obstacle["shape"] == "table":
                if (np.abs(obstacle["position"][0] - robot_pos[0] + add_r) < (self.robot_r + obstacle["len_x"]) or 
                    np.abs(obstacle["position"][1] - robot_pos[1] + add_r) < (self.robot_r + obstacle["len_y"])):
                    intersect = True
            elif obstacle["shape"] == "chair":
                # Предполагаем радиус столкновения 0.5 м для стула и человека
                if np.abs(np.linalg.norm(np.array(obstacle["position"][0:2]) - robot_pos + add_r)) < (self.robot_r + 0.35):
                    intersect = True
        
        return intersect
    
    def intersect_with_walls(self, robot_pos, add_r=0):
        intersect = False

        if ((robot_pos[0] + (self.robot_r + add_r) < -0.2  and robot_pos[1] + (self.robot_r + add_r) < -0.8)
            or (robot_pos[0] - (self.robot_r + add_r) > 3.5  and robot_pos[1] + (self.robot_r + add_r) < 0.4)
            or (robot_pos[0] + (self.robot_r + add_r) < 1.2  and robot_pos[1] - (self.robot_r + add_r) > 1.2 )
            or (robot_pos[1] - (self.robot_r + add_r) > 6.8)):
            intersect = True
        
        return intersect

    def get_target_position(self):
        poses_bowl = [
                      np.array([-0.6, -1.1, 0.8]),
                      np.array([3.9, 0.1, 0.8]),
                    #   np.array([0.8, 2.5, 0.8]),
                      np.array([0.8, 3.5, 0.8]),
                    #   np.array([0.8, 4.5, 0.8]),
                      np.array([2.5, 7.1, 0.8]),
                    #   np.array([3.5, 7.1, 0.8])
                      ]

        goal_position = random.choice(range(len(poses_bowl)))
        num_of_envs = 0
        return poses_bowl[goal_position], poses_bowl, goal_position

    def get_robot_position(self, x_goal, y_goal, traning_radius=0, traning_angle=0, tuning=0):
        # return [4,3,0], -np.pi
        traning_radius_start = 1.2
        k = 0
        self.change_line += 1
        reduce_r = 1
        reduce_phi = 1
        track_width = 1.2
        
        # if self.change_line >= self.repeat:
        #     reduce_r = np.random.rand()
        #     reduce_phi = np.random.rand()
        #     self.change_line=0
        # print("reduce", reduce_r)
        # traning_radius = 0 #random.uniform(0.0, 4)
        while True:
            k += 1
            robot_pos = np.array([x_goal, y_goal, 0.1])

            alpha = np.random.rand()*2*np.pi
            robot_pos += (traning_radius_start+reduce_r*(traning_radius)) * np.array([np.cos(alpha), np.sin(alpha), 0])

            goal_world_position = np.array([x_goal, y_goal])
            nx = np.array([-1,0])
            ny = np.array([0,1])
            to_goal_vec = goal_world_position - robot_pos[0:2]
            
            cos_angle = np.dot(to_goal_vec, nx) / np.linalg.norm(to_goal_vec) / np.linalg.norm(nx)
            n = np.random.randint(2)
            
            quadrant = self._get_quadrant(nx, ny, to_goal_vec)
            # robot_pos = np.array([     2.8964,        2.17,         0.1])
            # return [     2.8964,        2.17,         0.1], 4*np.pi/2, True #quadrant*np.arccos(cos_angle) + ((-1)**n)*reduce_phi*traning_angle, True
            if (not self.intersect_with_obstacles(robot_pos[0:2], 0.1)
                and  ((robot_pos[0] > 1.7 and robot_pos[0] < 4.3 and robot_pos[1] > 1.4 and robot_pos[1] < 6.4)
                or (robot_pos[0] > 0 and robot_pos[0] < 3 and robot_pos[1] > -1 and robot_pos[1] < 0.8))):
                print("robot_pos", robot_pos, traning_radius)
                print("obstacles in get robot pos", self.obstacles)
                n = np.random.randint(2)
                return robot_pos, quadrant*np.arccos(cos_angle) + ((-1)**n)*reduce_phi*traning_angle, True
            elif k >= 1000:
                print("can't get correct robot position: ", x_goal, y_goal, robot_pos, reduce_r*traning_radius)
                return 0, 0, False
                  
    def _get_quadrant(self, nx, ny, vector):
        LR = vector[0]*nx[1] - vector[1]*nx[0]
        mult = 1
        if LR < 0:
            mult = -1
        return mult

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



def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
