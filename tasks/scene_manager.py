import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
d = Path().resolve()#.parent
general_path = str(d) + "/standalone_examples/Aloha_graph/Aloha"
log = general_path + "/logs/"

import random
class Scene_controller:
    def __init__(self):
        self.targets = []
        self.obstacles = []
        self.change_line = 0
        self.repeat = 5
        self.obstacles_shape = ["cylinder", "cube", "table"]
        self._set_obstacle_position()
        self.robot_r = 0.34

    def _set_obstacle_position(self):
        self.obstacles.append(self._set_obstacle(self.obstacles_shape[2], [0.5, 5], 0, 0.5, 1))
        self.obstacles.append(self._set_obstacle(self.obstacles_shape[0], [1, 3.5], 0.35))
        self.obstacles.append(self._set_obstacle(self.obstacles_shape[0], [2.2, 5.3], 0.2))
        self.obstacles.append(self._set_obstacle(self.obstacles_shape[1], [3, 3.5], 0, 0.05, 0.5))
        self.obstacles.append(self._set_obstacle(self.obstacles_shape[1], [5, 2.5], 0, 0.05, 0.5))
        self.obstacles.append(self._set_obstacle(self.obstacles_shape[0], [5.8, 0.7], 0.2))
        self.obstacles.append(self._set_obstacle(self.obstacles_shape[0], [7, 2.5], 0.35))
        self.obstacles.append(self._set_obstacle(self.obstacles_shape[2], [7.5, 1], 0, 0.5, 1))

    def _set_obstacle(self, shape, position, radius=0, len_x=0, len_y=0, height=0.5):
        return {"shape": shape,
                "position": position,
                "radius": radius,
                "len_x": len_x,
                "len_y":len_y,
                "height": height}

    def no_intersect_with_obstacles(self, robot_pos, add_r = 0):
        no_interect = True
        
        for obstacle in self.obstacles:
            if obstacle["shape"] == "cylinder":
                if  np.abs(np.linalg.norm(obstacle["position"] - robot_pos)) < (self.robot_r + obstacle["radius"] + add_r):
                # print(np.linalg.norm(pos_obstacle - robot_pos))
                    no_interect = False
                    # print("interect with obst shape ", obstacle["shape"])
            elif obstacle["shape"] == "cube" or obstacle["shape"] == "table":
                if  np.abs(np.linalg.norm(obstacle["position"][0] - robot_pos[0])) < (self.robot_r + obstacle["len_x"] + add_r) and np.abs(np.linalg.norm(obstacle["position"][1] - robot_pos[1])) < (self.robot_r + obstacle["len_y"] + add_r):
                    no_interect = False
                    # print("interect with obst shape ", obstacle["shape"])
            else:
                print("there is no obstacle of a given shape", obstacle["shape"])
        
        return no_interect

    def get_target_position(self, event, eval, evalp):
        poses_bowl = [[7.5, 1, 0.7],[7.5, 0.8, 0.7],[7.5, 1.2, 0.7], [0.5, 4.8, 0.7],[0.5, 5.2, 0.7],[0.5, 5, 0.7]]
        
        if event == 0:
            num_of_envs = np.random.choice([0,1,2])
        elif event == 1:
            num_of_envs = np.random.choice([3,4,5])

        if not eval and not evalp:
            goal_position = poses_bowl[num_of_envs]
        else:
            goal_position = poses_bowl[num_of_envs]

        goal_position = poses_bowl[num_of_envs]
        return goal_position, poses_bowl, num_of_envs

    def get_robot_position(self, x_goal, y_goal, traning_radius=0, traning_angle=0, tuning=0):
        # return [4,3,0], -np.pi
        traning_radius_start = 1.2
        k = 0
        self.change_line += 1
        reduce_r = 1
        reduce_phi = 1
        track_width = 1.2
        
        if self.change_line >= self.repeat:
            reduce_r = np.random.rand()
            reduce_phi = np.random.rand()
            self.change_line=0
        # print("reduce", reduce_r)
        while True:
            k += 1
            robot_pos = np.array([x_goal, y_goal, 0.1])

            alpha = np.random.rand()*2*np.pi
            robot_pos += (traning_radius_start + reduce_r*(traning_radius))*np.array([np.cos(alpha), np.sin(alpha), 0])

            goal_world_position = np.array([x_goal, y_goal])
            nx = np.array([-1,0])
            ny = np.array([0,1])
            to_goal_vec = goal_world_position - robot_pos[0:2]
            
            cos_angle = np.dot(to_goal_vec, nx) / np.linalg.norm(to_goal_vec) / np.linalg.norm(nx)
            n = np.random.randint(2)
            
            quadrant = self._get_quadrant(nx, ny, to_goal_vec)
            # return [4,3,0], quadrant*np.arccos(cos_angle) + ((-1)**n)*reduce_phi*traning_angle, True
            if self.no_intersect_with_obstacles(robot_pos[0:2], 0.25) and robot_pos[0] < 7.5-self.robot_r and robot_pos[0] > 0.5+self.robot_r and robot_pos[1] < 5.5-self.robot_r and robot_pos[1] > 0.5+self.robot_r:
                n = np.random.randint(2)
                return robot_pos, quadrant*np.arccos(cos_angle) + ((-1)**n)*reduce_phi*traning_angle, True
            elif k >= 1000:
                print("can't get correct robot position: ", robot_pos, quadrant*np.arccos(cos_angle) + reduce_phi*traning_angle, reduce_r*traning_radius)
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
