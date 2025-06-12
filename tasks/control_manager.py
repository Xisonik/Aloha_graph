import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
import random
import networkx as nx
from collections import deque
import json
import os

# Путь к директории логов
d = Path().resolve()
general_path = str(d) + "/standalone_examples/Aloha_graph/Aloha"
log = general_path + "/logs/"


class PID_controller:
    def __init__(self, kp, ki, kd):
        self.kp = kp  # Пропорциональный коэффициент
        self.ki = ki  # Интегральный коэффициент
        self.kd = kd  # Дифференциальный коэффициент
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        # Интегральная составляющая
        self.integral += error * dt
        
        # Дифференциальная составляющая
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        
        # Общий результат
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # Сохранение предыдущей ошибки
        self.prev_error = error
        return output


class Control_module:
    def __init__(self, kp_linear=0.5, ki_linear=0.01, kd_linear=0.02, kp_angular=1, ki_angular=0.01, kd_angular=0.05):
        self.path = []
        self.current_pos = 0
        self.kp_linear = kp_linear
        self.ki_linear = ki_linear
        self.kd_linear = kd_linear
        self.kp_angular = kp_angular
        self.ki_angular = ki_angular
        self.kd_angular = kd_angular

        # Errors for linear control
        self.prev_linear_error = 0
        self.integral_linear = 0

        # Errors for angular control
        self.prev_angular_error = 0
        self.integral_angular = 0
        self.targets_position = []
        self.target_position = np.array([0,0])
        self.goal = (0,0)
        self.ratio = 10
        self.start = True
        self.end = False
        self.ferst_ep = True
        self.dijkstra_first_calling = True
        self.first = True
        self.shift=[0,0]
        self.scene_controller = None
    
    def update(self, current_position, target_position, targets_positions, scene_controller=None, key=0):
        print("f_u ", scene_controller)
        self.ferst_ep = True
        self.start = True
        self.end = False
        algorithm = 1
        self.target_position = target_position
        self.targets_positions = targets_positions
        self.scene_controller = scene_controller
        self.get_path(current_position, target_position, algorithm, scene_controller=scene_controller, key=key)

    def _create_grid_with_diagonals(self, width, height):
        print("f112")
        graph = nx.grid_2d_graph(width, height)
        for x in range(width):
            for y in range(height):
                if x + 1 < width and y + 1 < height:
                    graph.add_edge((x, y), (x + 1, y + 1))
                if x + 1 < width and y - 1 >= 0:
                    graph.add_edge((x, y), (x + 1, y - 1))
        return graph
    
    def find_boundary_nodes(self, graph):
        print("f113")
        boundary_nodes = set()
        max_degree = max(dict(graph.degree()).values())
        for node in graph.nodes():
            if graph.degree(node) < max_degree:
                boundary_nodes.add(node)
        return boundary_nodes

    def find_expanded_boundary(self, graph, boundary_nodes, excluded_nodes=set()):
        print("f114")
        expanded_boundary = set()
        for node in boundary_nodes:
            expanded_boundary.update(
                neighbor for neighbor in graph.neighbors(node) if neighbor not in excluded_nodes
            )
        return expanded_boundary - boundary_nodes

    def assign_edge_weights(self, graph, boundary_nodes, expanded_boundary):
        print("f115")
        for u, v in graph.edges():
            if v in boundary_nodes or u in boundary_nodes:
                graph[u][v]['weight'] = 3
            elif v in expanded_boundary or u in expanded_boundary:
                graph[u][v]['weight'] = 2
            else:
                graph[u][v]['weight'] = 1

    def remove_straight_segments(self, path):
        print("f4")
        if len(path) < 3:
            return path
        filtered_path = [path[0]]
        for i in range(1, len(path) - 1):
            x1, y1 = path[i - 1]
            x2, y2 = path[i]
            x3, y3 = path[i + 1]
            if (x3 - x1) * (y2 - y1) != (y3 - y1) * (x2 - x1):
                filtered_path.append(path[i])
        filtered_path.append(path[-1])
        return filtered_path
    
    def heuristic(self, node, goal, prev_node):
        print("f116")
        dx = abs(node[0] - goal[0])
        dy = abs(node[1] - goal[1])
        if prev_node:
            prev_dx = node[0] - prev_node[0]
            prev_dy = node[1] - prev_node[1]
            if prev_dx != 0 or prev_dy != 0:
                angle_penalty = abs((dx * prev_dy - dy * prev_dx) / (dx * dx + dy * dy))
                return dx + dy + angle_penalty
        return dx + dy

    def find_nearest_reachable_node(self, graph, target):
        print("f3")
        # print("get nearest cell: ", target)
        if target in graph and len(list(graph.neighbors(target))) > 0:
            return target
        min_distance = float('inf')
        nearest_node = None
        target_x, target_y = target
        for node in graph.nodes:
            if len(list(graph.neighbors(node))) > 0:
                x, y = node
                distance = abs(target_x - x) + abs(target_y - y)
                if distance < min_distance:
                    min_distance = distance
                    nearest_node = node
        # print("get nearest cell: ", target, nearest_node)
        return nearest_node
    
    def get_path(self, current_position, target_position, algorithm=1, scene_controller=None, save_to_disk=False, key=None):
        print("f2 ", key)
        shift = self.shift
        grid_path = []
        if algorithm == 0:
            grid_path = self.get_path_A_star(current_position, target_position, key=key)
        elif algorithm == 1:
            grid_path = self.get_path_dijkstra(current_position, target_position, key=key)
        if not grid_path:
            return []
        semple_path = self.remove_zigzags(grid_path)
        trinned_grid_path = self.remove_straight_segments(semple_path)
        if save_to_disk or self.first:
            self.first = False
            self.save_to_disk(trinned_grid_path, grid_path, scene_controller=scene_controller, key=key)
        self.path = [np.array([(point[0])/self.ratio+shift[0], (point[1])/self.ratio+shift[1]]) for point in trinned_grid_path]
        return self.path
        
    def get_path_dijkstra(self, current_position, target_position, scene_controller=None, key=0):
        print("f1 ", scene_controller)
        
        ratio = self.ratio
        shift = self.shift
        print("shift ", shift)
        grid_graph = self.get_scene_grid(scene_controller)
        start = self.find_nearest_reachable_node(grid_graph, (int((current_position[0]-shift[0])*ratio), int((current_position[1]-shift[1])*ratio)))
        goal = self.find_nearest_reachable_node(grid_graph, (int((target_position[0]-shift[0])*ratio), int((target_position[1]-shift[1])*ratio)))
        print("curren position is ", current_position, start)
        print("goal position is ", target_position, goal)
        paths_file = os.path.join(log, "all_paths.json")

        if self.dijkstra_first_calling:
            print("dijkstra init, should work ones")
            self.dijkstra_first_calling = False
            self.all_paths = {}

            if os.path.exists(paths_file):
                print(f"Loading paths from {paths_file}")
                try:
                    with open(paths_file, 'r') as f:
                        loaded_paths = json.load(f)
                    # Преобразуем строковые ключи обратно в кортежи
                    for config_key, targets in loaded_paths.items():
                        # k0, k1 = map(int, config_key.split('_'))
                        # config_tuple = (k0, k1)
                        self.all_paths[config_key] = {}
                        for target_str, nodes in targets.items():
                            target = tuple(map(int, target_str.split(',')))
                            self.all_paths[config_key][target] = {}
                            for node_str, path in nodes.items():
                                node = tuple(map(int, node_str.split(',')))
                                self.all_paths[config_key][target][node] = [tuple(p) for p in path]
                    print(f"Loaded {len(self.all_paths)} configurations")
                except Exception as e:
                    print(f"Error loading paths file: {e}")
                    self.all_paths = {}
            else:
                print(f"No paths file found at {paths_file}, please generate paths first")
                from .scene_manager import Scene_controller
                scene_controller = Scene_controller()
                
                scene_controller.generate_positions_for_contole_module(key=-1)
                _, obstacles, _ = scene_controller.get_obstacles()
                from itertools import combinations

                arr = range(len(obstacles))
                results = []

                for r in range(1, len(arr) + 1):
                    for combo in combinations(arr, r):
                        combined = ''.join(str(x) for x in combo)
                        results.append(combined)
                # results.insert(0, '')
                all_configs = results
                missing_configs = [config for config in all_configs if config not in self.all_paths]
                print("missing_configs is ", missing_configs)
                for config_key in missing_configs:
                    print(f"Computing paths for configuration: {config_key}")
                    positions_for_obstacles = [int(ch) for ch in config_key]
                    scene_controller.generate_positions_for_contole_module(key=len(positions_for_obstacles), positions_for_obstacles=positions_for_obstacles)
                    config_graph = self.get_scene_grid(scene_controller=scene_controller)
                    targets = []
                    print("self.targets_positions is ", self.targets_positions)
                    for i in range(len(self.targets_positions)):
                        target_node = self.find_nearest_reachable_node(
                            config_graph,
                            (int((self.targets_positions[i][0]-shift[0])*ratio), int((self.targets_positions[i][1]-shift[1])*ratio))
                        )
                        if target_node:
                            targets.append(target_node)
                    self.all_paths[config_key] = {target: {} for target in targets}
                    for target in targets:
                        for node in config_graph.nodes():
                            if nx.has_path(config_graph, node, target):
                                path = nx.shortest_path(config_graph, source=node, target=target, weight='weight')
                                self.all_paths[config_key][target][node] = path

                # Сохраняем в JSON, преобразуя кортежи в строки
                json_paths = {}
                for config_key, targets in self.all_paths.items():
                    config_str = f"{config_key}"
                    json_paths[config_str] = {}
                    for target, nodes in targets.items():
                        target_str = f"{target[0]},{target[1]}"
                        json_paths[config_str][target_str] = {}
                        for node, path in nodes.items():
                            node_str = f"{node[0]},{node[1]}"
                            json_paths[config_str][target_str][node_str] = [list(p) for p in path]
                
                print(f"Saving paths to {paths_file}")
                try:
                    with open(paths_file, 'w') as f:
                        json.dump(json_paths, f, indent=4)
                    print(f"Saved {len(self.all_paths)} configurations")
                except Exception as e:
                    print(f"Error saving paths file: {e}")

        if scene_controller is  None:
            scene_controller = self.scene_controller
        _, _, obstacles_id = scene_controller.get_obstacles()
        key = ''.join(str(x) for x in sorted(obstacles_id))
        import time
        start_time = time.time()

        path = self.all_paths.get(f"{key}", {}).get(goal, {}).get(start, None)
        end_time = time.time()  # Засекаем время окончания

        elapsed_time = end_time - start_time  # Вычисляем разницу
        print(f"Время выполнения: {elapsed_time:.4f} секунд")
        print(f"path for config {key}, start {start}, goal {goal}: ", path)
        if path is None:
            print(f"No path found for config {key}, start {start}, goal {goal}")
            return []
        return path
        
    def get_scene_grid(self, scene_controller = None):
        print("f0")
        room_len_x = 6
        room_len_y = 8
        ratio = 10
        self.ratio = ratio
        ratio_x = ratio * room_len_x
        ratio_y = ratio * room_len_y
        shift = self.shift
        
        if scene_controller is None:
            from .scene_manager import Scene_controller
            scene_controller = Scene_controller()
            scene_controller.generate_positions_for_contole_module(key=-1)

        if not hasattr(self, 'scene_graphs_cache'):
            print("Initializing scene graphs cache")
            self.scene_graphs_cache = {}
            
            # Load all graphs at initialization
            graphs_dir = os.path.join(log, "graphs")
            os.makedirs(graphs_dir, exist_ok=True)
            _, obstacles, _ = scene_controller.get_obstacles()
            k_max = len(obstacles)
            from itertools import combinations

            arr = range(len(obstacles))
            results = []

            for r in range(1, len(arr) + 1):
                for combo in combinations(arr, r):
                    combined = ''.join(str(x) for x in combo)
                    results.append(combined)

            print(results)
            # Load all possible configurations (8x8 = 64 graphs)
            for k in results:
                graph_file = os.path.join(graphs_dir, f"graph_{k}.json")
                if os.path.exists(graph_file):
                    try:
                        with open(graph_file, 'r') as f:
                            graph_data = json.load(f)
                        G = nx.Graph()
                        for edge in graph_data["edges"]:
                            u = tuple(edge["u"])
                            v = tuple(edge["v"])
                            G.add_edge(u, v, weight=edge["weight"])
                        self.scene_graphs_cache[k] = G
                        print(f"Loaded graph for configuration ({k})")
                    except Exception as e:
                        print(f"Error loading graph file {graph_file}: {e}")
                else:
                    positions_for_obstacles = [int(ch) for ch in k]
                    print("obstacles in make grid", positions_for_obstacles)
                    scene_controller.generate_positions_for_contole_module(key=len(positions_for_obstacles), positions_for_obstacles=positions_for_obstacles)
                    
                    G = self._create_grid_with_diagonals(ratio_x, ratio_y)
                    for edge in list(G.edges):
                        node_1, node_2 = edge
                        if (scene_controller.intersect_with_obstacles(np.array([(node_1[0])/ratio+shift[0], (node_1[1])/ratio+shift[1]]))
                            or scene_controller.intersect_with_obstacles(np.array([(node_2[0])/ratio+shift[0], (node_2[1])/ratio+shift[1]]))
                            or scene_controller.intersect_with_walls(np.array([(node_1[0])/ratio+shift[0], (node_1[1])/ratio+shift[1]]), add_r=1/ratio)
                            or scene_controller.intersect_with_walls(np.array([(node_2[0])/ratio+shift[0], (node_2[1])/ratio+shift[1]]), add_r=1/ratio)):
                            G.remove_edge(node_1, node_2)

                    boundary_nodes = self.find_boundary_nodes(G)
                    expanded_boundary = [boundary_nodes]
                    levels = 3
                    for i in range(levels):
                        expanded_boundary.append(self.find_expanded_boundary(G, expanded_boundary[-1]))
                    expanded_boundary.reverse()

                    for u, v in G.edges():
                        G[u][v]['weight'] = 1
                    for u, v in G.edges():
                        for i in range(1, len(expanded_boundary)):
                            set_nodes = expanded_boundary[i]
                            prev_set_nodes = expanded_boundary[i-1]
                            if (v in prev_set_nodes and u in set_nodes) or (v in set_nodes and u in set_nodes):
                                G[u][v]['weight'] = 1 + i

                    # Cache the created graph
                    self.scene_graphs_cache[k] = G

                    # Save to file for future use
                    graph_data = {
                        "edges": [{"u": list(u), "v": list(v), "weight": G[u][v]["weight"]} for u, v in G.edges()]
                    }
                    try:
                        with open(graph_file, 'w') as f:
                            json.dump(graph_data, f, indent=4)
                        print(f"Saved graph for configuration ({k})")
                    except Exception as e:
                        print(f"Error saving graph file {graph_file}: {e}")

        # Return graph from cache
        _, _, obstacles_id = scene_controller.get_obstacles()
        key = ''.join(str(x) for x in sorted(obstacles_id))
        return self.scene_graphs_cache[key]
    
    def get_path_A_star(self, current_position, target_position, scene_controller=None, key=None):
        print("f11")
        shift = self.shift
        ratio = self.ratio
        G = self.get_scene_grid(scene_controller)
        heuristic = lambda a, b: math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        start = self.find_nearest_reachable_node(G, (int((current_position[0]-shift[0])*ratio), int((current_position[1]-shift[1])*ratio)))
        goal = self.find_nearest_reachable_node(G, (int((target_position[0]-shift[0])*ratio), int((target_position[1]-shift[0])*ratio)))
        print("goal_graph is ", goal, target_position)
        self.goal = goal
        grid_path = nx.astar_path(G, start, self.goal, heuristic=heuristic)
        return grid_path

    def save_to_disk(self, trinned_grid_path, grid_path, scene_controller=None, key=None):
        print("f6")
        G = self.get_scene_grid(scene_controller)
        pos = {node: node for node in G.nodes}
        node_colors = []
        for node in G.nodes:
            if node in trinned_grid_path:
                node_colors.append('green')
            elif node in grid_path:
                node_colors.append('red')
            else:
                node_colors.append('lightblue')
        plt.figure(figsize=(32, 32))
        nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=200)
        edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, font_color='red')
        plt.title(f"Граф с узлами, соответствующими клеткам сетки (config {key})")
        graph_img_file = os.path.join(log, f"grid_graph_{key}.png") if key else os.path.join(log, "grid_graph.png")
        plt.savefig(graph_img_file, dpi=200)
        plt.close()
        print("grid path:", grid_path)

    def remove_zigzags(self, path):
        print("f119")
        if len(path) < 3:
            return path
        simplified_path = [path[0]]
        for i in range(1, len(path) - 1):
            x1, y1 = simplified_path[-1]
            x2, y2 = path[i]
            x3, y3 = path[i + 1]
            if (x1 == x3 or y1 == y3 or abs(x1 - x3) == abs(y1 - y3)):
                continue
            simplified_path.append(path[i])
        simplified_path.append(path[-1])
        return simplified_path
 
    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def get_lookahead_point(self, current_position, lookahead_distance=0.7):
        path = self.path
        for i in reversed(range(len(path) - 1)):
            segment_start = np.array(path[i])
            segment_end = np.array(path[i + 1])
            segment_vector = segment_end - segment_start
            segment_length = np.linalg.norm(segment_vector)
            # print("segment_start", segment_start)
            to_segment_start = current_position - segment_start
            projection = np.dot(to_segment_start, segment_vector) / segment_length
            if projection < 0:
                closest_point = segment_start
            elif projection > segment_length:
                closest_point = segment_end
            else:
                closest_point = segment_start + (segment_vector / segment_length) * projection
            distance_to_closest = np.linalg.norm(current_position - closest_point)
            if distance_to_closest <= lookahead_distance:
                remaining_distance = lookahead_distance - distance_to_closest
                lookahead_point = closest_point + (segment_vector / segment_length) * remaining_distance
                return lookahead_point
        if np.linalg.norm(current_position - path[-1]) < 0.3:
            self.end = True
        return np.array(path[-1])

    def pure_pursuit_controller(self, current_position, current_orientation_euler, linear_velocity=0.3, lookahead_distance=0.35):
        # print("current_position: ", current_position)
        # print("lookahead_distance ", lookahead_distance)
        if np.linalg.norm(self.target_position[0:2] - current_position[0:2]) < 1 or np.linalg.norm(self.path[-1][0:2] - current_position[0:2]) < max(0.2,1/self.ratio):
            self.end = True
        current_heading = -np.pi - current_orientation_euler if current_orientation_euler < 0 else np.pi - current_orientation_euler
        # print("current_heading: ", math.degrees(current_heading), current_heading)
        # print("star end ", self.start, self.end)
        if not self.start and not self.end:
            lookahead_point = self.get_lookahead_point(current_position, lookahead_distance)
            # print("lookahead_point: ", lookahead_point)
            to_target = lookahead_point - current_position
            # print("to_target: ", to_target)
            target_angle = np.arctan2(to_target[1], to_target[0])
            # print("target_angle: ", math.degrees(target_angle), target_angle)
            alpha = self.normalize_angle(target_angle - current_heading)
            # print("alpha: ", math.degrees(alpha), alpha)апк
            curvature = 2 * np.sin(alpha) / lookahead_distance
            angular_velocity = curvature * linear_velocity
            # print("angular_velocity: ", angular_velocity)
            max_angular_velocity = math.pi*0.4
            angular_velocity = np.clip(angular_velocity, -max_angular_velocity, max_angular_velocity)
            return linear_velocity*(max_angular_velocity-angular_velocity)/max_angular_velocity, angular_velocity
        else:
            if self.ferst_ep:
                self.ferst_ep = False
                return 0, 0
            angular_velocity = 1
            linear_velocity = 0
            nx = np.array([-1,0])
            ny = np.array([0,1])
            # print("check", self.target_position[0:2], current_position[0:2])
            to_goal_vec = self.target_position[0:2] - current_position[0:2] if self.end else self.path[1] - current_position[0:2] 
            # if self.end:
                # print("target pos ", self.target_position, to_goal_vec)
            quadrant = self.get_quadrant(nx, ny, to_goal_vec)
            cos_angle = np.dot(to_goal_vec, nx) / np.linalg.norm(to_goal_vec) / np.linalg.norm(nx)
            # print("need angle ", math.degrees(quadrant*np.arccos(cos_angle)))
            true_angle = quadrant*np.arccos(cos_angle)
            normalize_angle_lambda = lambda angle: angle if angle >= 0 else angle + 2 * math.pi
            angle_1 = normalize_angle_lambda(true_angle) + np.pi
            if angle_1 >= 2*np.pi:
                angle_1 -= 2*np.pi
            # print("angle_1 ", math.degrees(angle_1))
            if angle_1 == 2*np.pi:
                angle_1 = 0
            angle_2 = 2*np.pi - normalize_angle_lambda(current_heading)
            # print("angle_2 ", math.degrees(angle_2))
            if angle_2 == 2*np.pi:
                angle_2 = 0
            sign = 1
            if angle_2 > angle_1:
                if angle_2 - angle_1 < 2*np.pi - angle_2 + angle_1:
                    sign = 1
                else:
                    sign = -1
            elif angle_1 > angle_2:
                if angle_1 - angle_2 < 2*np.pi - angle_1 + angle_2:
                    sign = -1
                else:
                    sign = 1
            angular_velocity *= sign
            if self.start and abs(angle_1-angle_2) < np.pi/80:
                # print("end turning to path start with angle", math.degrees(abs(true_angle)))
                self.start = False
            return linear_velocity, angular_velocity
    
    def get_quadrant(self, nx, ny, vector):
        LR = vector[0]*nx[1] - vector[1]*nx[0]
        mult = 1 if LR >= 0 else -1
        return mult


from pprint import pprint
from embed_nn import SceneEmbeddingNetwork
import torch.optim as optim


class Graph_manager:
    def __init__(self):
        pass

    def init_embedding_nn(self):
        device = self.device
        self.embedding_net = SceneEmbeddingNetwork(object_feature_dim=518).to(device)
        self.embedding_optimizer = optim.Adam(self.embedding_net.parameters(), lr=0.001)


def euler_from_quaternion(vec):
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
    return roll_x, pitch_y, yaw_z


def get_quaternion_from_euler(roll, yaw=0, pitch=0):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return np.array([qx, qy, qz, qw])
