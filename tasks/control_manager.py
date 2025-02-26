import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
d = Path().resolve()#.parent
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

import random
import networkx as nx
import numpy as np
from collections import deque
from .scene_manager import Scene_controller

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
        self.ratio = 1
        self.start = True
        self.end = False
        self.ferst_ep = True
        self.G = self.get_scene_grid()
        self.dijkstra_first_calling = True
        self.first = True
    
    def update(self, current_position, target_position, targets_positions):
        self.ferst_ep = True
        self.start = True
        self.end = False
        algorithm = 1
        self.target_position = target_position
        self.targets_positions = targets_positions
        self.get_path(current_position, target_position, algorithm)

    def _create_grid_with_diagonals(self, width, height):
        # Создаем стандартную сетку
        graph = nx.grid_2d_graph(width, height)

        # Добавляем диагональные ребра
        for x in range(width):
            for y in range(height):
                if x + 1 < width and y + 1 < height:
                    graph.add_edge((x, y), (x + 1, y + 1))  # Диагональ вниз вправо
                if x + 1 < width and y - 1 >= 0:
                    graph.add_edge((x, y), (x + 1, y - 1))  # Диагональ вверх вправо

        return graph
    
    def find_boundary_nodes(self, graph):
        boundary_nodes = set()
        max_degree = max(dict(graph.degree()).values())
        for node in graph.nodes():
            neighbors = set(graph.neighbors(node))
            
            # Узлы, у которых есть соседние недостижимые (препятствия)
            if graph.degree(node) < max_degree:
                boundary_nodes.add(node)

        return boundary_nodes

    def find_expanded_boundary(self, graph, boundary_nodes, excluded_nodes=set()):
        expanded_boundary = set()
        
        for node in boundary_nodes:
            expanded_boundary.update(
                neighbor for neighbor in graph.neighbors(node) if neighbor not in excluded_nodes
            )
        
        return expanded_boundary - boundary_nodes

    def assign_edge_weights(self, graph, boundary_nodes, expanded_boundary):
        for u, v in graph.edges():
            if v in boundary_nodes or u in boundary_nodes:
                graph[u][v]['weight'] = 3
            elif v in expanded_boundary or u in expanded_boundary:
                graph[u][v]['weight'] = 2
            else:
                graph[u][v]['weight'] = 1

    def remove_straight_segments(self, path):
        print("f4")
        """
        Убирает все прямые участки из пути, оставляя только точки изгибов.
        :param path: список точек [(x1, y1), (x2, y2), ...]
        :return: сокращенный путь
        """
        if len(path) < 3:
            return path  # Если точек меньше 3, ничего не удаляем

        filtered_path = [path[0]]  # Добавляем начальную точку

        for i in range(1, len(path) - 1):
            x1, y1 = path[i - 1]
            x2, y2 = path[i]
            x3, y3 = path[i + 1]

            # Проверяем, лежат ли три точки на одной прямой (проверка коллинеарности)
            if (x3 - x1) * (y2 - y1) != (y3 - y1) * (x2 - x1):
                filtered_path.append(path[i])  # Оставляем только точки изгибов

        filtered_path.append(path[-1])  # Добавляем последнюю точку
        return filtered_path
    
    def heuristic(node, goal, prev_node):
        dx = abs(node[0] - goal[0])
        dy = abs(node[1] - goal[1])
        if prev_node:
            # Штраф за изменение направления
            prev_dx = node[0] - prev_node[0]
            prev_dy = node[1] - prev_node[1]
            if prev_dx != 0 or prev_dy != 0:
                angle_penalty = abs((dx * prev_dy - dy * prev_dx) / (dx * dx + dy * dy))
                return dx + dy + angle_penalty
        return dx + dy

    from collections import deque

    def find_nearest_reachable_node(self, graph, target):
        print("f3")
        """
        Находит ближайший узел с рёбрами к целевому узлу.
        
        :param graph: networkx.Graph — граф
        :param target: tuple (x, y) — целевой узел
        :return: ближайший узел с рёбрами или None, если таких нет
        """
        if target in graph and len(list(graph.neighbors(target))) > 0:
            return target  # Если у цели есть рёбра, то возвращаем её саму
        
        min_distance = float('inf')
        nearest_node = None
        
        target_x, target_y = target  # Разбираем координаты целевого узла

        for node in graph.nodes:
            if len(list(graph.neighbors(node))) > 0:  # Узел имеет рёбра
                x, y = node
                distance = abs(target_x - x) + abs(target_y - y)  # Манхэттенская метрика

                if distance < min_distance:  # Если нашли более близкий узел
                    min_distance = distance
                    nearest_node = node

        return nearest_node  # Вернёт ближайший узел с рёбрами или None
    
    def get_path(self, current_position, target_position, algoritm=1, save_to_disk=False):
        print("f2")
        grid_path = []
        if algoritm == 0:
            grid_path = self.get_path_A_star(current_position, target_position)
        elif algoritm == 1:
            grid_path = self.get_path_dijkstra(current_position, target_position)

        semple_path = self.remove_zigzags(grid_path)
        trinned_grid_path = self.remove_straight_segments(semple_path)
        if save_to_disk or self.first:
            self.first = False
            self.save_to_disk(trinned_grid_path, grid_path)
        
        self.path = []
        for point in trinned_grid_path:
            self.path.append(np.array([point[0]/self.ratio, point[1]/self.ratio]))
        print("path:", self.path)
        return self.path
        
        
    def get_path_dijkstra(self, current_position, target_position,):
        print("f1")
        grid_graph = self.G
        ratio = self.ratio
        start = self.find_nearest_reachable_node(grid_graph, (int(current_position[0]*ratio), int(current_position[1]*ratio)))
        # goal = self.find_nearest_reachable_node(grid_graph, (int(target_position[0]*ratio), int(target_position[1]*ratio)))
        goal = self.find_nearest_reachable_node(grid_graph, (int(target_position[0]*ratio), int(target_position[1]*ratio)))           
        if self.dijkstra_first_calling:
            self.dijkstra_first_calling = False
            targets = []
            for i in range(len(self.targets_positions)):
                targets.append(self.find_nearest_reachable_node(grid_graph, (int(self.targets_positions[i][0]*ratio), int(self.targets_positions[i][1]*ratio))))#tuple())
            print("work ", targets)

            """
            Находит кратчайшие пути от каждой доступной вершины до каждой цели с использованием алгоритма Дейкстры.
            
            :param grid_graph: Взвешенный граф (networkx.Graph)
            :param targets: Список целевых вершин
            :return: Словарь {целевой узел: {узел: [список узлов пути]}}
            """
            self.all_paths = {target: {} for target in targets}  # Словарь для хранения кратчайших путей
            
            for target in targets:
                for node in grid_graph.nodes():
                    if nx.has_path(grid_graph, node, target):  # Проверяем, существует ли путь
                        path = nx.shortest_path(grid_graph, source=node, target=target, weight='weight')
                        self.all_paths[target][node] = path
        path = self.all_paths.get(goal, {}).get(start, None)

        return path
        
    def get_scene_grid(self):
        print("f0")
        room_len_x = 8
        room_len_y = 6
        ratio = 10 # 1m/4 = 0.25m
        self.ratio = ratio
        ratio_x = ratio*room_len_x
        ratio_y = ratio*room_len_y

        scene_controller = Scene_controller()

        G = self._create_grid_with_diagonals(ratio_x, ratio_y)
        for edge in G.edges:
            node_1, node_2 = edge
            if not scene_controller.no_intersect_with_obstacles(np.array([node_1[0]/ratio,node_1[1]/ratio]),add_r = 0.1) or not scene_controller.no_intersect_with_obstacles(np.array([node_2[0]/ratio,node_2[1]/ratio]), add_r = 1/ratio):
                G.remove_edge(node_1, node_2)
        # G.add_edges_from(edges)
        boundary_nodes = self.find_boundary_nodes(G)
        expanded_boundary = []
        expanded_boundary.append(boundary_nodes)
        levels = 3
        for i in range(levels):
            expanded_boundary.append(self.find_expanded_boundary(G, expanded_boundary[-1]))
        expanded_boundary.reverse()
        for u, v in G.edges():
            G[u][v]['weight'] = 1
        for u, v in G.edges():
            for i in range(1,len(expanded_boundary)):
                set_nodes = expanded_boundary[i]
                prev_set_nodes = expanded_boundary[i-1]
                if (v in prev_set_nodes and u in set_nodes) or (v in set_nodes and u in set_nodes):
                    G[u][v]['weight'] = 1 + i
        # Найдём узлы со степенью 0
        isolated_nodes = [node for node, degree in dict(G.degree()).items() if degree == 0]
        # G.remove_nodes_from(isolated_nodes)

        return G
    
    def get_path_A_star(self, current_position, target_position):
        print("f11")
        ratio = self.ratio
        G = self.G

        # Функция эвристики (манхэттенское расстояние)
        heuristic = lambda a, b: math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        start = self.find_nearest_reachable_node(G, (int(current_position[0]*ratio), int(current_position[1]*ratio)))
        goal = self.find_nearest_reachable_node(G, (int(target_position[0]*ratio), int(target_position[1]*ratio)))
        print("goal_graph is ", goal, target_position)
        self.goal = goal#(4,4)
        grid_path = nx.astar_path(G, start, self.goal, heuristic=heuristic)

        return grid_path

    def save_to_disk(self, trinned_grid_path, grid_path):
        print("f6")
        G = self.G
        pos = {node: node for node in G.nodes}  # Узел (x, y) будет на позиции (x, y)
        node_colors = []
        for node in G.nodes:
            if node in trinned_grid_path:
                node_colors.append('green')  # Выделенный цвет
            elif node in grid_path:
                node_colors.append('red')  # Выделенный цвет
            else:
                node_colors.append('lightblue')  # Обычный цвет
        # Визуализируем граф
        plt.figure(figsize=(32, 32))
        nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=200)
        edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, font_color='red')
        plt.title("Граф с узлами, соответствующими клеткам сетки")
        
        # Сохраняем изображение в файл
        plt.savefig(log + "grid_graph.png", dpi=200)
        print("grid path:", grid_path)

    def remove_zigzags(self, path):
        if len(path) < 3:
            return path  # Если путь слишком короткий, не трогаем его

        simplified_path = [path[0]]  # Начинаем с первого узла

        for i in range(1, len(path) - 1):
            x1, y1 = simplified_path[-1]  # Последняя сохранённая точка
            x2, y2 = path[i]  # Текущая точка
            x3, y3 = path[i + 1]  # Следующая точка
            
            # Проверяем, лежит ли точка (x2, y2) на одной прямой или диагонали с (x1, y1) и (x3, y3)
            if (x1 == x3 or y1 == y3 or abs(x1 - x3) == abs(y1 - y3)):  
                continue  # Точка (x2, y2) не нужна, пропускаем её
            
            simplified_path.append(path[i])  # Добавляем только важные точки

        simplified_path.append(path[-1])  # Добавляем последний узел

        return simplified_path
                                                                                      
    
    def get_PID_controle_velocity(self, current_position, current_orientation_euler, target_position):
        
        """
        Управляет движением робота с помощью PID-контроллера.

        :param current_position: tuple, текущее положение робота (x, y).
        :param current_orientation_euler: float, текущий угол ориентации робота (yaw) в радианах.
        :param target_position: tuple, целевая позиция (x, y).
        :param dt: float, прошедшее время (разница между текущим и предыдущим временем).
        :return: tuple, линейная и угловая скорости (v, omega).
        """
        print("current_step ", self.current_pos)
        if len(self.path) == 0:
            print("error, path length is 0")
            return 0, 0
        dt = 0.1
        # Расчет ошибок по положению
        error_x = self.path[self.current_pos+1][0] - current_position[0]
        error_y = self.path[self.current_pos+1][1] - current_position[1]
        # Перевод ошибки положения в полярные координаты
        distance_error = math.sqrt(error_x**2 + error_y**2)

        error_x = self.path[self.current_pos+2][0]/self.ratio - current_position[0]
        error_y = self.path[self.current_pos+2][1]/self.ratio - current_position[1]
        target_angle = math.atan2(error_y, error_x)

        # Угловая ошибка
        angular_error = target_angle - current_orientation_euler

        # Нормализация угловой ошибки в диапазон [-pi, pi]
        angular_error = (angular_error + math.pi) % (2 * math.pi) - math.pi

        # PID для линейной скорости
        self.integral_linear += distance_error * dt
        derivative_linear = (distance_error - self.prev_linear_error) / dt
        linear_velocity = (
            self.kp_linear * distance_error +
            self.ki_linear * self.integral_linear +
            self.kd_linear * derivative_linear
        )
        self.prev_linear_error = distance_error

        # PID для угловой скорости
        self.integral_angular += angular_error * dt
        derivative_angular = (angular_error - self.prev_angular_error) / dt
        angular_velocity = (
            self.kp_angular * angular_error +
            self.ki_angular * self.integral_angular +
            self.kd_angular * derivative_angular
        )
        self.prev_angular_error = angular_error

        # Ограничение скоростей
        angular_velocity = max(min(angular_velocity, -0.5), 0.5)
        linear_velocity = max(min(linear_velocity, 0), 1)

        if np.linalg.norm(self.path[self.current_pos+1][0:2]-current_position[0:2]) < 0.2:
            self.current_pos += 1

        return linear_velocity, angular_velocity
    
    def get_vector_controle_velocity(self):
        v, w = 0, 0
        goal_world_position = self.goal_position
        goal_world_position[2] = 0
        pos_obstacles = np.array([4.1,-0.6, 0])#([[3.5,-0.8],[4,1]])
        current_jetbot_position, current_jetbot_orientation = self.jetbot.get_world_pose()
        r_g = 0.95
        r_o = 0.63
        next_position = [np.cos(euler_from_quaternion(current_jetbot_orientation)[0]), np.sin(euler_from_quaternion(current_jetbot_orientation)[0])]
        
        to_goal_vec = goal_world_position[0:2] - current_jetbot_position[0:2]
        to_obs_vec = pos_obstacles[0:2] - current_jetbot_position[0:2]
        current_dist_to_goal = np.linalg.norm(to_goal_vec)
        current_dist_to_obs = np.linalg.norm(to_obs_vec)

        nx = np.array([-1,0])
        ny = np.array([0,1])

        v_g = 0.6 if current_dist_to_goal > (r_g + 0.15) else ((current_dist_to_goal+0.05)/r_g - 1)
        v_o = 0 if current_dist_to_obs > (r_o + 0.45) else 0.5 - (current_dist_to_obs - r_o)
        move_vec = to_goal_vec/current_dist_to_goal*v_g - to_obs_vec/current_dist_to_obs*v_o
        quadrant = self.get_quadrant(nx, ny, move_vec)
        cos_angle = np.dot(move_vec, nx) / np.linalg.norm(move_vec) / np.linalg.norm(nx)
        delta_angle = (euler_from_quaternion(current_jetbot_orientation)[0] - quadrant*np.arccos(cos_angle))
        sign = np.sign(delta_angle)
        delta_angle = math.degrees(abs(delta_angle))
        orientation_error = delta_angle if delta_angle < 180 else (360 - delta_angle)
        w =  sign * min(orientation_error/10, 1.7)
        v = min(v_g,0.8) - min(abs(w/5), 0.7)

        return v, w

    def normalize_angle(self, angle):
        """Нормализует угол в диапазон [-π, π]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def get_lookahead_point(self, current_position, lookahead_distance=0.7):
        """
        Находит точку на пути, которая находится на расстоянии lookahead_distance от текущей позиции.
        :param path: Массив точек пути [[x1, y1], [x2, y2], ...].
        :param current_position: Текущая позиция робота [x, y].
        :param lookahead_distance: Расстояние до целевой точки.
        :return: Точка на пути [x, y].
        """
        path = self.path
        for i in reversed(range(len(path) - 1)):
            segment_start = np.array(path[i])
            segment_end = np.array(path[i + 1])
            segment_vector = segment_end - segment_start
            segment_length = np.linalg.norm(segment_vector)
            
            # Вектор от текущей позиции до начала сегмента
            print("segment_start", segment_start)
            to_segment_start = current_position - segment_start
            
            # Проекция текущей позиции на сегмент
            projection = np.dot(to_segment_start, segment_vector) / segment_length
            
            if projection < 0:
                # Текущая позиция перед началом сегмента
                closest_point = segment_start
            elif projection > segment_length:
                # Текущая позиция за концом сегмента
                closest_point = segment_end
            else:
                # Текущая позиция внутри сегмента
                closest_point = segment_start + (segment_vector / segment_length) * projection
            
            # Расстояние до ближайшей точки на сегменте
            distance_to_closest = np.linalg.norm(current_position - closest_point)
            
            if distance_to_closest <= lookahead_distance:
                # Ищем точку на сегменте, которая находится на расстоянии lookahead_distance
                remaining_distance = lookahead_distance - distance_to_closest
                lookahead_point = closest_point + (segment_vector / segment_length) * remaining_distance
                return lookahead_point
        
        # Если не нашли точку, возвращаем последнюю точку пути
        if np.linalg.norm(current_position - path[-1]) < 0.3:
            self.end = True
        return np.array(path[-1])

    def pure_pursuit_controller(self, current_position, current_orientation_euler, linear_velocity=0.3, lookahead_distance=0.35):
        """
        Pure Pursuit Controller.
        :param path: Массив точек пути [[x1, y1], [x2, y2], ...].
        :param current_position: Текущая позиция робота [x, y].
        :param current_heading: Текущий угол робота (в радианах).
        :param lookahead_distance: Расстояние до целевой точки.
        :param linear_velocity: Желаемая линейная скорость.
        :return: Линейная и угловая скорость.
        """
        print("current_position: ", current_position)
        print("lookahead_distance ", lookahead_distance)
        if np.linalg.norm(self.target_position[0:2] - current_position[0:2]) < 1 or np.linalg.norm(self.path[-1][0:2] - current_position[0:2]) < max(0.2,1/self.ratio):
            self.end = True
        current_heading = 0
        if current_orientation_euler < 0:
            current_heading = -np.pi - current_orientation_euler
        else:
            current_heading = np.pi - current_orientation_euler
        print("current_heading: ", math.degrees(current_heading), current_heading)
        print("star end ", self.start, self.end)
        if not self.start and not self.end:
            # current_position = np.array(current_position)
            
            # Находим целевую точку (lookahead point)
            lookahead_point = self.get_lookahead_point(current_position, lookahead_distance)
            print("lookahead_point: ", lookahead_point)
            # Вектор от робота к целевой точке
            to_target = lookahead_point - current_position
            print("to_target: ", to_target)
            # Угол до целевой точки
            target_angle = np.arctan2(to_target[1], to_target[0])
            print("target_angle: ", math.degrees(target_angle), target_angle)
            # Ошибка по углу (разница между текущим направлением и направлением на цель)
            alpha = self.normalize_angle(target_angle - current_heading)
            print("alpha: ", math.degrees(alpha), alpha)
            # Кривизна дуги
            curvature = 2 * np.sin(alpha) / lookahead_distance
            # Угловая скорость
            angular_velocity = curvature * linear_velocity
            print("angular_velocity: ", angular_velocity)
            # return 0, 0.8
            max_angular_velocity = math.pi*0.4
            angular_velocity = np.clip(angular_velocity, -max_angular_velocity, max_angular_velocity)

            return linear_velocity*(max_angular_velocity-angular_velocity)/max_angular_velocity, angular_velocity
        else:
            if self.ferst_ep:
                self.ferst_ep = False
                return 0, 0
            angular_velocity = 1
            linear_velocity = 0
            true_angle = 0
            nx = np.array([-1,0])
            ny = np.array([0,1])
            
            
            if self.start:
                to_goal_vec = self.path[1] - current_position[0:2]
            elif self.end:
                to_goal_vec = self.target_position[0:2] - current_position[0:2]
                print("target pos ", self.target_position, to_goal_vec)

            quadrant = self.get_quadrant(nx, ny, to_goal_vec)
            cos_angle = np.dot(to_goal_vec, nx) / np.linalg.norm(to_goal_vec) / np.linalg.norm(nx)
            print("need angle ", math.degrees(quadrant*np.arccos(cos_angle)))
            true_angle = quadrant*np.arccos(cos_angle)
            normalize_angle = lambda angle: angle if angle >= 0 else angle + 2 * math.pi
            angle_1 = normalize_angle(true_angle)
            angle_1 = angle_1 + np.pi
    
            # Нормализуем угол, если он больше 360
            if angle_1 >= 2*np.pi:
                angle_1 -= 2*np.pi
            print("angle_1 ", math.degrees(angle_1))
            if angle_1 == 2*np.pi:
                angle_1 = 0
            angle_2 = 2*np.pi - normalize_angle(current_heading)
            # angle_2 = 2 * math.pi - angle_2
            print("angle_2 ", math.degrees(angle_2))
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

            angular_velocity *= sign #np.sign(true_angle)

            if self.start and abs(angle_1-angle_2) < np.pi/80:
                print("end turning tu path start with angle", math.degrees(abs(true_angle)))
                self.start = False

            return linear_velocity, angular_velocity
    
    def get_quadrant(self, nx, ny, vector):
        LR = vector[0]*nx[1] - vector[1]*nx[0]
        mult = 1
        if LR < 0:
            mult = -1
        return mult

from pprint import pprint 

from embed_nn import SceneEmbeddingNetwork
import torch.optim as optim

class Graph_manager():
    def __init__(self):
        pass

    def init_embedding_nn(self):
        device  = self.device
        self.embedding_net = SceneEmbeddingNetwork(object_feature_dim=518).to(device)
        self.embedding_optimizer = optim.Adam(self.embedding_net.parameters(), lr=0.001)


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
