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
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.main_config import MainConfig

def generate_obstacles_json(scene_controller=None):
    """Генерирует JSON-файл с информацией о препятствиях и цели, если файл еще не существует."""
    from scene_manager import Scene_controller
    scene_controller = Scene_controller()
    # Инициализация кэша при первом вызове
    obstacles_prop, obstacles, _ = scene_controller.get_obstacles(key=-1)
    from itertools import combinations

    arr = range(len(obstacles))
    results = []

    for r in range(1, len(arr) + 1):
        for combo in combinations(arr, r):
            combined = ''.join(str(x) for x in combo)
            results.append(combined)
    all_configs = results
    _, goal_positions , _ = scene_controller.get_target_position()

    obstacles_cache = {}
    
    # Загружаем все возможные конфигурации (8x8 = 64)
    for config_key in all_configs:
        positions_for_obstacles = [int(ch) for ch in config_key]
        scene_controller.generate_positions_for_contole_module(key=len(positions_for_obstacles), positions_for_obstacles=positions_for_obstacles)
        obstacles_prop, obstacles, _ = scene_controller.get_obstacles(key=config_key)     
        for gp in range(len(goal_positions)):
            goal_position = goal_positions[gp]
            key_str = f"{gp}_{config_key}"
            output_file = os.path.join("/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/Aloha/gjst", f"obstacles_and_goal_{key_str}.json")
            
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r') as f:
                        obstacles_cache[(config_key)] = json.load(f)
                    print(f"Loaded obstacles for configuration ({gp}_{config_key})")
                except Exception as e:
                    print(f"Error loading obstacles file {output_file}: {e}")
            else:
                # Если файл не существует, создаем новую конфигурацию
                objects = []
                # Добавляем препятствия
                for idx, obstacle_prop in enumerate(obstacles_prop):
                    position = obstacle_prop["position"]
                    bbox_center = [position[0], position[1], 0.35]
                    bbox_extent = [0.6, 0.6, 0.7]
                    name = f"{obstacle_prop['shape']}"
                    description = "obstacle" if name != "table" else "soft obstacle"
                    
                    objects.append({
                        "id": idx,
                        "bbox_extent": bbox_extent,
                        "bbox_center": bbox_center,
                        "name": name,
                        "description": description
                    })
                
                # Добавляем цель
                bbox_center = [float(goal_position[0]), float(goal_position[1]), 0.6]
                bbox_extent = [0.2, 0.2, 0.7]
                objects.append({
                    "id": len(obstacles_prop),
                    "bbox_extent": bbox_extent,
                    "bbox_center": bbox_center,
                    "name": "bowl (0.20)",
                    "description": "goal"
                })
                
                # Сохраняем в кэш и в файл
                obstacles_cache[(config_key)] = objects
                try:
                    with open(output_file, 'w') as f:
                        json.dump(objects, f, indent=4)
                    print(f"Saved obstacles for configuration ({gp}_{config_key})")
                except Exception as e:
                    print(f"Error saving obstacles file {output_file}: {e}")

    # # Преобразуем key в кортеж для использования в кэше
    # key_tuple = tuple(key)
    
    # # Возвращаем конфигурацию из кэша
    # return self.obstacles_cache[key_tuple]

if __name__ == "__main__":
    generate_obstacles_json()