from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, Optional, Tuple, Union
import numpy as np

from pathlib import Path
d = Path().resolve()#.parent
general_path = str(d) + "/standalone_examples/Aloha_graph/Aloha"

@dataclass
class MainConfig:
    tuning: bool = False
    headless: bool = True
    reward_mode: int = 1
    cup_usd_path: str = general_path + "/assets/objects/bowl.usd"
    table_usd_path: str = general_path + "/assets/scenes/scenes_sber_kitchen_for_BBQ/table/table.usd"
    jetbot_asset_path: str = general_path + "/assets/aloha/aloha.usd"
    room_usd_path: str = general_path + "/assets/scenes/scenes_sber_kitchen_for_BBQ/sber_kitchen_ft.usd"
    # room_usd_path: str = general_path + "/assets/scenes/scenes_sber_kitchen_for_BBQ/kitchen_wow.usd"
    goal_image_path: str = general_path + '/img/goal.png'
    goalb_image_path: str = general_path + '/img/goal_bowl.png'
    train_log_dir: str = general_path + "/models/SAC"
    log_path: str = general_path + "/logs/log.txt"
    camera_usd_local_path = general_path + '/assets/aloha/aloloha_v03_cameras.usd'
    camera_image_saved_path = general_path + "/img/"
    loss_path:str = general_path + '/loss.pt'
    load_emb_nn:str = general_path + "/emb_models/scene_embedding_epoch_465000.pth"
    training_mode:int = 0
    scene_file_test: str = general_path + "/scene/scene_test/"
    scene_file_train: str = general_path + "/scene/scene_train/"
    # load_policy: str = general_path + "/models/SAC/nav_with_graph_ahahah_work_95000_steps.zip"#fool
    load_policy: str = general_path + "/models/SAC/tired_1005000_steps.zip"
    eval:bool = False
    eval_radius:float = 2.8
    eval_angle:float = 0
    eval_print:bool = False
    eval_log_path: str = general_path + "/logs/eval_log_fora1.txt"
    segmentation:bool = False
    memory:bool = False
    graph:bool = True
    control:bool = True
    imitation:bool = False
    test_level:int = 1
    name: str = "bye_bye_chumba"
