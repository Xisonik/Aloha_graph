import os
import json
from pathlib import Path


ROOT_PATH = Path(__file__).parents[1]
CONFIG_PATH = (ROOT_PATH / "config.json").resolve()


def get_asset_path(key):
    with open(CONFIG_PATH, 'r') as config_file:
        config = json.load(config_file)

    return Path(config[key]).as_posix()