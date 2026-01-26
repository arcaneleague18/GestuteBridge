import os
import string

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(ROOT_DIR, "MP_Data")
DEFAULT_ACTIONS_PATH = os.path.join(ROOT_DIR, "actions.txt")

def get_actions(data_path=None):
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    if os.path.exists(data_path):
        actions = sorted(
            name for name in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, name))
        )
        if actions:
            return actions
    return []

def save_actions(actions, file_path=None):
    if file_path is None:
        file_path = DEFAULT_ACTIONS_PATH
    with open(file_path, 'w', encoding='utf-8') as f:
        for action in actions:
            f.write(f'{action}\n')

def load_actions(file_path=None, data_path=None):
    if file_path is None:
        file_path = DEFAULT_ACTIONS_PATH
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            actions = [line.strip() for line in f if line.strip()]
        if actions:
            return actions
    return get_actions(data_path=data_path)
