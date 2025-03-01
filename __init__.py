# Made by Jim.Wang V1 for ComfyUI
import os
import subprocess
import importlib.util
import sys
import filecmp
import shutil

import __main__

python = sys.executable




from .DynamicTool import DynamicLora,DynamicScene

NODE_CLASS_MAPPINGS = {
    "Dynamic_Lora_Loader":DynamicLora,
    "Dynamic_Scene_Loader":DynamicScene,
}


print('\033[34mHailuo07 DynamicLora Nodes: \033[92mLoaded\033[0m')