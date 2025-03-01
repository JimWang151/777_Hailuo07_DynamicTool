

import folder_paths
import comfy.utils
import comfy.sd
import os
import re
import random
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

class DynamicLora:
    def __init__(self):
        self.selected_loras = SelectedLoras()
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "text": ("STRING", {
                                "multiline": True,
                                "default": ""}),
                            }}

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "dynamic_loras"
    CATEGORY = "Hailuo07"

    def dynamic_loras(self, model, clip, text):
        result = (model, clip)
        
        lora_items = self.selected_loras.updated_lora_items_with_text(text)

        if len(lora_items) > 0:
            for item in lora_items:
                result = item.apply_lora(result[0], result[1])
            
        return result
 
# maintains a list of lora objects made from a prompt, preserving loaded loras across changes
class SelectedLoras:
    def __init__(self):
        self.lora_items = []

    # returns a list of loaded loras using text from LoraTextExtractor
    def updated_lora_items_with_text(self, text):
        available_loras = self.available_loras()
        self.update_current_lora_items_with_new_items(self.items_from_lora_text_with_available_loras(text, available_loras))
        
        for item in self.lora_items:
            if item.lora_name not in available_loras:
                raise ValueError(f"Unable to find lora with name '{item.lora_name}'")
            
        return self.lora_items

    def available_loras(self):
        return folder_paths.get_filename_list("loras")
    
    def items_from_lora_text_with_available_loras(self, lora_text, available_loras):
        return LoraItemsParser.parse_lora_items_from_text(lora_text, self.dictionary_with_short_names_for_loras(available_loras))
    
    def dictionary_with_short_names_for_loras(self, available_loras):
        result = {}
        
        for path in available_loras:
            result[os.path.splitext(os.path.basename(path))[0]] = path
        
        return result

    def update_current_lora_items_with_new_items(self, lora_items):
        if self.lora_items != lora_items:
            existing_by_name = dict([(existing_item.lora_name, existing_item) for existing_item in self.lora_items])
            
            for new_item in lora_items:
                new_item.move_resources_from(existing_by_name)
            
            self.lora_items = lora_items

class LoraItemsParser:

    @classmethod
    def parse_lora_items_from_text(cls, lora_text, loras_by_short_names = {}, default_weight=1, weight_separator=":"):
        return cls(lora_text, loras_by_short_names, default_weight, weight_separator).execute()

    def __init__(self, lora_text, loras_by_short_names, default_weight, weight_separator):
        self.lora_text = lora_text
        self.loras_by_short_names = loras_by_short_names
        self.default_weight = default_weight
        self.weight_separator = weight_separator
        self.prefix_trim_re = re.compile("\A<(lora|lyco):")
        self.comment_trim_re = re.compile("\s*#.*\Z")
    
    def execute(self):
        return [LoraItem(elements[0], elements[1], elements[2])
            for line in self.lora_text.splitlines()
            for elements in [self.parse_lora_description(self.description_from_line(line))] if elements[0] is not None]
    
    def parse_lora_description(self, description):
        if description is None:
            return (None,)
        
        lora_name = None
        strength_model = self.default_weight
        strength_clip = None
        
        remaining, sep, strength = description.rpartition(self.weight_separator)
        if sep == self.weight_separator:
            lora_name = remaining
            strength_model = float(strength)
            
            remaining, sep, strength = remaining.rpartition(self.weight_separator)
            if sep == self.weight_separator:
                strength_clip = strength_model
                strength_model = float(strength)
                lora_name = remaining
        else:
            lora_name = description
        
        if strength_clip is None:
            strength_clip = strength_model
        
        return (self.loras_by_short_names.get(lora_name, lora_name), strength_model, strength_clip)

    def description_from_line(self, line):
        result = self.comment_trim_re.sub("", line.strip())
        result = self.prefix_trim_re.sub("", result.removesuffix(">"))
        return result if len(result) > 0 else None
        

class LoraItem:
    def __init__(self, lora_name, strength_model, strength_clip):
        self.lora_name = lora_name
        self.strength_model = strength_model
        self.strength_clip = strength_clip
        self._loaded_lora = None
    
    def __eq__(self, other):
        return self.lora_name == other.lora_name and self.strength_model == other.strength_model and self.strength_clip == other.strength_clip
    
    def get_lora_path(self):
        return folder_paths.get_full_path("loras", self.lora_name)
        
    def move_resources_from(self, lora_items_by_name):
        existing = lora_items_by_name.get(self.lora_name)
        if existing is not None:
            self._loaded_lora = existing._loaded_lora
            existing._loaded_lora = None

    def apply_lora(self, model, clip):
        if self.is_noop:
            return (model, clip)
        
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, self.lora_object, self.strength_model, self.strength_clip)
        return (model_lora, clip_lora)

    @property
    def lora_object(self):
        if self._loaded_lora is None:
            lora_path = self.get_lora_path()
            if lora_path is None:
                raise ValueError(f"Unable to get file path for lora with name '{self.lora_name}'")
            self._loaded_lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        
        return self._loaded_lora

    @property
    def is_noop(self):
        return self.strength_model == 0 and self.strength_clip == 0

class LoraTextExtractor:
    def __init__(self):
        self.lora_spec_re = re.compile("(<(?:lora|lyco):[^>]+>)")
        self.selected_loras = SelectedLoras()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "text": ("STRING", {
                                "multiline": True,
                                "default": ""}),
                            }}

    RETURN_TYPES = ("STRING", "STRING", "LORA_STACK")
    RETURN_NAMES = ("Filtered Text", "Extracted Loras", "Lora Stack")
    FUNCTION = "process_text"
    CATEGORY = "utils"

    def process_text(self, text):
        extracted_loras = "\n".join(self.lora_spec_re.findall(text))
        filtered_text = self.lora_spec_re.sub("", text)

        # the stack format is a list of tuples of full path, model weight, clip weight,
        # e.g. [('styles\\abstract.safetensors', 0.8, 0.8)]
        lora_stack = [(item.get_lora_path(), item.strength_model, item.strength_clip) for item in self.selected_loras.updated_lora_items_with_text(extracted_loras)]
        
        return (filtered_text, extracted_loras, lora_stack)


import os
import random
from PIL import Image

class DynamicScene:
    def __init__(self):
        self.selected_loras = SelectedLoras()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "category": ("STRING", {"multiline": False, "tooltip": "Key category."}),
                "key_words": ("STRING", {"multiline": False, "tooltip": "Specify the key words of individual image"}),
                "flag": ("STRING", {"multiline": False, "tooltip": "Specify the key words of individual image"}),
                "tar_mask": ("STRING", {"multiline": False, "tooltip": "Specify the key words of individual image"}),
            },
        }

    RETURN_TYPES = ("IMAGE","STRING")
    RETURN_NAMES = ("img","mask_desc")
    FUNCTION = "dynamic_scene"
    CATEGORY = "Hailuo07"

    def dynamic_scene(self, category, key_words, flag,tar_mask):

        img = self.load_image(category, key_words, flag)

        # 转换为PyTorch张量 (C, H, W)
        tensor = to_tensor(img)
        tensor=tensor.permute(1,2,0)
        tensor=tensor.unsqueeze(0)
        return (tensor,tar_mask)

    def load_image(self, category: str, key_words: str, flag: str) -> Image.Image:
        """
        加载指定目录下的PNG文件。

        :param category: 目标目录，为空则在当前img文件夹中查找，否则在对应的子目录中查找。
        :param key_words: 文件名前缀，为空时随机选择文件，否则选择以该前缀开头的文件。
        :param flag: 文件名后缀（不包括扩展名），为空时随机选择符合前缀的文件，否则选择符合前缀且以该后缀结尾的文件。
        :return: 加载的Image对象。
        """
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if category:
                data_dir = os.path.join(current_dir, 'img', category)
            else:
                data_dir = os.path.join(current_dir, 'img')

            # 确保目录存在
            if not os.path.exists(data_dir):
                print(f"目录不存在 {data_dir}。")
                img = Image.new('RGBA', (100, 100), (255, 255, 255, 0))
                return img

            # 获取所有PNG文件
            png_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.png')]

            selected_files = []

            # key_words 不为空时筛选以 key_words 开头的文件
            if key_words:
                selected_files = [f for f in png_files if f.startswith(key_words)]
            else:
                selected_files = png_files.copy()  # 如果 key_words 为空，保留所有文件

            # 如果 flag 不为空，进一步筛选以 flag 结尾的文件（不包括扩展名）
            if flag:
                selected_files = [f for f in selected_files if f[:-4].endswith(flag)]

            # 如果没有任何匹配的文件，返回默认图片
            if not selected_files:
                print(f"没有找到符合条件的文件（category={category}, key_words={key_words}, flag={flag}）")
                return self._load_default_image()

            # 随机选择一个文件
            chosen_file = random.choice(selected_files)

            file_path = os.path.join(data_dir, chosen_file)
            # 返回加载的图像
            img=Image.open(file_path).convert("RGBA")
            return img

        except Exception as e:
            print(f"加载图片时出错: {e}")
            img=Image.new('RGBA', (100, 100), (255, 255, 255, 0))
            return img



if __name__ == "__main__":
    image_loader = DynamicScene()
    image = image_loader.load_image("", "", "")
    image.show()
