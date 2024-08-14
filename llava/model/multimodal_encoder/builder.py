# This file is modified from https://github.com/haotian-liu/LLaVA/

import os
from mindnlp.transformers import AutoConfig, PretrainedConfig, PreTrainedModel
from .clip_encoder import CLIPVisionTower
from .siglip_encoder import SiglipVisionTower

def build_vision_tower(
    model_name_or_path: str, config: PretrainedConfig
) -> PreTrainedModel:
    ## skip vision tower instantiation
    if model_name_or_path is None:
        return None

    vision_tower_arch = None
    if config.resume_path and "radio" not in model_name_or_path:
        assert os.path.exists(
            model_name_or_path
        ), f"Resume vision tower path {model_name_or_path} does not exist!"
        vision_tower_cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        vision_tower_arch = vision_tower_cfg.architectures[0].lower()
    vision_tower_name = (
        vision_tower_arch if vision_tower_arch is not None else model_name_or_path
    )

    if "radio" in vision_tower_name:
        raise NotImplementedError("Not Implemented")
    elif "clip" in vision_tower_name:
        vision_tower = CLIPVisionTower(model_name_or_path, config)
    elif "siglip" in vision_tower_name:
        vision_tower = SiglipVisionTower(model_name_or_path, config)
    else:
        raise ValueError(f"Unknown vision tower: {model_name_or_path}")

    config.mm_hidden_size = vision_tower.config.hidden_size
    return vision_tower
