# This file is modified from https://github.com/haotian-liu/LLaVA/

from abc import abstractmethod

import mindnlp.core.nn as nn
import mindnlp.core.ops as ops

from mindnlp.transformers import AutoConfig, PreTrainedModel
from mindnlp.transformers.image_processing_utils import BaseImageProcessor


class VisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = getattr(args, "mm_vision_select_layer", -2)
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        self.cfg_only = None

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def _maybe_resize_pos_embeds(
        self,
        model: PreTrainedModel,
        image_processor: BaseImageProcessor,
        resolution: int = -1,
        interpolate_mode: str = "linear",
    ):
        if resolution in [model.config.image_size, -1]:
            return
        print(f"Resizing vision model's position embeddings to support higher vision resolution: from {model.config.image_size} to {resolution} ...")
        embeddings = model.vision_model.embeddings
        patch_size = embeddings.patch_size
        num_new_tokens = int((resolution // patch_size) ** 2)

        old_embeddings = embeddings.position_embedding
        if interpolate_mode == "linear":
            ## Step 1: Calculate the corresponding patch ID (pid) in the current resolution (M patches) based on the target resolution (N patches). Formula: pid = pid / N * M
            ## Step 2:  Obtain new embeddings by interpolating between the embeddings of the two nearest calculated patch IDs. Formula: new_embeds = (pid - floor(pid)) * embeds[ceil(pid)] + (ceil(pid) - pid) * embeds[floor(pid)]
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
            new_embeddings = nn.Embedding(
                num_new_tokens,
                old_embedding_dim,
                dtype=old_embeddings.weight.dtype,
            )
            mapped_indices = (
                ops.arange(num_new_tokens)
                / (num_new_tokens - 1)
                * (old_num_tokens - 1)
            )
            floor_indices = ops.clamp(mapped_indices.floor().long(), min=0, max=old_num_tokens - 1)
            ceil_indices = ops.clamp(mapped_indices.ceil().long(), min=0, max=old_num_tokens - 1)
            interpolated_embeds = (mapped_indices - floor_indices)[:, None] * old_embeddings.weight.data[
                ceil_indices, :
            ] + (ceil_indices - mapped_indices)[:, None] * old_embeddings.weight.data[floor_indices, :]
            new_embeddings.weight.data = interpolated_embeds
        else:
            raise NotImplementedError

        new_embeddings.requires_grad_(old_embeddings.weight.requires_grad)
        ## update vision encoder's configurations
        model.config.image_size = resolution
        if hasattr(image_processor, "crop_size"):
            # CLIP vision tower
            image_processor.crop_size = resolution
        else:
            # SIGLIP vision tower
            assert hasattr(image_processor, "size")
            image_processor.size = {"height": resolution, "width": resolution}
        ## TODO define a '_reinitialize' method for VisionTower
        embeddings.position_embedding = new_embeddings
        embeddings.image_size = resolution
        embeddings.num_patches = embeddings.num_positions = num_new_tokens
        embeddings.position_ids = (
            ops.arange(embeddings.num_positions).expand((1, -1))
        )

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return ops.zeros(1, self.hidden_size, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
