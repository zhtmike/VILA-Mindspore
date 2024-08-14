# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

from .configuration_siglip import SiglipVisionConfig
from .modeling_siglip import SiglipVisionModel
from .image_processing_siglip import SiglipImageProcessor

if TYPE_CHECKING:
    from .configuration_siglip import (
        SIGLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SiglipConfig,
        SiglipTextConfig,
        SiglipVisionConfig,
    )
    from .processing_siglip import SiglipProcessor
    from .tokenization_siglip import SiglipTokenizer
    from .image_processing_siglip import SiglipImageProcessor

    from .modeling_siglip import (
        SiglipModel,
        SiglipPreTrainedModel,
        SiglipTextModel,
        SiglipVisionModel,
    )
