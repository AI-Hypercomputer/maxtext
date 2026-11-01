# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LingBot-World model family (robbyant/lingbot-world-fast): causal camera-controlled Wan world model.

Files:
  transformer_lingbot.py  `LingBotWorldModel`, the block-causal DiT with camera injection and a KV cache.
  camera_lingbot.py       Action sequence (camera trajectory) -> Plücker conditioning (host-side numpy).
  pipeline_lingbot.py     `LingBotWorldPipeline` (Wan pipeline reuse + causal few-step sampler), for inference/sample.py.
  checkpoint_lingbot.py   Configs / key renaming of the original-layout `.pth` components, for to_maxtext_diffusers.py.
  configs/                Model YAMLs.
"""
