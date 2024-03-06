// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

local accelerate = import 'accelerate-smoke.libsonnet';
local ci = import 'ci.libsonnet';
local hfBert = import 'hf-bert.libsonnet';
local huggingfaceDiffusers = import 'hf-diffusers.libsonnet';
local huggingfaceGlue = import 'hf-glue.libsonnet';
local huggingfaceGPT2 = import 'hf-llm.libsonnet';
local llama2 = import 'llama2-model.libsonnet';
local mnist = import 'mnist.libsonnet';
local resnet50_mp = import 'resnet50-mp.libsonnet';
local stableDif = import 'sd-model.libsonnet';

// Add new models here
std.flattenArrays([
  accelerate.configs,
  ci.configs,
  hfBert.configs,
  huggingfaceDiffusers.configs,
  huggingfaceGlue.configs,
  huggingfaceGPT2.configs,
  mnist.configs,
  resnet50_mp.configs,
  stableDif.configs,
  llama2.configs,
])
