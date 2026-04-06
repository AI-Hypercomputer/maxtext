<!--
 # Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 -->

# src/maxtext

The contents of `src/MaxText` have moved to `src/maxtext` as part of a larger 
[restructuring effort in MaxText](https://github.com/AI-Hypercomputer/maxtext/blob/2790ed289c0c4cb704645d5d2ab91da26711b891/RESTRUCTURE.md).
This directory only contains shim files to temporarily support legacy commands like `python3 -m MaxText.train ...`.
These legacy commands are now deprecated and will be removed soon. Please migrate your existing commands and avoid using
legacy ones. The new command locations can be found at:

* `MaxText.decode` → `maxtext.inference.decode`
* `MaxText.distillation.train_distill` → `maxtext.trainers.post_train.distillation.train_distill`
* `MaxText.maxengine_server` → `maxtext.inference.maxengine.maxengine_server`
* `MaxText.rl.evaluate_rl` → `maxtext.trainers.post_train.rl.evaluate_rl`
* `MaxText.rl.train_rl` → `maxtext.trainers.post_train.rl.train_rl`
* `MaxText.sft.sft_trainer` → `maxtext.trainers.post_train.sft.train_sft`
* `MaxText.train` → `maxtext.trainers.pre_train.train`
* `MaxText.train_compile` → `maxtext.trainers.pre_train.train_compile`
* `MaxText.train_tokenizer` → `maxtext.trainers.tokenizer.train_tokenizer`
