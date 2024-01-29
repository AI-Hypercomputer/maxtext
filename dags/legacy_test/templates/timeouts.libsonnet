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

{
  local timeouts = self,

  // Conversions in secondss
  one_minute: 60,
  one_hour: 3600,
  ten_hours: 10 * timeouts.one_hour,

  Minutes(x):: { timeout: x * timeouts.one_minute },
  Hours(x):: { timeout: x * timeouts.one_hour },
}
