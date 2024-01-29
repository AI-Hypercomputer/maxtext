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

local timeouts = import 'timeouts.libsonnet';

{
  Functional:: {
    mode: 'func',
    timeout: timeouts.one_hour,
    // Run at midnight PST daily
    schedule: '0 8 * * *',
    tpuSettings+: {
      preemptible: true,
    },
  },
  Convergence:: {
    mode: 'conv',
    timeout: timeouts.ten_hours,
    // Run at 22:00 PST on Sunday and Wednesday
    schedule: '0 6 * * 0,5',
  },
  Experimental:: {
    schedule: null,
  },
  PreemptibleTpu:: {
    tpuSettings+: {
      preemptible: true,
    },
  },
  Suspended:: {
    cronJob+:: {
      spec+: {
        suspend: true,
      },
    },
  },
  Unsuspended:: {
    cronJob+:: {
      spec+: {
        suspend: false,
      },
    },
  },
}
