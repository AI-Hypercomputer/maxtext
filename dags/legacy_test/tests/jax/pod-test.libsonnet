// Copyright 2021 Google LLC
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

local common = import 'common.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local podTest = self.podTest,
  podTest:: common.JaxTest + mixins.Functional {
    modelName: 'pod-%s-%s' % [self.jaxlibVersion, self.tpuSettings.softwareVersion],

    // Never trigger the run (Feb 31st does not exist)
    schedule: '0 0 31 2 *',

    setup: |||
      %(installLocalJax)s
      %(maybeBuildJaxlib)s
      %(printDiagnostics)s
    ||| % self.scriptConfig,
    runTest: |||
      # Very basic smoke test
      python3 -c "import jax; assert jax.device_count() == 32, jax.device_count()"

      # Slightly-less-basic smoke test
      python3 <<'EOF'
      import jax
      import jax.numpy as jnp
      x = jnp.ones(jax.local_device_count())
      y = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(x)
      print(y)
      assert jnp.array_equal(y, x * jax.device_count())
      EOF
    ||| % self.scriptConfig,
  },

  local v2_32 = self.v2_32,
  v2_32:: {
    accelerator: tpus.v2_32,
  },

  configs: [
    podTest + common.jaxlibHead + common.tpuVmBaseImage + v2_32,
    podTest + common.jaxlibLatest + common.tpuVmBaseImage + v2_32,
  ],
}
