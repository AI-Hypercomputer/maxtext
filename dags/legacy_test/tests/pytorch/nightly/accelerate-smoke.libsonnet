local experimental = import '../experimental.libsonnet';
local common = import 'common.libsonnet';
local tpus = import 'templates/tpus.libsonnet';

{
  local accelerate = self.accelerate,
  accelerate:: common.PyTorchTest + common.Functional {
    modelName: 'accelerate',
    mode: 'smoke',
    command: [
      'accelerate',
      'test',
    ],
  },
  local pjrt = self.pjrt,
  pjrt:: common.PyTorchTpuVmMixin + common.Accelerate,

  local v2_8 = self.v2_8,
  v2_8:: {
    accelerator: tpus.v2_8,
  },
  local v4_8 = self.v4_8,
  v4_8:: {
    accelerator: tpus.v4_8,
  },

  configs: [
    accelerate + v2_8 + pjrt,
    accelerate + v4_8 + pjrt,
  ],
}
