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
  // Base object for Kubernetes volumes.
  VolumeSpec:: {
    local volume = self,

    name: error 'Must set `volumeName`',
    mountPath: error 'Must set `mountPath`',
    claim: error 'Must set `claim`',
    readOnly: error 'Must set `readOnly',

    ContainerMixin:: {
      volumeMounts+: [
        {
          name: volume.name,
          mountPath: volume.mountPath,
          readOnly: volume.readOnly,
        },
      ],
    },
    PodSpecMixin:: {
      volumes+: [
        {
          name: volume.name,
        } + volume.claim,
      ],
      containerMap+: {
        train+: volume.ContainerMixin,
      },
    },
  },

  // Combines a map of VolumeSpec into a single pod spec mixin.
  combinedMixin(volumeMap): std.foldl(
    function(next, rest) rest + next,
    [volumeMap[v].PodSpecMixin for v in std.objectFields(volumeMap) if volumeMap[v] != null],
    {}
  ),

  MemoryVolumeSpec:: self.VolumeSpec {
    local volume = self,

    readOnly: false,
    claim: {
      emptyDir: {
        medium: 'Memory',
      },
    },
  },
  // Volume corresponding to a PersistentVolumeClaim
  PersistentVolumeSpec:: self.VolumeSpec {
    local volume = self,

    readOnly: true,
    claim: {
      persistentVolumeClaim: {
        claimName: volume.name,
      },
    },
  },
}
