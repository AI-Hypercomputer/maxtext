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
  local metrics = self,

  TensorBoardSourceHelper:: {
    local aggregateAssertionsMapToList(map) = [
      {
        tag: tag,
        strategy: strategy,
        assertion: map[tag][strategy],
      }
      for tag in std.objectFields(map)
      for strategy in std.objectFields(map[tag])
    ],

    aggregateAssertionsMap:: {},
    aggregate_assertions: aggregateAssertionsMapToList(self.aggregateAssertionsMap),
  },
  MetricCollectionConfigHelper:: {
    local helper = self,

    sourceMap:: {
      tensorboard: {},
      perfzero: {},
      literals: {},
    },

    sources: [
      { [source]: helper.sourceMap[source] }
      for source in std.objectFields(std.prune(self.sourceMap))
    ],
  },
}
