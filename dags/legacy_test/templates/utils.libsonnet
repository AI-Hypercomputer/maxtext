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
  // Formats multi-line script string into valid Kubernetes Container args.
  scriptCommand(script):
    [
      '/bin/bash',
      '-c',
      |||
        set -u
        set -e
        set -x

        %s
      ||| % script,
    ],

  // Converts list of command parts (e.g. ['python3', 'script.py']) to a
  // multi-line string.
  toCommandString(list):
    std.join(' \\\n', [std.escapeStringBash(arg) for arg in list]),

  // Return an object of the form {cluster: tests} where cluster is either the
  // cluster preferred for the accelerator used in the test or the default
  // cluster.
  // Args:
  //   tests: array of tests
  //   defaultCluster: default cluster to use if accelerator is not in
  //     clusterAccelerators
  //   acceleratorClusters: object of the form {acceleratorName: cluster}
  splitByCluster(tests, defaultCluster, acceleratorClusters={}):
    local getCluster(test) = (
      if std.objectHas(acceleratorClusters, test.accelerator.name) then
        acceleratorClusters[test.accelerator.name]
      else
        defaultCluster
    );
    local clusters = std.set(std.objectValues(acceleratorClusters) + [defaultCluster]);
    {
      [cluster]: [test for test in tests if getCluster(test) == cluster]
      for cluster in clusters
    },

  // Returns an object of the form {"cluster/gen/test_name.yaml": cronJobYaml}.
  // Skips tests with schedule == null.
  // Args:
  //   tests: object of the form {"test_name": test}
  //   defaultCluster: default cluster to use if accelerator is not in
  //     clusterAccelerators
  //   acceleratorClusters: object of the form {acceleratorName: cluster}
  //
  // Use with jsonnet -S -m output_dir/ ...
  cronJobOutput(tests, defaultCluster, acceleratorClusters={}):
    local clusterTests = self.splitByCluster(
      std.filter(function(test) test.schedule != null, std.objectValues(tests)), defaultCluster, acceleratorClusters
    );
    std.foldl(
      function(result, clusterName) result + {
        ['%s/gen/%s.yaml' % [clusterName, test.testName]]: std.manifestYamlDoc(test.cronJob)
        for test in clusterTests[clusterName]
      },
      std.objectFields(clusterTests),
      {},
    ),
}
