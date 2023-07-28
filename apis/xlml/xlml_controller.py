# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The controller to run XL ML tests."""

from airflow.utils import task_group
from apis.xlml import task


# TODO(wcromar): Can I define this in terms of BaseTask?
def run(job_task: task.TPUTask) -> task_group.TaskGroup:
  """Run a test job.

  Args:
    job_task: Tasks for a test job.

  Returns:
    A task group with chained up four tasks: provision, run_model, post_process
    and clean_up.
  """
  with task_group.TaskGroup(
      group_id=job_task.task_test_config.benchmark_id, prefix_group_id=True
  ) as tg:
    provision, tpu_name, ssh_keys = job_task.provision()
    run_model = job_task.run_model(tpu_name, ssh_keys)
    post_process = job_task.post_process(tpu_name, ssh_keys)
    clean_up = job_task.clean_up(tpu_name)

    provision >> run_model >> post_process >> clean_up

  return tg
