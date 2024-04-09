"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

"""Utilities for Tensorboard in Vertex AI."""

import os

import jax

import max_logging
import max_utils

from cloud_accelerator_diagnostics import tensorboard
from cloud_accelerator_diagnostics import uploader


class VertexTensorboardManager:
  """Class to create Vertex AI Tensorboard and upload logs to that instance."""

  def __init__(self):
    self.uploader_flag = False

  def __del__(self):
    """Stop the Tensorboard uploader thread."""
    if self.uploader_flag:
      uploader.stop_upload_to_tensorboard()

  def setup(self):
    """Creates Tensorboard instance and Experiment in Vertex AI.
    
    Returns:
      URL to view Vertex Tensorboard created in Google Cloud Project.
    """
    max_logging.log("Setting up Tensorboard and Experiment in Vertex AI.")

    vertex_tensorboard_project = os.environ.get("TENSORBOARD_PROJECT")
    vertex_tensorboard_region = os.environ.get("TENSORBOARD_REGION")
    if not vertex_tensorboard_project or not vertex_tensorboard_region:
      max_logging.log("Either config.vertex_tensorboard_project or config.vertex_tensorboard_region is not set.")
      return None

    # Create Vertex Tensorboard instance
    vertex_tensorboard_name = os.environ.get("TENSORBOARD_NAME")
    instance_id = tensorboard.create_instance(project=vertex_tensorboard_project,
                                              location=vertex_tensorboard_region,
                                              tensorboard_name=vertex_tensorboard_name)
    # Failed to create Vertex Tensorboard instance
    if instance_id is None:
      return None

    # Create Vertex Experiment
    vertex_experiment_name = os.environ.get("EXPERIMENT_NAME")
    _, tensorboard_url = tensorboard.create_experiment(project=vertex_tensorboard_project,
                                                      location=vertex_tensorboard_region,
                                                      experiment_name=vertex_experiment_name,
                                                      tensorboard_name=vertex_tensorboard_name)
    return tensorboard_url

  def upload_data(self, tensorboard_dir):
    """Starts an uploader to continously monitor and upload data to Vertex Tensorboard.

    Args:
      tensorboard_dir: directory that contains Tensorboard data.
    """
    tensorboard_project = os.environ.get("TENSORBOARD_PROJECT")
    tensorboard_region = os.environ.get("TENSORBOARD_REGION")
    tensorboard_name = os.environ.get("TENSORBOARD_NAME")
    experiment_name = os.environ.get("EXPERIMENT_NAME")

    if not tensorboard_project or not tensorboard_region or not tensorboard_name or not experiment_name:
      max_logging.log("Vertex Tensorboard configurations are not set. Data will not be uploaded to Vertex AI.")
      self.uploader_flag = False

    max_logging.log(f"Data will be uploaded to Vertex Tensorboard instance: {tensorboard_name} "
                    f"and Experiment: {experiment_name} in {tensorboard_region}.")
    uploader.start_upload_to_tensorboard(project=tensorboard_project,
                                        location=tensorboard_region,
                                        experiment_name=experiment_name,
                                        tensorboard_name=tensorboard_name,
                                        logdir=tensorboard_dir)
    self.uploader_flag = True

  def configure_vertex_tensorboard(self, config):
    """Creates Vertex Tensorboard and start thread to upload data to Vertex Tensorboard."""
    if jax.process_index()==0:
      if not os.environ.get("TENSORBOARD_PROJECT"):
        if not config.vertex_tensorboard_project:
          os.environ["TENSORBOARD_PROJECT"] = max_utils.get_project()
        else:
          os.environ["TENSORBOARD_PROJECT"] = config.vertex_tensorboard_project

      if not os.environ.get("TENSORBOARD_REGION"):
        os.environ["TENSORBOARD_REGION"] = config.vertex_tensorboard_region

      if not os.environ.get("TENSORBOARD_NAME"):
        vertex_tensorboard_project = os.environ.get("TENSORBOARD_PROJECT")
        os.environ["TENSORBOARD_NAME"] = f"{vertex_tensorboard_project}-tb-instance"

      if not os.environ.get("EXPERIMENT_NAME"):
        os.environ["EXPERIMENT_NAME"] = config.run_name

      if config.use_vertex_tensorboard: # running MaxText on GCE
        tensorboard_url = self.setup()
        if tensorboard_url is None:
          raise ValueError("Unable to create Tensorboard and Experiment in Vertex AI.")
        max_logging.log(f"View your Vertex AI Tensorboard at: {tensorboard_url}")
        self.upload_data(config.tensorboard_dir)
      elif os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"): # running MaxText via XPK
        self.upload_data(config.tensorboard_dir)
