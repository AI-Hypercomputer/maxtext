# pylint: disable=unused-argument, no-name-in-module
"""
Cloud Monitoring API v3 Prototype
"""

import subprocess
import sys
from google.cloud import monitoring_v3
from google.cloud import compute_v1
from google.api import metric_pb2
import requests
import time
import os

def get_metadata(project_id, zone, instance_id):
  """
  Fetches metadata

  Args:
    project_id
    zone
    instance_id
  
  Returns:
    metadata as json
  """
  r = requests.get(url="https://compute.googleapis.com/compute/v1/projects/\
                   {project_id}/zones/{zone}/instances/{instance_id}")
  metadata = r.json()
  return metadata

def create_custom_metric(metric_name, description):
  """
  Creates a custom metric

  Args:
    metric_name
    description
  
  Returns:
    Response from create request
  """
  project_id = get_project()
  project_name = f"projects/{project_id}"

  client = monitoring_v3.MetricServiceClient()

  descriptor = metric_pb2.MetricDescriptor()
  descriptor.type = "custom.googleapis.com/" + metric_name
  descriptor.metric_kind = metric_pb2.MetricDescriptor.MetricKind.GAUGE
  descriptor.value_type = metric_pb2.MetricDescriptor.ValueType.DOUBLE
  descriptor.description = description

  request = monitoring_v3.CreateMetricDescriptorRequest(
      name=project_name,
      metric_descriptor=descriptor
  )

  response = client.create_metric_descriptor(request=request)

  return response


def write_time_series_step(metric_name, monitoring_enabled, step=1):
  """
  Writes a time series object for a specified custom metric

  Args:
    metric_name
    monitoring_enabled
    step
  """

  zone = get_zone()
  project_id = get_project()

  if not monitoring_enabled:
    return

  client = get_metrics_service_client()
  project_name = f"projects/{project_id}"

  seconds_since_epoch_utc = time.time()
  nanos_since_epoch_utc = int(
      (seconds_since_epoch_utc - int(seconds_since_epoch_utc)) * 10**9
  )
  interval = monitoring_v3.types.TimeInterval(
      {
          "end_time": {
              "seconds": int(seconds_since_epoch_utc),
              "nanos": nanos_since_epoch_utc,
          }
      }
  )

  event_time = time.strftime(
      "%d %b %Y %H:%M:%S UTC", time.gmtime(seconds_since_epoch_utc)
  )
  print(
      "Emitting metric ",
      metric_name,
      " for step = ",
      step,
      " at: ",
      event_time,
  )

  instance_id = get_instance_id(project_id, zone)

  series = monitoring_v3.types.TimeSeries()
  series.metric.type = "custom.googleapis.com/" + metric_name
  series.resource.type = "gce_instance"
  series.resource.labels["instance_id"] = str(instance_id)
  series.resource.labels["zone"] = zone
  series.metric.labels["step_num"] = str(step)
  series.metric.labels["worker"] = os.uname().nodename
  series.metric.labels["event_time"] = event_time
  series.points = [
      monitoring_v3.types.Point(
          interval=interval,
          value=monitoring_v3.types.TypedValue(
              double_value=step
          ),
      )
  ]

  client.create_time_series(name=project_name, time_series=[series], metadata=get_metadata(project_id, zone, instance_id))
  print(
      "Time series added for step",
      step,
      "and instance_id ",
      instance_id,
      " and zone ",
      zone,
  )

def get_instance_id(project_id, zone):
  """
  Fetches instance id of a node

  Args:
    project_id
    zone
  """
  client = get_compute_instances_client()
  instance_name = os.uname().nodename
  instance = client.get(project=project_id, zone=zone, instance=instance_name)
  return instance.id

def get_project():
  """
  Fetches id of project in use
  """
  completed_command = subprocess.run(["gcloud", "config", "get", "project"], check=True, capture_output=True)
  project_outputs = completed_command.stdout.decode().strip().split('\n')
  if len(project_outputs) < 1 or project_outputs[-1]=='':
    sys.exit("You must specify the project in the PROJECT flag or set it with 'gcloud config set project <project>'")
  return project_outputs[-1]

def get_zone():
  """
  Fetches zone in use
  """
  subprocess.run("gcloud config set compute/zone us-central2-b")
  completed_command = subprocess.run(["gcloud", "config", "get", "compute/zone"], check=True, capture_output=True)
  zone_outputs = completed_command.stdout.decode().strip().split('\n')
  if len(zone_outputs) < 1 or zone_outputs[-1]=='':
    sys.exit("You must specify the zone in the ZONE flag or set it with 'gcloud config set compute/zone <zone>'")
  return zone_outputs[-1]

def get_compute_instances_client():
  """
  Fetches cloud compute instances client
  """
  return compute_v1.InstancesClient()

def get_metrics_service_client():
  """
  Fetches cloud monitoring API client
  """
  return monitoring_v3.MetricServiceClient()
