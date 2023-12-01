# pylint: disable=unused-argument, no-name-in-module
"""
Cloud Monitoring API v3 Prototype
"""

import subprocess
import sys
from google.cloud import monitoring_v3
from google.cloud import compute_v1
from google.api import metric_pb2
import time
import os

import max_logging

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


def write_time_series_step(metric_name, monitoring_enabled, pyconfig, step=1):
  """
  Writes a time series object for a specified custom metric

  Args:
    metric_name
    monitoring_enabled
    step
  """

  zone = pyconfig.config.cloud_zone
  project_id = get_project()

  if not monitoring_enabled:
    return []

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
  max_logging.log(
      f"Emitting metric {metric_name} for step = {step} at: {event_time}")

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

  client.create_time_series(name=project_name, time_series=[series])
  dashboard_link = pyconfig.config.cloud_monitoring_dashboard+project_name
  max_logging.log(
      f"Time series added for step {step} and instance_id {instance_id} and zone {zone}\
        \n View dashboards or use metrics: {dashboard_link}")
  return [series]

def get_time_series_step_data(metric_name):
  """
  Retrieves time series data

  Args:
    metric_name
  """
  project_id = get_project()
  project_name = f"projects/{project_id}"
  instance_name = os.uname().nodename

  mql = """
  fetch gce_instance
  | metric 'custom.googleapis.com/{metric_name}'
  | filter (metric.worker == '{worker_id}')
  | every 1m
  | within -1d, 1d # one day, starting 1 day ago
  """

  client = get_query_service_client()
  request = monitoring_v3.QueryTimeSeriesRequest({
      "name": project_name,
      "query": mql.format(
          metric_name=metric_name, worker_id=instance_name
      ),
  })

  result = client.query_time_series(request)
  return result.time_series_data


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

def get_query_service_client():
  """
  Fetches cloud monitoring query service client
  """
  return monitoring_v3.QueryServiceClient()
