from google.cloud import monitoring_v3
from google.cloud import compute_v1
from google.api import metric_pb2
import time
import os

PROJECT_ID="cloud-tpu-multipod-dev"
ZONE = "us-central2-b"

def create_custom_metric(metric_name, description):
    metric_name = "checkpoint_init"
    project_name = f"projects/{PROJECT_ID}"

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

    print(response)


def write_time_series_step(metric_name, step, status):
  """
  Emits a data point when a training step is STARTED and COMPLETED.
  Args:
    metric_name: name of the metric
    step: training step
    status: STARTED if the training step is started, else COMPLETED
  """
  client = get_metrics_service_client()
  project_name = f"projects/{PROJECT_ID}"

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
      "Emitting status = ",
      status,
      " for step = ",
      step,
      " at: ",
      event_time,
  )

  instance_id = get_instance_id()

  series = monitoring_v3.types.TimeSeries()
  series.metric.type = "custom.googleapis.com/" + metric_name
  series.resource.type = "gce_instance"
  series.resource.labels["instance_id"] = str(instance_id)
  series.resource.labels["zone"] = ZONE
  series.metric.labels["step_num"] = str(step)
  series.metric.labels["worker"] = os.uname().nodename
  series.metric.labels["status"] = status
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
  print(
      "Time series added for step",
      step,
      "and instance_id ",
      instance_id,
      " and zone ",
      ZONE,
  )

def get_instance_id():
  client = get_compute_instances_client()
  instance_name = os.uname().nodename
  instance = client.get(project=PROJECT_ID, zone=ZONE, instance=instance_name)
  return instance.id

def get_compute_instances_client():
  return compute_v1.InstancesClient()

def get_metrics_service_client():
  return monitoring_v3.MetricServiceClient()

if __name__ == "__main__":
  create_custom_metric('checkpointing_init', get_metrics_service_client(), "This is a checkpointing init metric.")