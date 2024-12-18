import datetime
import os
import time

import max_logging
import requests  # type: ignore[pyi-error]
import jax

from google.api import metric_pb2, monitored_resource_pb2
from google.api_core.exceptions import GoogleAPIError
from google.cloud import monitoring_v3
from urllib3.util.retry import Retry


_METADATA_SERVER_URL = "http://metadata.google.internal/computeMetadata/v1/"
_METADATA_HEADERS = {"Metadata-Flavor": "Google"}


def report_heartbeat_thread(stop_event):
  timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%f")
  zone = get_node_zone()
  workload_id = os.getenv("JOB_NAME", f"maxtext-llama2-tpu-{timestamp}")
  local_rank = os.getenv("LOCAL_RANK", "0")
  global_rank = jax.process_index()
  project_id = get_gcp_project_id()
  while not stop_event.is_set():
    report_heartbeat(workload_id, local_rank, str(global_rank), project_id, zone)
    time.sleep(5)


def report_heartbeat(workload_id: str, local_rank: str, global_rank: str, project_id: str, zone: str):
  try:
    now = time.time()
    seconds = int(now)
    nanos = int((now - seconds) * 10**9)

    # Create a TimeSeries object for the heartbeat metric
    series = monitoring_v3.TimeSeries(
        metric=metric_pb2.Metric(
            type="compute.googleapis.com/workload_process/heartbeat",
            labels={
                # TODO : "gpu_index" for now due to legacy reasons. To be renamed to "local_rank" when schema ready
                "gpu_index": local_rank,
                "instance_id": _get_gcp_metadata(category="instance", attribute="id"),
            },
        ),
        resource=monitored_resource_pb2.MonitoredResource(
            type="compute.googleapis.com/WorkloadProcess",
            labels={
                "project_id": project_id,
                "location": zone,
                "workload_id": workload_id,
                "replica_id": "0",
                "process_id": global_rank,
            },
        ),
        points=[
            monitoring_v3.Point(
                interval=monitoring_v3.TimeInterval(end_time={"seconds": seconds, "nanos": nanos}),
                value=monitoring_v3.TypedValue(bool_value=True),
            ),
        ],
    )

    # Send data to Google Cloud Monitoring
    client = monitoring_v3.MetricServiceClient()
    client.create_time_series(
        request={"name": f"projects/{project_id}", "time_series": [series]},
        timeout=30,
    )
    max_logging.log("Heartbeat metric successfully sent to GCP.")
  except GoogleAPIError as e:
    max_logging.log(f"Failed to send heartbeat to GCP: {e}")
  except Exception as e:
    max_logging.log(f"Unexpected error while sending heartbeat to GCP: {e}")


def _get_gcp_metadata(category, attribute, timeout=5, retries=3):
  """
  Fetch the specified attribute from GCP metadata server.

  Args:
    category (str): The high-level metadata category (ex: 'instance', 'project').
    attribute (str): The attribute to fetch under this category (ex: 'id', 'zone').
    timeout (int): Timeout for the request in seconds.
    retries (int): Number of retry attempts for transient failures.

  Returns:
    str: The metadata value as a string, or None if the request fails.
  """
  target_url = f"{_METADATA_SERVER_URL}{category}/{attribute}"

  session = requests.Session()
  retry_strategy = Retry(
      total=retries,
      backoff_factor=0.5,
      # Retry on the following status codes
      status_forcelist=[429, 500, 502, 503, 504],
  )
  adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
  session.mount("http://", adapter)

  try:
    response = session.get(target_url, headers=_METADATA_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text
  except requests.exceptions.RequestException as e:
    max_logging.log(f"Failed to retrieve metadata for {category}/{attribute}: {e}")
    return None


def get_gcp_project_id():
  """Returns the project id of the current GCP project."""
  return _get_gcp_metadata("project", "project-id")


def get_node_zone():
  """Returns the zone of the GCE instance."""
  zone_path = _get_gcp_metadata("instance", "zone")
  if zone_path is None:
    return None
  # example zone_path: "projects/123456789/zones/us-central1-a"
  return zone_path.rsplit("/", 1)[-1]
