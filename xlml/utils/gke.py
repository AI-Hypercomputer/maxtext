import base64
import concurrent.futures
import datetime
import logging
import tempfile
from typing import Any, Dict, Optional

from airflow.decorators import task, task_group
import google.auth
import google.auth.transport.requests
from google.cloud import container_v1
import kubernetes

from xlml.apis import gcp_config


def get_authenticated_client(
    project_name: str, region: str, cluster_name: str
) -> kubernetes.client.ApiClient:
  container_client = container_v1.ClusterManagerClient()
  cluster_path = f'projects/{project_name}/locations/{region}/clusters/{cluster_name}'
  response = container_client.get_cluster(name=cluster_path)
  creds, _ = google.auth.default()
  auth_req = google.auth.transport.requests.Request()
  creds.refresh(auth_req)
  configuration = kubernetes.client.Configuration()
  configuration.host = f'https://{response.endpoint}'
  with tempfile.NamedTemporaryFile(delete=False) as ca_cert:
    ca_cert.write(base64.b64decode(response.master_auth.cluster_ca_certificate))
  configuration.ssl_ca_cert = ca_cert.name
  configuration.api_key_prefix['authorization'] = 'Bearer'
  configuration.api_key['authorization'] = creds.token

  return kubernetes.client.ApiClient(configuration)


@task_group
def run_job(
    body: Dict[str, Any],
    gcp: gcp_config.GCPConfig,
    cluster_name: str,
    job_create_timeout: datetime.timedelta,
):
  """Run a batch job directly on a GKE cluster.

  Args:
    body: Dict that defines a Kubernetes `Job`.
    gcp: GCP config with the project name and zone of the GKE cluster.
    cluster_name: Name of the GCP cluster.
    job_create_timeout: Amount of time to wait for all pods to become active.
  """

  @task
  def deploy_job():
    client = get_authenticated_client(gcp.project_name, gcp.zone, cluster_name)

    jobs_client = kubernetes.client.BatchV1Api(client)
    resp = jobs_client.create_namespaced_job(namespace='default', body=body)

    logging.info(f'response: {resp}')

    return resp.metadata.name

  @task.sensor(
      poke_interval=60, timeout=job_create_timeout.total_seconds(), mode='reschedule'
  )
  def stream_logs(name: str):
    client = get_authenticated_client(gcp.project_name, gcp.zone, cluster_name)

    batch_api = kubernetes.client.BatchV1Api(client)
    job = batch_api.read_namespaced_job(namespace='default', name=name)

    # TODO(wcromar): Handle other conditions (e.g. unschedulablility)
    logging.info(f'Job status: {job.status}')
    if job.status.failed:
      raise RuntimeError(f'Job has {job.status.failed} failed pods.')

    core_api = kubernetes.client.CoreV1Api(client)
    pod_label_selector = f'batch.kubernetes.io/job-name={name}'
    pods = core_api.list_namespaced_pod(
        namespace='default', label_selector=pod_label_selector
    )

    if len(pods.items) != body['spec']['parallelism']:
      logging.info('Waiting for all pods to be created...')
      return False

    def _watch_pod(name, namespace) -> Optional[int]:
      logs_watcher = kubernetes.watch.Watch()

      logging.info(f'Waiting for pod {name} to start...')
      pod_watcher = kubernetes.watch.Watch()
      for event in pod_watcher.stream(
          core_api.list_namespaced_pod,
          namespace,
          field_selector=f'metadata.name={name}',
      ):
        status = event['object'].status
        logging.info(f'Pod {event["object"].metadata.name} status: {status.phase}')
        if status.phase != 'Pending':
          break

      logging.info(f'Streaming pod logs for {name}...')
      for line in logs_watcher.stream(
          core_api.read_namespaced_pod_log, name, namespace, _request_timeout=3600
      ):
        logging.info(f'{name}] {line}')

      logging.warning(f'Lost logs stream for {name}.')

      pod = core_api.read_namespaced_pod(namespace='default', name=name)
      if pod.status.container_statuses:
        container_status = pod.status.container_statuses[0]
        if pod.status.container_statuses[0].state.terminated:
          exit_code = container_status.state.terminated.exit_code
          if exit_code:
            logging.error(f'Pod {name} had non-zero exit code {exit_code}')

          return exit_code

      logging.warning(f'Unknown status for pod {name}')
      return None

    with concurrent.futures.ThreadPoolExecutor() as executor:
      futures = []
      for pod in pods.items:
        f = executor.submit(_watch_pod, pod.metadata.name, pod.metadata.namespace)
        futures.append(f)

      # Wait for pods to complete, and exit with the first non-zero exit code.
      for f in concurrent.futures.as_completed(futures):
        exit_code = f.result()
        if exit_code:
          raise RuntimeError('Non-zero exit code')

    return True

  name = deploy_job()
  stream_logs(name)
