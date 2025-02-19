# syntax=docker/dockerfile:experimental
# Use Python 3.10 as the base image
FROM us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest

ENTRYPOINT [ "/usr/pathways/run/cloud_pathways_server_sanitized" ]

CMD [ "--server_port=29001", "--node_type=resource_manager", "--instance_count=1", "--instance_type=tpuv4:2x2x1", "--gcs_scratch_location=gs://cloud-pathways-staging/tmp" ]