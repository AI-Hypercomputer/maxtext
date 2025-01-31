GPU_NAME="vipannalla-mlperf-v41-a3-may"
ZONE="us-central1-a"
PROJECT="cloud-tpu-inference-test"


ssh_to_gpu() {
  gcloud compute ssh --zone ${ZONE} ${GPU_NAME} --project ${PROJECT} -- -o ProxyCommand='corp-ssh-helper %h %p'
}

copy_maxtext_files() {
  gcloud compute scp --zone ${ZONE} --project ${PROJECT} \
    $PWD/MaxText/profiler.py \
    ${GPU_NAME}:/scratch/jwyang-workspace/maxtext/MaxText/ \
    --scp-flag "-o ProxyCommand=corp-ssh-helper %h %p"
}