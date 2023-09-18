PROJECT_ID=tpu-prod-env-vlp-2nic
gcloud config set project $PROJECT_ID
ZONE=us-east5-b
gcloud config set compute/zone $ZONE

WORKLOAD_NAME=${USER}-st


while true; do

    workloads=$(python3 ../experimental/users/vbarr/multipod/xpk/xpk.py workload list --cluster bodaborgprivate5 | grep -E "${USER}-st.* (finished|failed)" | awk '{print $1}')

    # Loop through the workloads and print each one
    for workload in $workloads; do
        echo "deleting $workload"
        python3 ../experimental/users/vbarr/multipod/xpk/xpk.py workload delete --cluster bodaborgprivate5 --workload $workload
    done

    # Sleep for 1 minute before running the script again
    echo "sleep 30s"
    sleep 30
done