
total_sleep_time=0
sleep_time=5
has_recent_version=0
good_version=432
while [ $has_recent_version -eq 0 ]
do
    version=$(gcloud version | head -n 1)
    version_num=$(echo "$version" | grep -oE '[0-9]+' | head -n 1)
    if [ "$version_num" -ge "$good_version" ]
    then
        echo "go me"
        # gcloud alpha compute tpus queued-resources delete {run_name} --force --quiet --async --project={project} --zone={zone}
        has_recent_version=1
    else
        sleep $sleep_time
        total_sleep_time=$(($total_sleep_time + $sleep_time))
        echo "Slept for a total of $total_sleep_time"
    fi
done