# queued_resource_name=mattdavidow-d16
# project=tpu-prod-env-multipod
# zone=us-central2-b
# readFile="my_startup_script.sh"

queued_resource_name=$1
project=$2
zone=$3
readFile=$4

queuedResource=$(gcloud alpha compute tpus queued-resources describe "$queued_resource_name" --project "$project" --zone "$zone")

echo $queuedResource
nodeIds=$(echo "$queuedResource" | grep "nodeId" | awk -F ': ' '{print $2}')

if [[ "${#nodeIds[@]}" == 0 ]]; then
  echo "No nodes found in queued resource $queuedResource"
  exit 1
fi

echo "Found nodeIds: $nodeIds"


readFileContents=$(cat "$readFile")
echo $readFileContents  

for nodeId in $nodeIds; do
  echo $nodeId
  (
    echo "Updating startup script for node $nodeId"
    #gcloud alpha compute tpus tpu-vm update "$nodeId" --project "$project" --zone "$zone" --update-metadata="startup-script=$readFileContents"
    gcloud alpha compute tpus tpu-vm update "$nodeId" --project "$project" --zone "$zone" --metadata-from-file="startup-script=$readFile"
  )
  # threads+=($!)
done