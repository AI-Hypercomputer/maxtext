def create_kill_command_str():
  # pylint: disable=line-too-long
  return f"""if [[ $SLICE_ID -eq 0 && $WORKER_ID -eq 0 ]]; then
  gcloud alpha compute tpus queued-resources delete args.RUN_NAME --force --quiet --project=args.PROJECT --zone=args.ZONE
  fi"""

print(create_kill_command_str())

