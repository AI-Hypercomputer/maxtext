"""Helper script to patch Kubernetes manifests for JobSet."""

import yaml
import sys

manifest_file = sys.argv[1]
built_image = sys.argv[2] if (len(sys.argv) > 2 and sys.argv[2] != "") else None
kube_dns_ip = sys.argv[3] if len(sys.argv) > 3 else "34.118.224.10"
workload_name = sys.argv[4] if len(sys.argv) > 4 else "unknown"

with open(manifest_file, "r", encoding="utf-8") as f:
  data = list(yaml.load_all(f, Loader=yaml.FullLoader))

for doc in data:
  if doc and doc.get("kind") == "JobSet":
    replicated_jobs = doc.get("spec", {}).get("replicatedJobs", [])

    # Swap order of replicated jobs so worker starts before pathways-head under InOrder startup policy
    worker_job = None
    head_job = None
    for job in replicated_jobs:
      if job.get("name") == "worker":
        worker_job = job
      elif job.get("name") == "pathways-head":
        head_job = job
    if worker_job and head_job:
      replicated_jobs.remove(worker_job)
      replicated_jobs.insert(0, worker_job)
      print("Swapped replicatedJobs order: 'worker' will start before 'pathways-head'.")

    for job in replicated_jobs:
      if job.get("name") == "worker":
        pod_template = job.get("template", {}).get("spec", {}).get("template", {})

        metadata = pod_template.setdefault("metadata", {})
        labels = metadata.setdefault("labels", {})
        labels["kueue.x-k8s.io/podset"] = "worker"
        annotations = metadata.setdefault("annotations", {})

        # 1. Add skip-tpu-webhook-check annotation
        annotations["cloud.google.com/skip-tpu-webhook-check"] = "true"
        print("Added skip-tpu-webhook-check=true annotation.")

        # 2. Get topology from nodeSelector, set it in annotations, and remove from nodeSelector
        pod_spec = pod_template.setdefault("spec", {})
        node_selector = pod_spec.setdefault("nodeSelector", {})
        if "cloud.google.com/gke-tpu-topology" in node_selector:
          topology = node_selector["cloud.google.com/gke-tpu-topology"]
          annotations["cloud.google.com/gke-tpu-slice-topology"] = topology
          print(
              f"Moved cloud.google.com/gke-tpu-topology={topology} "
              "to annotations as cloud.google.com/gke-tpu-slice-topology."
          )

        # 3. Patch placement-policy-name if it has mismatched suffix due to XPK version differences
        if "cloud.google.com/placement-policy-name" in node_selector:
          val = node_selector["cloud.google.com/placement-policy-name"]
          if "-placement-policy" in val and "-ss-" not in val:
            node_selector["cloud.google.com/placement-policy-name"] = val.replace(
                "-placement-policy", "-ss-placement-policy"
            )
            print(f"Patched placement-policy-name to {node_selector['cloud.google.com/placement-policy-name']}")

        # 3.5 Inject dnsConfig to resolve internal pods DNS and external googleapis.com DNS over hostNetwork
        pod_spec["dnsConfig"] = {"nameservers": [kube_dns_ip, "8.8.8.8"]}
        print(f"Injected dnsConfig nameservers: [{kube_dns_ip}, 8.8.8.8] to worker pod spec.")

        # 5. Add connection handshake timeout and VLLM_TORCH_PROFILER_DIR to pathways-worker container
        containers = pod_spec.setdefault("containers", [])
        for container in containers:
          if container.get("name") == "pathways-worker":
            args = container.setdefault("args", [])
            if not any("temporary_flag_for_debugging_pipe_unreachable_timeout" in a for a in args):
              args.append("--temporary_flags_for_debugging=temporary_flag_for_debugging_pipe_unreachable_timeout=30m")
              print("Added pipe_unreachable_timeout=30m to pathways-worker args.")

            # Inject VLLM_TORCH_PROFILER_DIR env var
            env = container.setdefault("env", [])
            env.append(
                {
                    "name": "VLLM_TORCH_PROFILER_DIR",
                    "value": f"gs://runner-maxtext-logs/{workload_name}/tensorboard/sampler_tpu_profile",
                }
            )
            print(f"Injected VLLM_TORCH_PROFILER_DIR to pathways-worker container for workload {workload_name}")
      elif job.get("name") == "pathways-head":
        pod_template = job.get("template", {}).get("spec", {}).get("template", {})
        metadata = pod_template.setdefault("metadata", {})
        labels = metadata.setdefault("labels", {})
        labels["kueue.x-k8s.io/podset"] = "pathways-head"

        pod_spec = pod_template.setdefault("spec", {})
        containers = pod_spec.setdefault("containers", [])
        for container in containers:
          if container.get("name") == "jax-tpu":
            if built_image:
              container["image"] = built_image
              print(f"Replaced placeholder image with built image: {built_image}")
            command = container.get("command", [])
            if len(command) >= 3 and "python3 scripts/patch_vllm_sampler.py;" in command[2]:
              with open("scripts/patch_vllm_sampler.py", "r", encoding="utf-8") as pf:
                patch_content = pf.read()
              injected_patch = f"cat << 'EOF' > scripts/patch_vllm_sampler.py\n{patch_content}EOF\npython3 scripts/patch_vllm_sampler.py;"  # pylint: disable=line-too-long
              command[2] = command[2].replace("python3 scripts/patch_vllm_sampler.py;", injected_patch)
              print("Injected local scripts/patch_vllm_sampler.py into pathways-head command.")
            elif len(command) >= 3 and "bash scripts/run_all.sh" in command[2]:
              with open("scripts/run_all.sh", "r", encoding="utf-8") as rf:
                run_all_content = rf.read()
              injected_script = f"mkdir -p scripts && cat << 'EOF' > scripts/run_all.sh\n{run_all_content}EOF\nchmod +x scripts/run_all.sh;"  # pylint: disable=line-too-long
              command[2] = command[2].replace("bash scripts/run_all.sh", f"{injected_script} bash scripts/run_all.sh")
              print("Injected local scripts/run_all.sh into pathways-head command.")

with open(manifest_file, "w", encoding="utf-8") as f:
  yaml.dump_all(data, f)
