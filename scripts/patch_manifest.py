"""Helper script to patch Kubernetes manifests for JobSet."""

import subprocess
import sys
import yaml

manifest_file = sys.argv[1]
built_image = (
    sys.argv[2]
    if (len(sys.argv) > 2 and sys.argv[2] != "")
    else "gcr.io/cloud-tpu-multipod-dev/mohitkhatwani-rl:07152026-clean-v4"
)
kube_dns_ip = sys.argv[3] if (len(sys.argv) > 3 and sys.argv[3] != "") else "34.118.224.10"
workload_name = sys.argv[4] if (len(sys.argv) > 4 and sys.argv[4] != "") else "unknown"
gke_nodepool = sys.argv[5] if (len(sys.argv) > 5 and sys.argv[5] != "") else None

with open(manifest_file, "r", encoding="utf-8") as f:
  data = list(yaml.load_all(f, Loader=yaml.FullLoader))

for doc in data:
  if doc and doc.get("kind") == "JobSet":
    replicated_jobs = doc.get("spec", {}).get("replicatedJobs", [])

    startup_policy = doc.get("spec", {}).setdefault("startupPolicy", {})
    startup_policy["startupPolicyOrder"] = "AnyOrder"
    print("Set startupPolicyOrder to AnyOrder.")

    for job in replicated_jobs:
      pod_template = job.get("template", {}).get("spec", {}).get("template", {})
      pod_spec = pod_template.setdefault("spec", {})
      pod_spec["priorityClassName"] = "super-high"

      if job.get("name") == "pathways-head":
        node_selector = pod_spec.setdefault("nodeSelector", {})
        node_selector["cloud.google.com/gke-nodepool"] = "cpu-np"
        print("Set pathways-head nodeSelector to cpu-np.")

      if job.get("name") == "worker":
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
          del node_selector["cloud.google.com/gke-tpu-topology"]
          print(
              f"Moved cloud.google.com/gke-tpu-topology={topology} "
              "to annotations as cloud.google.com/gke-tpu-slice-topology."
          )

        if "cloud.google.com/placement-policy-name" in node_selector:
          del node_selector["cloud.google.com/placement-policy-name"]
          print("Removed cloud.google.com/placement-policy-name from nodeSelector.")

        accelerator = node_selector.get("cloud.google.com/gke-tpu-accelerator")
        if accelerator == "tpu7x":
          if gke_nodepool:
            try:
              res = subprocess.check_output(
                  [
                      "/usr/bin/kubectl",
                      "get",
                      "nodes",
                      "-l",
                      f"cloud.google.com/gke-nodepool={gke_nodepool}",
                      "-o",
                      "jsonpath={.items[0].metadata.labels.cloud\\.google\\.com/reservation-name}",
                  ],
                  text=True,
              ).strip()
              res_name = res if res else "ghostfish-bk33tcf1hfazr"
            except (subprocess.SubprocessError, OSError, ValueError):
              res_name = "ghostfish-bk33tcf1hfazr"
            node_selector["cloud.google.com/reservation-name"] = res_name
            node_selector["cloud.google.com/gke-nodepool"] = gke_nodepool
            print(
                f"Injected cloud.google.com/reservation-name={res_name} and gke-nodepool={gke_nodepool} into worker nodeSelector."  # pylint: disable=line-too-long
            )
          else:
            node_selector["cloud.google.com/reservation-name"] = "cloudtpu-20260317203000-769538580"
            if "cloud.google.com/gke-nodepool" in node_selector:
              del node_selector["cloud.google.com/gke-nodepool"]
            print("Injected cloud.google.com/reservation-name=cloudtpu-20260317203000-769538580 for NAP cluster.")
        elif accelerator == "tpu-v5p-slice":
          node_selector["cloud.google.com/reservation-name"] = "cloudtpu-20240716121201-595617744"
          print("Injected cloud.google.com/reservation-name=cloudtpu-20240716121201-595617744 into nodeSelector.")

        # 3. Patch placement-policy-name if it has mismatched suffix due to XPK version differences (REMOVED)

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
            if not any(e.get("name") == "VLLM_TORCH_PROFILER_DIR" for e in env if isinstance(e, dict)):
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
            resources = container.setdefault("resources", {})
            limits = resources.setdefault("limits", {})
            requests = resources.setdefault("requests", {})
            limits["memory"] = "200G"
            requests["memory"] = "200G"
            print("Set jax-tpu container memory limit and request to 200G.")
            if built_image:
              container["image"] = built_image
              print(f"Replaced placeholder image for jax-tpu with built image: {built_image}")
            command = container.get("command", [])
            if len(command) >= 3 and "python3 scripts/patch_vllm_sampler.py;" in command[2]:
              with open("scripts/patch_vllm_sampler.py", "r", encoding="utf-8") as pf:
                patch_content = pf.read()
              injected_patch = f"mkdir -p scripts && cat << 'EOF' > scripts/patch_vllm_sampler.py\n{patch_content}EOF\npython3 scripts/patch_vllm_sampler.py;"  # pylint: disable=line-too-long
              command[2] = command[2].replace("python3 scripts/patch_vllm_sampler.py;", injected_patch)
              print("Injected local scripts/patch_vllm_sampler.py into pathways-head jax-tpu container command.")
            elif len(command) >= 3 and "bash scripts/run_all.sh" in command[2]:
              with open("scripts/run_all.sh", "r", encoding="utf-8") as rf:
                run_all_content = rf.read()
              injected_script = f"mkdir -p scripts && cat << 'EOF' > scripts/run_all.sh\n{run_all_content}EOF\nchmod +x scripts/run_all.sh;"  # pylint: disable=line-too-long
              command[2] = command[2].replace("bash scripts/run_all.sh", f"{injected_script} bash scripts/run_all.sh")
              print("Injected local scripts/run_all.sh into pathways-head command.")

        init_containers = pod_spec.setdefault("initContainers", [])
        for container in init_containers:
          if container.get("name") == "pathways-proxy":
            resources = container.setdefault("resources", {})
            limits = resources.setdefault("limits", {})
            requests = resources.setdefault("requests", {})
            limits["memory"] = "50G"
            requests["memory"] = "50G"
            print("Reduced pathways-proxy container memory limit and request to 50G.")

with open(manifest_file, "w", encoding="utf-8") as f:
  yaml.dump_all(data, f)
