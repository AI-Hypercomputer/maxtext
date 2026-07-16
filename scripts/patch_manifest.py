"""Helper script to patch Kubernetes manifests for JobSet."""

import subprocess
import sys
import yaml

manifest_file = sys.argv[1]
built_image = sys.argv[2] if (len(sys.argv) > 2 and sys.argv[2] != "") else None
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

    # Swap order of replicated jobs so worker starts before pathways-head
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

    jobset_meta = doc.setdefault("metadata", {})
    jobset_labels = jobset_meta.setdefault("labels", {})
    jobset_annotations = jobset_meta.setdefault("annotations", {})

    # Check if target cluster is slice-scheduler based (e.g. forrest-ss-e2e-cluster)
    is_slice_scheduler = False
    try:
      cluster_info = subprocess.check_output(["/usr/bin/kubectl", "config", "current-context"], text=True)
      if "forrest-ss-e2e-cluster" in cluster_info:
        is_slice_scheduler = True
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
      pass

    if is_slice_scheduler:
      try:
        subprocess.run(
            [
                "/usr/bin/kubectl",
                "patch",
                "validatingwebhookconfiguration",
                "validating-webhook.slice.accelerator.gke.io",
                "--type=json",
                '-p=[{"op": "replace", "path": "/webhooks/0/rules/0/resources", "value": ["slices_disabled"]}]',
            ],
            check=False,
        )
        print("Patched validating-webhook.slice.accelerator.gke.io to disable slice update checking.")
      except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        print(f"Webhook patch warning: {e}")

      jobset_labels["cloud.google.com/slice-prefix"] = workload_name
      jobset_annotations["alpha.jobset.sigs.k8s.io/exclusive-topology"] = "cloud.google.com/gke-tpu-slice"
      print("Configured Slice Scheduler BYOS metadata labels and exclusive-topology annotation.")

    for job in replicated_jobs:
      pod_template = job.get("template", {}).get("spec", {}).get("template", {})
      pod_spec = pod_template.setdefault("spec", {})
      pod_spec["priorityClassName"] = "super-high"

      tolerations = pod_spec.setdefault("tolerations", [])
      if not any(
          t.get("key") == "google.com/tpu" and t.get("effect") == "NoSchedule" for t in tolerations if isinstance(t, dict)
      ):
        tolerations.append(
            {
                "key": "google.com/tpu",
                "operator": "Exists",
                "effect": "NoSchedule",
            }
        )
        print("Explicitly added google.com/tpu NoSchedule toleration.")

      if not any(t.get("key") == "kwok.x-k8s.io/node" for t in tolerations if isinstance(t, dict)):
        tolerations.append(
            {
                "key": "kwok.x-k8s.io/node",
                "operator": "Exists",
                "effect": "NoSchedule",
            }
        )
        print("Added KWOK node toleration.")

      if job.get("name") == "worker":
        # Remove replicatedJob metadata exclusive-topology if present so it doesn't conflict
        rep_job_meta = job.get("template", {}).setdefault("metadata", {})
        rep_job_annotations = rep_job_meta.setdefault("annotations", {})
        if "alpha.jobset.sigs.k8s.io/exclusive-topology" in rep_job_annotations:
          del rep_job_annotations["alpha.jobset.sigs.k8s.io/exclusive-topology"]

        rep_job_labels = rep_job_meta.setdefault("labels", {})
        rep_job_labels["cloud.google.com/gke-tpu-accelerator"] = "tpu7x"
        rep_job_labels["cloud.google.com/gke-tpu-topology"] = "4x4x4"

        metadata = pod_template.setdefault("metadata", {})
        labels = metadata.setdefault("labels", {})
        labels["kueue.x-k8s.io/podset"] = "worker"
        annotations = metadata.setdefault("annotations", {})

        # 1. Add skip-tpu-webhook-check annotation
        annotations["cloud.google.com/skip-tpu-webhook-check"] = "true"
        print("Added skip-tpu-webhook-check=true annotation.")

        # 2. Set gke-tpu-slice-topology annotation
        pod_spec = pod_template.setdefault("spec", {})
        node_selector = pod_spec.setdefault("nodeSelector", {})
        annotations["cloud.google.com/gke-tpu-slice-topology"] = "4x4x4"

        if "cloud.google.com/placement-policy-name" in node_selector:
          del node_selector["cloud.google.com/placement-policy-name"]
          print("Removed cloud.google.com/placement-policy-name from nodeSelector.")

        if "cloud.google.com/gke-accelerator-count" in node_selector:
          del node_selector["cloud.google.com/gke-accelerator-count"]
          print("Removed cloud.google.com/gke-accelerator-count from nodeSelector.")

        if is_slice_scheduler:
          if "cloud.google.com/reservation-name" in node_selector:
            del node_selector["cloud.google.com/reservation-name"]
          node_selector["cloud.google.com/gke-tpu-topology"] = "4x4x4"
          node_selector["cloud.google.com/gke-tpu-accelerator"] = "tpu7x"
          print("Set nodeSelector strictly to gke-tpu-topology: 4x4x4 and gke-tpu-accelerator: tpu7x.")
        else:
          accelerator = node_selector.get("cloud.google.com/gke-tpu-accelerator")
          if accelerator == "tpu7x":
            try:
              res = subprocess.check_output(
                  [
                      "/usr/bin/kubectl",
                      "get",
                      "nodes",
                      "-l",
                      "cloud.google.com/gke-tpu-accelerator=tpu7x",
                      "-o",
                      "jsonpath={.items[0].metadata.labels.cloud\\.google\\.com/reservation-name}",
                  ],
                  text=True,
              ).strip()
              res_name = res if res else "cloudtpu-20260710003900-159478293"
            except (subprocess.SubprocessError, OSError, ValueError):
              res_name = "cloudtpu-20260710003900-159478293"
            node_selector["cloud.google.com/reservation-name"] = res_name
            print(f"Injected cloud.google.com/reservation-name={res_name} into nodeSelector.")
          elif accelerator == "tpu-v5p-slice":
            node_selector["cloud.google.com/reservation-name"] = "cloudtpu-20240716121201-595617744"
            print("Injected cloud.google.com/reservation-name=cloudtpu-20240716121201-595617744 into nodeSelector.")

        # 3. Patch placement-policy-name if it has mismatched suffix due to XPK version differences (REMOVED)

        # 3.5 Inject dnsConfig to resolve internal pods DNS and external googleapis.com DNS over hostNetwork
        pod_spec["dnsConfig"] = {"nameservers": [kube_dns_ip, "8.8.8.8"]}
        print(f"Injected dnsConfig nameservers: [{kube_dns_ip}, 8.8.8.8] to worker pod spec.")

        # 3.6 Inject gke-nodepool if provided
        if gke_nodepool:
          node_selector["cloud.google.com/gke-nodepool"] = gke_nodepool
          print(f"Injected cloud.google.com/gke-nodepool={gke_nodepool} to worker nodeSelector.")

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
        job_template = job.setdefault("template", {})
        job_meta = job_template.setdefault("metadata", {})
        job_annotations = job_meta.setdefault("annotations", {})
        job_annotations["slice-provisioner.gke.io/skip-slice-injection"] = "true"

        pod_template = job_template.setdefault("spec", {}).setdefault("template", {})
        pod_meta = pod_template.setdefault("metadata", {})
        pod_annotations = pod_meta.setdefault("annotations", {})
        pod_annotations["slice-provisioner.gke.io/skip-slice-injection"] = "true"

        pod_spec = pod_template.setdefault("spec", {})
        node_selector = pod_spec.setdefault("nodeSelector", {})
        node_selector.pop("cloud.google.com/gke-tpu-slice", None)

        if is_slice_scheduler or gke_nodepool in ["np1", "np2"]:
          node_selector["cloud.google.com/gke-nodepool"] = "large-cpu-pool"
          print("Set pathways-head nodeSelector to large-cpu-pool.")
        elif gke_nodepool:
          node_selector["cloud.google.com/gke-nodepool"] = gke_nodepool

        containers = pod_spec.setdefault("containers", [])
        for container in containers:
          c_name = container.get("name")
          resources = container.setdefault("resources", {})
          limits = resources.setdefault("limits", {})
          requests = resources.setdefault("requests", {})
          if c_name == "jax-tpu":
            limits["memory"] = "200G"
            requests["memory"] = "180G"
            print("Set jax-tpu container memory limit to 200G and request to 180G.")
          elif c_name == "pathways-resource-manager":
            limits["memory"] = "50G"
            requests["memory"] = "30G"
          elif c_name == "pathways-proxy":
            limits["memory"] = "20G"
            requests["memory"] = "10G"

          if c_name == "jax-tpu":
            if built_image:
              container["image"] = built_image
              print(f"Replaced placeholder image with built image: {built_image}")
            elif not container.get("image"):
              default_runner_img = "gcr.io/cloud-tpu-multipod-dev/mohitkhatwani-rl:agentic"
              container["image"] = default_runner_img
              print(f"Set missing jax-tpu image to base runner image: {default_runner_img}")
            command = container.get("command", [])
            if len(command) >= 3 and "python3 scripts/patch_vllm_sampler.py;" in command[2]:
              with open("scripts/patch_vllm_sampler.py", "r", encoding="utf-8") as pf:
                patch_content = pf.read()
              injected_patch = f"mkdir -p /tmp/scripts && cat << 'EOF' > /tmp/scripts/patch_vllm_sampler.py\n{patch_content}EOF\npython3 /tmp/scripts/patch_vllm_sampler.py;"  # pylint: disable=line-too-long
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
