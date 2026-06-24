"""Helper script to patch Kubernetes manifests for JobSet."""

import yaml
import sys

manifest_file = sys.argv[1]

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
          del node_selector["cloud.google.com/gke-tpu-topology"]

        # 3. Patch placement-policy-name if it has mismatched suffix due to XPK version differences
        if "cloud.google.com/placement-policy-name" in node_selector:
          val = node_selector["cloud.google.com/placement-policy-name"]
          if "-placement-policy" in val and "-ss-" not in val:
            node_selector["cloud.google.com/placement-policy-name"] = val.replace(
                "-placement-policy", "-ss-placement-policy"
            )
            print(f"Patched placement-policy-name to {node_selector['cloud.google.com/placement-policy-name']}")

        # 4. Remove exclusive-topology annotation from replicatedJob metadata
        job_template = job.get("template", {})
        job_metadata = job_template.setdefault("metadata", {})
        job_annotations = job_metadata.setdefault("annotations", {})
        if "alpha.jobset.sigs.k8s.io/exclusive-topology" in job_annotations:
          del job_annotations["alpha.jobset.sigs.k8s.io/exclusive-topology"]
          print("Removed alpha.jobset.sigs.k8s.io/exclusive-topology annotation.")

        # 5. Add connection handshake timeout to pathways-worker container
        containers = pod_spec.setdefault("containers", [])
        for container in containers:
          if container.get("name") == "pathways-worker":
            args = container.setdefault("args", [])
            if not any("temporary_flag_for_debugging_pipe_unreachable_timeout" in a for a in args):
              args.append("--temporary_flags_for_debugging=temporary_flag_for_debugging_pipe_unreachable_timeout=30m")
              print("Added pipe_unreachable_timeout=30m to pathways-worker args.")
      elif job.get("name") == "pathways-head":
        pod_spec = job.get("template", {}).get("spec", {}).get("template", {}).get("spec", {})
        containers = pod_spec.setdefault("containers", [])
        for container in containers:
          if container.get("name") == "jax-tpu":
            command = container.get("command", [])
            if len(command) >= 3 and "python3 scripts/patch_vllm_sampler.py;" in command[2]:
              with open("scripts/patch_vllm_sampler.py", "r", encoding="utf-8") as pf:
                patch_content = pf.read()
              injected_patch = f"cat << 'EOF' > scripts/patch_vllm_sampler.py\n{patch_content}EOF\npython3 scripts/patch_vllm_sampler.py;"  # pylint: disable=line-too-long
              command[2] = command[2].replace("python3 scripts/patch_vllm_sampler.py;", injected_patch)
              print("Injected local scripts/patch_vllm_sampler.py into pathways-head command.")

with open(manifest_file, "w", encoding="utf-8") as f:
  yaml.dump_all(data, f)
