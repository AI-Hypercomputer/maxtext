## Instructions


Unless I told you to edit the code, always avoid directly modifying the code before asking. ("Implement, edit, change" are words for editing, "why" "what happened" are just for you to investigate and tell me the answer)

**IMPORTANT**: After founding a crash in the log, tell me immediately in the chat what is the crash, where is the log and your analysis before making further code changes. Before making code changes, tell me what is your plan and give grounded reasons that this is the solution to the crash.

**IMPORTANT**: For every claim you give to me, please make sure there exist sound evidence and not only based on toy experiments, and clearly state to me the evidence and validation.

**IMPORTANT**: For every data you present to me, tell me the exact data path. For every experiment, record core data to local workspace under MyStuff/Data/ and tell me in the chat the command you ran and the Pantheon log path.

Always squash or amend to keep exactly 1 commit on branches.
Always ask before pushing code.
Always ask before testing code.

> [!CAUTION]
> **Running Modified Code on remote TPU VM:**
> When running workloads with local modifications on the remote TPU VM, ensure the execution command runs from `/app/src/` (with `/app/src` prepended to `PYTHONPATH`), NOT from `/deps/src/` (which runs the pre-baked, unmodified container code).


## Subagent Management & Progress Reporting Guidelines

- **Periodic Status Updates (5-Minute Audit Rule)**:
  When subagents or long-running tasks are active, set up a 5-minute recurring schedule timer to actively query subagent progress and update the user. Do not wait silently for long periods without providing progress updates.
- **Workflow Logging (`progress.md`)**:
  For multi-step or parallel subagent experiments, maintain a live progress log file at `MyStuff/Docs/<task-name>/progress.md` to track dispatch timelines, subagent conversation IDs, task milestones, and audit findings. Make sure don't overwrite the markdown files, you can append and modify.
- **Single Active Jobset Limit**:
  When running workloads on shared XPK GKE clusters (`mlperf-v5p`), ensure only **1 active JobSet** per subagent is submitted at a time. Clean up stale/finished workloads (`xpk workload delete`) to prevent cluster resource exhaustion.
- **Short XPK Workload Names (< 20 Characters)**:
  Always keep `XPK_WORKLOAD` names short (e.g. `dlco-m1-01`) to prevent Kubernetes JobSet 63-byte label string truncation errors (`metadata.labels: Invalid value... must be no more than 63 bytes`).
- **Explicit Pantheon Log Error Auditing**:
  When checking Pantheon/Kubernetes logs for remote TPU workloads, always perform an explicit search for error keywords (`ERROR`, `Syncer crashed`, `INVALID_ARGUMENT`, `JaxRuntimeError`, `Traceback`). Never consider a run successful based solely on learner training step progress, because the background syncer thread may crash in parallel while the learner continues executing several training steps. Make sure you see the final xpk exit and see the loss logs for the whole process then you can make the conclusion that the execution is clean.
- **Final Synthesis & Documentation**:
  Upon completing an experiment or fix, write a comprehensive final synthesis report under `MyStuff/Docs/<task-name>/report.md` (or `<task-name>-fix.md`) detailing the root cause, solution diffs, local unit test results, remote TPU execution logs, and direct GCP Pantheon log links.


More skills at `./MyStuff/skills`:
- **diloco_doc.md**: Documentation on using the DiLoCo (Distributed Low-Communication) training technique in MaxText for efficient distributed training across multiple clusters/regions.
- **xpk_debug_guide.md**: Instructions for tracking job status, debugging issues, and retrieving logs for XPK jobs running on GKE.
- **xpk_clusters.md**: Guide on accessing and using shared XPK GKE clusters, detailing cluster availability (v4, v5e, v5p) and job submission workflows.
- **visualizing_flops/**: Guide for inspecting and visualizing training performance metrics (like TFLOP/s/device and MFU) using TensorBoard and CLI tools.
kubectl get jobset <RUN_NAME> \
      -o jsonpath='{.spec.replicatedJobs[0].template.spec.template.spec.containers[0].image}': to check the docker image of a run.

## Remote VM Setup

The venv is at ~/maxtext/maxtext_venv
Activate before executing commands.

## Python Env

```
source ~/../xpk_venv/bin/activate
# Or for maxtext venv
source ~/maxtext/maxtext_venv/bin/activate
# here you go
```

Always store output files and images to the local working directory (e.g., MyStuff/Data).

## The focus

The focus of my job is about diloco. @src/maxtext/trainers/diloco. 
