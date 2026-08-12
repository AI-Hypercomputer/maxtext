## Instructions


Unless I told you to edit the code, always avoid directly modifying the code before asking. ("Implement, edit, change" are words for editing, "why" "what happened" are just for you to investigate and tell me the answer)

**IMPORTANT**: After founding a crash in the log, tell me immediately in the chat what is the crash, where is the log and your analysis before making further code changes. Before making code changes, tell me what is your plan and give grounded reasons that this is the solution to the crash.

**IMPORTANT**: For every claim you give to me, please make sure there exist sound evidence and not only based on toy experiments, and clearly state to me the evidence and validation.

**IMPORTANT**: STRICT ZERO ESTIMATION / PROJECTION RULE: Never estimate, simulate, fabricate, or synthesize sub-operation timings, kernel breakdowns, or trace segments based on theoretical FLOP ratios, assumptions, or proportions. Every single metric, timestamp, kernel duration, and plot presented MUST be directly and strictly measured from actual execution logs, TensorBoard event files, or physical hardware traces (xplane.pb). If fine-grained sub-op data is not present in the hardware trace, state clearly that it is unavailable and explain what compiler flags or instrumentation are needed to collect real measurements.

**IMPORTANT**: For every data you present to me, tell me the exact data path. For every experiment, record core data to local workspace under MyStuff/Data/ and tell me in the chat the command you ran and the Pantheon log path.

**IMPORTANT**: When solving bugs:
You must corroborate hypothesis with evidence from bugreport. If you can't find any evidence from bugreport, your hypothesis must be rooted against prior knowledge, such as similar buganizer tickets, source code or documentation. When your hypothesis is rooted in prior knowledge, you must still prove similarity with evidence. Surface looking similarities without corroborating evidence is not valid. Any hypothesis without associated evidence is a guess, not investigation.

Always ask before pushing code.
Always ask before testing code unless I told you OK in the session.

When doing multiple rounds of testing, output the failure traceback and reasons for each round to keep me updated in the chat session.

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
- **Explicit Pantheon Log Error Auditing & Live Status Verification**:
  When checking Pantheon/Kubernetes logs for remote TPU workloads:
  1. **Live Pod Status**: Always check `kubectl get pods` to verify the actual pod phase (`Running`, `Completed`, or `Error`) before making any claims.
  2. **Timestamp Verification**: Always check log entry timestamps against current UTC time to ensure you are auditing the live/active execution window, rather than buffered or stale log snippets from previous container runs.
  3. **Explicit Error Search**: Always perform an explicit search for error keywords (`ERROR`, `Syncer crashed`, `INVALID_ARGUMENT`, `JaxRuntimeError`, `Traceback`) on the full log output. Never consider a run successful based solely on step progress or tail snippets, as background sync threads or layout resharding may fail asynchronously. Only claim a run is clean after confirming `EXIT_CODE=0` on a fully completed pod with 0 error matches.
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

GDoc Agent:
/google/bin/releases/gemini-agents-gdocs/gdocs