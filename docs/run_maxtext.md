# Run MaxText

MaxText provides flexible execution options ranging from local development and single-host experimentation to massively scalable training on thousands of chips. Choose the runbook that matches your infrastructure and goals.

## Local & Single Host
Ideal for development, debugging, and small-scale experimentation.

*   [**Localhost / Single VM**](run_maxtext/run_maxtext_localhost.md)
    *   The best starting point. Run directly on a single TPU VM or GPU machine (e.g., A3/H100).
    *   Great for learning the basics, testing configurations, and running small models.

*   [**Single Host GPU Guide**](run_maxtext/run_maxtext_single_host_gpu.md)
    *   Specific instructions for setting up and running on NVIDIA GPUs (A3/H100), including CUDA and Docker setup.

*   [**Decoupled Mode (No Cloud Dependencies)**](run_maxtext/decoupled_mode.md)
    *   Run tests and development loops completely offline without Google Cloud dependencies (GCS, JetStream, etc.).

## Multi-Host & Cluster (At Scale)
For large-scale training jobs running on GKE clusters.

*   [**Running with XPK (Recommended)**](run_maxtext/run_maxtext_via_xpk.md)
    *   The standard way to run production workloads on GKE.
    *   Uses the Accelerated Processing Kit (XPK) to orchestrate Docker containers across TPU/GPU clusters.

*   [**Running with Pathways**](run_maxtext/run_maxtext_via_pathways.md)
    *   Advanced orchestration using the Pathways backend on GKE.
    *   Supports both batch jobs and interactive "headless" workloads for development.

```{toctree}
:maxdepth: 1
:hidden:

run_maxtext/run_maxtext_localhost.md
run_maxtext/run_maxtext_single_host_gpu.md
run_maxtext/run_maxtext_via_xpk.md
run_maxtext/run_maxtext_via_pathways.md
run_maxtext/decoupled_mode.md
```
