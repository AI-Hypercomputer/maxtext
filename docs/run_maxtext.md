# Run MaxText

Choose your environment and orchestration method to run MaxText.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} 💻 Localhost / Single VM
:link: run_maxtext/run_maxtext_localhost
:link-type: doc

Get started quickly on a single machine. Clone the repo, install dependencies, and run your first training job on a single TPU or GPU VM.
:::

:::{grid-item-card} 🎮 Single-host GPU
:link: run_maxtext/run_maxtext_single_host_gpu
:link-type: doc

Run MaxText on single-host NVIDIA GPUs (e.g., A3 High/Mega). Includes Docker setup, NVIDIA Container Toolkit installation, and 1B/7B model training examples.
:::

:::{grid-item-card} 🏗️ At scale with XPK (GKE)
:link: run_maxtext/run_maxtext_via_xpk
:link-type: doc

Deploy to Google Kubernetes Engine (GKE) using XPK. Orchestrate large-scale training jobs on TPU or GPU clusters with simple CLI commands.
:::

:::{grid-item-card} 🌐 Multi-host via Pathways
:link: run_maxtext/run_maxtext_via_pathways
:link-type: doc

Run large-scale JAX jobs on TPUs using Pathways. Supports batch and headless (interactive) workloads on GKE.
:::

:::{grid-item-card} 🔌 Decoupled Mode
:link: run_maxtext/decoupled_mode
:link-type: doc

Run tests and local development without Google Cloud dependencies (no `gcloud`, GCS, or Vertex AI required).
:::
::::

```{toctree}
:hidden:
:maxdepth: 1

run_maxtext/run_maxtext_localhost.md
run_maxtext/run_maxtext_single_host_gpu.md
run_maxtext/run_maxtext_via_xpk.md
run_maxtext/run_maxtext_via_pathways.md
run_maxtext/decoupled_mode.md
```
