# How-to Guides

Practical step-by-step guides for common tasks, optimizations, and workflows in MaxText.

## Performance & Optimization
*   [**Optimization Factors**](guides/optimization.md)
    *   Running custom models, configuring sharding strategies, and writing high-performance Pallas kernels.
*   [**Monitoring & Debugging**](guides/monitoring_and_debugging.md)
    *   Tools for diagnosing performance issues, including Goodput monitoring, Cloud Logging, and XProf profiling.

## Data & Storage
*   [**Data Input Pipelines**](guides/data_input_pipeline.md)
    *   Configuring data loaders for high performance. Includes Grain (ArrayRecord), Hugging Face, and TFDS pipelines.
*   [**Checkpointing**](guides/checkpointing_solutions.md)
    *   Strategies for saving and restoring model state, including GCS checkpointing, emergency recovery, and multi-tier solutions.

## Development Workflows
*   [**Python Notebooks**](guides/run_python_notebook.md)
    *   Interactive development using Jupyter/Colab on TPUs. Covers local port-forwarding and Colab setups.

```{toctree}
:maxdepth: 1
:hidden:

guides/optimization.md
guides/data_input_pipeline.md
guides/checkpointing_solutions.md
guides/monitoring_and_debugging.md
guides/run_python_notebook.md
```
