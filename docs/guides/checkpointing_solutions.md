(checkpointing_solutions)=

# Checkpointing

::::\{grid} 1 2 2 2
:gutter: 2

:::\{grid-item-card} 💾 GCS Checkpointing
:link: checkpointing_solutions/gcs_checkpointing
:link-type: doc

Standard checkpointing to Google Cloud Storage.
:::

:::\{grid-item-card} 🚑 Emergency Checkpointing
:link: checkpointing_solutions/emergency_checkpointing
:link-type: doc

Handle preemption and recover training progress.
:::

:::\{grid-item-card} 🗄️ Multi-tier checkpointing
:link: checkpointing_solutions/multi_tier_checkpointing
:link-type: doc

Optimize storage costs and performance with multi-tier usage.
:::

:::\{grid-item-card} 🔁 Checkpoint conversion utilities
:link: checkpointing_solutions/convert_checkpoint
:link-type: doc

Convenient tools to convert between Hugging Face and MaxText checkpoint.
:::
::::

```{toctree}
---
hidden:
maxdepth: 1
---
checkpointing_solutions/gcs_checkpointing.md
checkpointing_solutions/emergency_checkpointing.md
checkpointing_solutions/multi_tier_checkpointing.md
checkpointing_solutions/convert_checkpoint.md
```
