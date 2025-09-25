# GCS bucket-based checkpointing

In this type of checkpointing, the checkpoints are saved directly to a Google
Cloud Storage (GCS) bucket. This is the most straightforward and common method.
The training process briefly pauses, for capturing the model's state and keeping
the training in sync with rest of the workers, then the model's state is
asynchronously serialized and written over the network to the specified GCS
bucket.

## Checkpoint loading priority

The system follows a specific order when deciding which checkpoint to load at startup. The first valid condition met is the one executed:

1.  **Resume Current Run**: If a checkpoint already exists for the current
    `run_name`, the system loads the latest fully-saved checkpoint. This is the
    default behavior to ensure minimal state loss when resuming after an
    interruption.
2.  **Load from Specific Path**: The system checks for a user-specified path.
    * If `load_parameters_path` is set, we load a parameter only checkpoint from that path..
    * If `load_full_state_path` is set, we load a full state checkpoint from that path.
    * **Note**: These two options are mutually exclusive and will cause an error if both are set.
3.  **Initialize from Scratch**: We don't load a checkpoint and initialize state instead.

### MaxText configuration

| Flag | Description | Type | Default |
| :--- | :--- | :--- | :--- |
| `enable_checkpointing` | A master switch to enable (`True`) or disable (`False`) saving checkpoints during the training run. | `boolean` | `False` |
| `async_checkpointing` | When set to (`True`), this flag makes checkpoint saving asynchronous. The training step is only blocked for the minimal time needed to capture the model's state, and the actual writing to storage happens in a background thread. This is highly recommended for performance. It's enabled by default. | `boolean` | `True` |
| `checkpoint_period` | The interval, in training steps, for how often a checkpoint is saved. | `integer` | `10000` |
| `enable_single_replica_ckpt_restoring` | If `True`, one replica reads the checkpoint from storage and then broadcasts it to all other replicas. This can significantly speed up restoration on multi-host systems by reducing redundant reads from storage.<br>**Note**: This feature is only compatible with training jobs that utilize a Distributed Data Parallel (DDP) strategy. | `boolean` | `False` |
| `load_parameters_path` | Specifies a path to a checkpoint directory to load a parameter only checkpoint.<br>**Example**: `"gs://my-bucket/my-previous-run/checkpoints/items/1000"` | `string` | `""` (disabled) |
| `load_full_state_path` | Specifies a path to a checkpoint directory to load a full checkpoint including optimizer state and step count from a specific directory.<br>**Example**: `"gs://my-bucket/my-interrupted-run/checkpoints/items/500"` | `string` | `""` (disabled) |
| `lora_input_adapters_path` | Specifies a parent directory containing LoRA (Low-Rank Adaptation) adapters. | `string` | `""` (disabled) |
| `force_unroll` | If `True`, unrolls the loop when generating a parameter-only checkpoint. | `boolean` | `False` |

## Storage and format configuration

These settings control the underlying storage mechanism ([Orbax](https://orbax.readthedocs.io)) for performance and compatibility.

| Flag | Description | Type | Default |
| :--- | :--- | :--- | :--- |
| `checkpoint_storage_target_data_file_size_bytes` | Sets a target file size for Orbax to chunk large arrays into smaller physical files. This can dramatically speed up loading over a network and in distributed environments. | `integer` | `2147483648` (2 GB) |
| `checkpoint_storage_use_ocdbt` | If `True`, uses the TensorStore **OCDBT** (Optionally-Cooperative Distributed B+ Tree)) key-value store as the underlying storage format for checkpointing. | `boolean` | `True` |
| `checkpoint_storage_use_zarr3` | If `True`, uses the Zarr v3 storage format within Orbax, which is optimized for chunked, compressed, N-dimensional arrays. | `boolean` | `True` |
| `checkpoint_storage_concurrent_gb` | Controls the concurrent I/O limit in gigabytes for the checkpointer. Larger models may require increasing this value to avoid I/O bottlenecks. | `integer` | `96` |
| `enable_orbax_v1` | A boolean flag to explicitly enable features and behaviors from Orbax version 1. | `boolean` | `False` |
| `source_checkpoint_layout` | Specifies the format of the checkpoint being **loaded**. This tells the system how to interpret the files at the source path.<br>**Options**: `"orbax"`, `"safetensors"` | `string` | `"orbax"` |
| `checkpoint_conversion_fn` | A user-defined function to process a loaded checkpoint dictionary into a format that the model can understand. This is essential for loading checkpoints from different frameworks or formats (e.g., converting keys from a Hugging Face SafeTensors file). | `function` or `None` | `None` |
