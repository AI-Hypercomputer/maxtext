# Checkpointing

Maxtext provides the ability to run training with following checkpointing options:

- enabled/disabled
- asynchronous - true/false
- checkpointing frequency

They are dictated by the following parameters:

- `enable_checkpointing` (`True`/`False`) 
- `checkpoint_period` (integer value)
- `async_checkpointing` (`True`/`False`)
