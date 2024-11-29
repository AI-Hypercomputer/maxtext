# Codebase Walkthrough

MaxText is purely written in JAX and python. Below are some folders and files
that show a high-level organization of the code and some key files.

File/Folder | Description
---------|---------------------------------
 `configs` | Folder contains all the config file, including model configs (llama2, mistral etc) , and pre-optimized configs for different model size on different TPUs
 `input_pipelines` | Input training data related code
 `layers` | Model layer implementation
 `end_to_end` | Example scripts to run Maxtext
 `Maxtext/train.py` | The main training script you will run directly
 `Maxtext/config/base.yaml` | The base configuration file containing all the related info: checkpointing, model arch, sharding schema, data input, learning rate, profile, compilation, decode
 `Maxtext/decode.py` | This is a script to run offline inference with a sample prompt
 `setup.sh`| Bash script used to install all needed library dependencies.

## Training configuration

The [MaxText/configs/base.yaml](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/configs/base.yml)
has a set of default configurations. These can be overridden directly via CLI
when invoking the MaxText train scripts. The command line parameters overwrite
the default values. A few of the key parameters are described below:

- `load_parameters_path`: maxtext checkpoint path.
- `base_output_directory`: Base path to save the outputs (logs and data).
- [`dataset_type`](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/configs/base.yml#L273): 
  synthetic, tfds, grain or hf (hugging face)
- `dataset_path`: for `dataset_type=tfds`, path to the dataset.
- `tokenizer_path`: Path to a tokenizer for the model. The tokenizers are
  present in ...
- `quantization`: Whether to use quantized training with AQT. Valid values are ['int8']
- `per_device_batch_size`: How many batches each TPU/device receives. To improve
  the MFU, you can increase this value. This can also be a fraction. For this
  tutorial, we will use the default value of 1.
- `enable_checkpointing`: Boolean value. Whether we want to generate a checkpoint.
- `checkpoint_period`: After how many steps should checkpointing be performed.
- `async_checkpointing`: Accepts a boolean value to set whether to use
  asynchronous checkpointing. Here, we set it to False.
- `attention`: On TPUv3 and earlier, we need to set the attention to
  `dot_product`. Newer versions support the flash attention value. On GPU use
  `cudnn_flash_te`.
- `steps`: Number of steps to train. For this tutorial, we will use a small
  value of 10 steps.
