# Full Finetuninhg LLama2/LLama3 Optimized configuration

In the pre-training section you saw the steps on how to do pre-training with
MaxText. To perform full fine tuning, you need to pass the checkpoint to the
training script. 

Following is the parameter to assign a checkpoint to the training script.

- `load_parameters_path`: Path to the checkpoint directory

The high level steps involve:
- Converting the model checkpoints to MaxText formatted checkpoints
- Preparing the dataset so that data can be fed into the training script.
  MaxText provides sample pipelines to load the data via tf.data or Pygrain from
  a disk or gcs bucket. Or it can also input data directly from the hugging face
  dataset.
- Running the training script with the checkpoint
- Note: You may need to change the training parameters to fit the model to the
  TPU or GPU shape and to obtain an optimized performance.

## Parameters to achieve high MFU

This content is in progress.
