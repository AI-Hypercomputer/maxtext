from typing import Sequence
import glob
import datetime
# import pyconfig
# from train import create_data_iterator
# from input_pipeline._grain_data_processing import get_datasets
from input_pipeline import _input_pipeline_utils
from input_pipeline import _grain_tokenizer
import grain.python as grain
import numpy as np
from absl import app

# def setup_dataloader(config):
#   mesh = None
#   data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
config = {
  "grain_train_files": "/tmp/gcsfuse/maxtext-dataset/array-record/c4/en/3.0.1/c4-train.array_record-*",
  "grain_worker_count": 2,
  "steps": 1000,
  "max_target_length": 2048,
  "tokenizer_path": "assets/tokenizer.llama2",
  "global_batch_size": 32,
}

def dummy_grain_iterator(config):
  data_files = glob.glob(config["grain_train_files"])
  dataset = grain.ArrayRecordDataSource(data_files)

  operations = []
  operations.append(_input_pipeline_utils.ParseFeatures())
  operations.append(_input_pipeline_utils.NormalizeFeatures())
  operations.append(_grain_tokenizer.TokenizeAndTrim(["inputs", "targets"], config["max_target_length"], 
                                                     config["tokenizer_path"], True, True))
  operations.append(
    grain.experimental.PackAndBatchOperation(
        batch_size=config["global_batch_size"],
        length_struct={"inputs": config["max_target_length"], "targets": config["max_target_length"]}
    )
  )
  operations.append(_input_pipeline_utils.ReformatPacking())
  operations.append(_input_pipeline_utils.ShiftData(axis=1))

  index_sampler = grain.IndexSampler(
    num_records=len(dataset),
    shard_options=grain.ShardOptions(
      shard_index=0, shard_count=1, drop_remainder=True
    ),
  )
  dataloader = grain.DataLoader(
    data_source=dataset,
    operations=operations,
    sampler=index_sampler,
    worker_count=config['grain_worker_count'],
  )
  return iter(dataloader)

def data_load_loop(config):
  # mesh = None
  # process_indices = None
  data_iterator = dummy_grain_iterator(config)
  start = datetime.datetime.now()
  batch = next(data_iterator)
  first_end = datetime.datetime.now()
  time_to_load_first_batch = first_end - start
  print(f"DUMMY DATALOADER : First step completed in {time_to_load_first_batch.seconds} seconds")

  for _ in np.arange(1, config["steps"]):
    batch = next(data_iterator)
  end = datetime.datetime.now()
  print(f"DUMMY DATALOADER : {config['steps']} batches loaded in {(end-start).seconds} seconds")

def main(argv: Sequence[str]) -> None:
  # pyconfig.initialize(argv)
  # config = pyconfig.config
  data_load_loop(config)

if __name__ == "__main__":
  app.run(main)
