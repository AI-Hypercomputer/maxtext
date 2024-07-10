from typing import Sequence
import datetime
import pyconfig
from train import create_data_iterator
from input_pipeline._grain_data_processing import make_grain_iterator
import numpy as np
from absl import app

# def setup_dataloader(config):
#   mesh = None
#   data_iterator, eval_data_iterator = create_data_iterator(config, mesh)


def data_load_loop(config):
  mesh = None
  process_indices = None
  data_iterator, _ = make_grain_iterator(config, mesh, True, True, process_indices)
  start = datetime.datetime.now()
  batch = next(data_iterator)
  first_end = datetime.datetime.now()
  time_to_load_first_batch = first_end - start
  print(f"DUMMY DATALOADER : First step completed in {time_to_load_first_batch.seconds} seconds")

  for _ in np.arange(1, config.steps):
    batch = next(data_iterator)
  end = datetime.datetime.now()
  print(f"DUMMY DATALOADER : {config.steps} batches loaded in {(end-start).seconds} seconds")

def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  config = pyconfig.config
  data_load_loop(config)
 
if __name__ == "__main__":
  app.run(main)
