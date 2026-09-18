import unittest
from types import SimpleNamespace
from maxtext.utils.train_utils import validate_train_config

class TestIndexerZeroObjectiveValidator(unittest.TestCase):

  def test_zero_objective_raises_value_error(self):
    config = SimpleNamespace(
        run_name="test_run",
        steps=10,
        dataset_path="",
        base_output_directory="gs://test",
        quantization="",
        packing=False,
        dataset_type="synthetic",
        use_indexer=True,
        indexer_sparse_training=False,
        indexer_loss_scaling_factor=0.0,
    )
    with self.assertRaisesRegex(ValueError, "zeroes the entire training objective"):
      validate_train_config(config)

  def test_valid_indexer_config(self):
    config = SimpleNamespace(
        run_name="test_run",
        steps=10,
        dataset_path="",
        base_output_directory="gs://test",
        quantization="",
        packing=False,
        dataset_type="synthetic",
        use_indexer=True,
        indexer_sparse_training=True,
        indexer_loss_scaling_factor=0.0,
    )
    validate_train_config(config)

    config2 = SimpleNamespace(
        run_name="test_run",
        steps=10,
        dataset_path="",
        base_output_directory="gs://test",
        quantization="",
        packing=False,
        dataset_type="synthetic",
        use_indexer=True,
        indexer_sparse_training=False,
        indexer_loss_scaling_factor=1.0,
    )
    validate_train_config(config2)

  def test_use_indexer_false(self):
    config = SimpleNamespace(
        run_name="test_run",
        steps=10,
        dataset_path="",
        base_output_directory="gs://test",
        quantization="",
        packing=False,
        dataset_type="synthetic",
        use_indexer=False,
        indexer_sparse_training=False,
        indexer_loss_scaling_factor=0.0,
    )
    validate_train_config(config)

if __name__ == '__main__':
  unittest.main()
