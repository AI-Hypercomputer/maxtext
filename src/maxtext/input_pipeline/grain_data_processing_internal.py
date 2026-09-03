"""Internal wrapper for grain_data_processing."""

import tensorflow as tf
from maxtext.input_pipeline import grain_data_processing
from maxtext.utils import max_logging

_original_find_data_files = grain_data_processing.find_data_files

def _patched_find_data_files(data_file_pattern, hf_access_token=None):
    if data_file_pattern.startswith("/cns/") or data_file_pattern.startswith("/bigstore/"):
        data_files = tf.io.gfile.glob(data_file_pattern)
        if not data_files:
            raise FileNotFoundError(f"No files found matching internal pattern: {data_file_pattern}")
        max_logging.log(f"Found {len(data_files)} files for train/eval with grain from internal path")
        return data_files
    return _original_find_data_files(data_file_pattern, hf_access_token)

grain_data_processing.find_data_files = _patched_find_data_files
from maxtext.input_pipeline.grain_data_processing import *
