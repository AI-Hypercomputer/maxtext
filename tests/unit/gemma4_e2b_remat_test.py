import os
import tempfile

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from maxtext.configs import pyconfig
from maxtext.trainers.pre_train.train_compile import main as train_compile_main

def test_gemma4_e2b_remat():
    temp_dir = tempfile.gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_remat_gemma4_e2b.pickle")
    hlo_dir = os.path.join(temp_dir, "hlo_dump_gemma4_e2b")
    
    if not os.path.exists(hlo_dir):
        os.makedirs(hlo_dir)
    
    # Run the compile step for gemma4-e2b with a remat policy
    train_compile_main((
        "",
        "src/maxtext/configs/models/gemma4-e2b.yml",
        f"compiled_trainstep_file={compiled_trainstep_file}",
        "compile_topology=v5e-1",
        "compile_topology_num_slices=1",
        "per_device_batch_size=1",
        "ici_fsdp_parallelism=1",
        "ici_tensor_parallelism=1",
        "max_target_length=16",
        "base_num_decoder_layers=10",
        "num_kv_shared_layers=5",
        "vocab_size=128",
        "vocab_size_per_layer_input=128",
        "remat_policy=save_dot_except_mlpwi",
        "dataset_type=synthetic",
        "model_name=gemma4-e2b",
        "scan_layers=False",
        "override_model_config=True",
        f'compile_xla_flags=--xla_dump_to={hlo_dir} --xla_dump_hlo_as_text=True',
    ))
    
    # Now we need to inspect the dump or simply check if remat is applied
    assert os.path.exists(compiled_trainstep_file)
    print("Compilation succeeded")

    # Find the dumped train_step HLO text
    dumped_hlo = None
    for f in os.listdir(hlo_dir):
        if f.endswith(".txt") and ("jit_train_step" in f or "jit_train_step" in f):
            dumped_hlo = os.path.join(hlo_dir, f)
            break
            
    assert dumped_hlo is not None, "Failed to capture HLO dump"
    
    with open(dumped_hlo, "r") as f:
        hlo_content = f.read()

    # The fix ensures this exists
    assert "rematted_computation" in hlo_content, "Remat bypass bug! No rematted_computation found in HLO."
    print("SUCCESS: Gemma 4 E2B successfully generated a rematted_computation block!")
    
if __name__ == "__main__":
    test_gemma4_e2b_remat()
