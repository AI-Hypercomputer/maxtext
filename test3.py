# from MaxText import checkpointing
# checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
#     #"tmp/test",
#     "gs://shuningjin-multipod-dev/test",
#     enable_checkpointing=True,
#     use_async=False,  # Synchronous saving for simplicity in conversion script
#     save_interval_steps=1,  # Save at step 0
#     use_ocdbt=True,
#     use_zarr3=True,
# )

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--hf_model_path", type=str, required=False, default="")
local_args, _ = parser.parse_known_args()
# Remove args defined in this test file to avoid error from pyconfig
model_args = [s for s in sys.argv if not s.startswith("--hf_model_path")]
print(model_args)
# main(model_args, local_args)
print(local_args)
print(type(local_args) == argparse.Namespace)
print(local_args.hf_model_path)
