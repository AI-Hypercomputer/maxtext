# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


r"""This is to convert checkpoint weight from linen to nnx structure.

Usage:

python -m tools/weight_inspector/convert_linen_checkpoint_to_nnx.py --source_path="/original-model/runner_direct_1/checkpoints/14/"  --output_path="/converted-model/runner_direct_1/checkpoints/14/"

"""

import jax
import orbax.checkpoint as ocp
from typing import Any, Dict
import numpy as np
import sys
import argparse # Import argparse
from etils import epath
import pprint



def load_full_checkpoint(checkpoint_dir: epath.Path) -> Dict[str, Any] | None:
    """Loads the entire PyTree checkpoint using Orbax."""
    items_path = checkpoint_dir / 'items'
    print(f"Loading full checkpoint from: {items_path}")
    if not items_path.exists():
        print(f"Error: Checkpoint items not found: {items_path}")
        return None
    try:
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        restored_object = orbax_checkpointer.restore(items_path)
        print(f"Successfully restored full checkpoint from {items_path}")
        return restored_object
    except Exception as e:
        print(f"An error occurred during checkpoint restoration: {e}")
        return None

def wrap_array_leaves(tree: Any) -> Any:
    """Recursively wraps only JAX/NumPy array leaf nodes in {'value': array} format."""
    def _wrap(leaf):
        if isinstance(leaf, (jax.Array, np.ndarray)):
            return {'value': leaf}
        return leaf  # Keep scalars as they are
    return jax.tree_util.tree_map(_wrap, tree)

def main(args):
    source_path = epath.Path(args.source_path)
    output_path = epath.Path(args.output_path)

    print(f"--- Converting Checkpoint ---")
    print(f"  Source (V1 - main): {source_path}")
    print(f"  Output (V2 - modelspy format): {output_path}")

    restored_main = load_full_checkpoint(source_path)
    if restored_main is None:
        sys.exit(1)

    if 'params' not in restored_main or 'params' not in restored_main['params']:
        print("Error: Expected structure {'params': {'params': ...}} not found in source.")
        sys.exit(1)

    # 1. Extract the core parameters from the main model
    main_core_params = restored_main['params']['params']
    # Wrap only the array leaves within the core parameters
    nnx_style_core_params = wrap_array_leaves(main_core_params)

    # 2. Process opt_state: Wrap only array leaves
    if 'opt_state' in restored_main:
        new_opt_state = wrap_array_leaves(restored_main['opt_state'])
    else:
        new_opt_state = None
        print("Warning: 'opt_state' not found in source checkpoint.")

    # 3. Construct the new state to save, matching the modelspy structure
    state_to_save = {
        'params': nnx_style_core_params,
        'opt_state': new_opt_state,
        'step': restored_main.get('step'),  # Keep step as a scalar
        'graphdef': None,  # Add to match modelspy structure
    }

    print("\n--- Structure of State to Save (types) ---")
    pprint.pprint(jax.tree_util.tree_map(lambda x: type(x), state_to_save), depth=4)

    save_items_path = output_path / 'items'
    print(f"--- Saving converted checkpoint to {save_items_path} ---")

    if jax.process_index() == 0:
        output_path.mkdir(parents=True, exist_ok=True)

    # Barrier to ensure directory is created before other processes proceed
    if jax.process_count() > 1:
      jax.experimental.multihost_utils.sync_global_devices("output_dir_creation")

    checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    try:
        checkpointer.save(save_items_path, state_to_save)
        checkpointer.wait_until_finished()
        print(f"✅ Conversion complete. Saved to {save_items_path}")
    except Exception as e:
        print(f"❌ Error during saving checkpoint: {e}")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Flax checkpoint format.')
    parser.add_argument('--source_path', type=str, required=True,
                        help='Path to the source "main" model checkpoint directory (containing items/).')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the converted "modelspy" format checkpoint directory.')

    args = parser.parse_args()
    main(args)
