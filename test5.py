# Create source nnx state

from MaxText import model_creation_utils
from MaxText import pyconfig
import os
from MaxText.globals import MAXTEXT_PKG_DIR

model_name = "gpt-oss-20b"


argv = [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"), f"model_name={model_name}"]
config = pyconfig.initialize(argv)

config = pyconfig.initialize(argv) # figure it out!
model, _ = model_creation_utils.create_nnx_model(config)

#model -> nnx state
breakpoint()
source_state = model.state #I think 

flat_source_state = source_state.flat_state()
for key,val in flat_source_state:
    print(".".join(str(k) for k in key))