import os
from MaxText.gcloud_stub import is_decoupled
from MaxText.globals import MAXTEXT_PKG_DIR

def get_test_config_path():
    """Centralized selection for returning absolute path to the test 
    config.

    If DECOUPLE_GCLOUD=TRUE, use decoupled_base_test.yml else base.yml.
    """
    base_cfg = "base.yml"
    if is_decoupled():
        base_cfg = "decoupled_base_test.yml"
    return os.path.join(MAXTEXT_PKG_DIR, "configs", base_cfg)

__all__ = ["get_test_config_path"]
