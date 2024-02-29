import os
from contextlib import contextmanager


try:
  import transformer_engine.jax as te
  from transformer_engine.common import recipe
  _IS_TRANSFORMER_ENGINE_INSTALLED = True

except ModuleNotFoundError as e:
  _IS_TRANSFORMER_ENGINE_INSTALLED = False


class TransformerEngineHelperBase:

    @staticmethod
    @contextmanager
    def fp8_autocast(dp_mesh_axis="data", tp_mesh_axis="tensor", fsdp_mesh_axis="fsdp"):
        raise NotImplementedError

class TENotInstalledHelper(TransformerEngineHelperBase):

    @staticmethod
    @contextmanager
    def fp8_autocast(dp_mesh_axis="data", tp_mesh_axis="tensor", fsdp_mesh_axis="fsdp"):
        try:
            yield
        finally:
            pass
    
    @staticmethod
    def extend_logical_axis_rules(rules):
        return rules

class TEInstalledHelper(TransformerEngineHelperBase):

    @staticmethod
    @contextmanager
    def fp8_autocast(dp_mesh_axis="data", tp_mesh_axis="tensor", fsdp_mesh_axis="fsdp"):
        fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.HYBRID,
                                           amax_history_len=1024, amax_compute_algo='max')
        enable_fp8 = bool(int((os.environ.get("ENABLE_FP8", False))))
        try:
            with te.fp8_autocast(enabled=enable_fp8,
                                 fp8_recipe=fp8_recipe,
                                 mesh_resource=te.MeshResource(dp_resource=dp_mesh_axis,
                                                               tp_resource=tp_mesh_axis,
                                                               fsdp_resource=fsdp_mesh_axis)):
                yield
        finally:
            pass

    @staticmethod
    def extend_logical_axis_rules(rules):
        # Apply fp8_autocast to correctly set sharding_resource up.
        with TEInstalledHelper.fp8_autocast():
            return te.flax.extend_logical_axis_rules(rules)

class TransformerEngineHelper(TransformerEngineHelperBase):

    @staticmethod
    def is_enabled_te():
        enable_te = bool(int((os.environ.get("ENABLE_TE", False))))
        return (_IS_TRANSFORMER_ENGINE_INSTALLED and enable_te)

    @staticmethod
    def get_helper():
        if TransformerEngineHelper.is_enabled_te():
            return TEInstalledHelper
        return TENotInstalledHelper

    @staticmethod
    @contextmanager
    def fp8_autocast(dp_mesh_axis="data", tp_mesh_axis="tensor", fsdp_mesh_axis="fsdp"):
        try:
            with TransformerEngineHelper.get_helper().fp8_autocast(dp_mesh_axis, tp_mesh_axis, fsdp_mesh_axis):
                yield
        finally:
            pass

    @staticmethod
    def extend_logical_axis_rules(rules):
        # Apply fp8_autocast to correctly set sharding_resource up.
        with TEInstalledHelper.fp8_autocast():
            return te.flax.extend_logical_axis_rules(rules)
