
import functools
import importlib
import os
from functools import lru_cache

from .. import_utils import _torch_available


@lru_cache
def is_torch_mlu_available(check_device: bool = False) -> bool:
  """
  Checks if `mlu` is available. JAX does not support MLUs, so this always returns False.
  """
  # JAX does not support MLU devices.
  return False

from typing import Any

from .import_utils import is_flax_available


def is_jax_tensor(x: Any) -> bool:
  """Tests if `x` is a Jax tensor or not.

  Safe to call even if jax is not installed.
  """
  if not is_flax_available():
    return False
  import jax.numpy as jnp

  return isinstance(x, jnp.ndarray)
LIBROSA_IMPORT_ERROR = """
{0} requires the librosa library. But that was not found in your environment. You can install them with pip:
`pip install librosa`
Please note that you may need to restart your runtime after installation.
"""NLTK_IMPORT_ERROR = """
{0} requires the NLTK library but it was not found in your environment. You can install it by referring to:
https://www.nltk.org/install.html. Please note that you may need to restart your runtime after installation.
"""
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Import utilities: Utilities related to imports and our lazy inits.
"""

import importlib.machinery
import importlib.metadata
import importlib.util
import json
import operator
import os
import re
import shutil
import subprocess
import sys
import warnings
from collections import OrderedDict
from enum import Enum
from functools import lru_cache
from itertools import chain
from types import ModuleType
from typing import Any, Optional, Union

from packaging import version

from MaxText import max_logging as logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# TODO: This doesn't work for all packages (`bs4`, `faiss`, etc.) Talk to Sylvain to see how to do with it better.
def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[tuple[bool, str], bool]:
    # Check if the package spec exists and grab its version to avoid importing a local directory
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            if pkg_name == "triton":
                try:
                    # import triton works for both linux and windows
                    package = importlib.import_module(pkg_name)
                    package_version = getattr(package, "__version__", "N/A")
                except Exception:
                    package_exists = False
            else:
                # For packages other than "triton", don't attempt the fallback and set as not available
                package_exists = False
        logger.debug(f"Detected {pkg_name} version: {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_JAX = os.environ.get("USE_JAX", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()


FORCE_TF_AVAILABLE = os.environ.get("FORCE_TF_AVAILABLE", "AUTO").upper()

ACCELERATE_MIN_VERSION = "0.26.0"
SCHEDULEFREE_MIN_VERSION = "1.2.6"
GGUF_MIN_VERSION = "0.10.0"
HQQ_MIN_VERSION = "0.2.1"
VPTQ_MIN_VERSION = "0.0.4"
TORCHAO_MIN_VERSION = "0.4.0"
AUTOROUND_MIN_VERSION = "0.5.0"
TRITON_MIN_VERSION = "1.0.0"

_accelerate_available, _accelerate_version = _is_package_available("accelerate", return_version=True)
_apex_available = _is_package_available("apex")
_apollo_torch_available = _is_package_available("apollo_torch")
_aqlm_available = _is_package_available("aqlm")
_vptq_available, _vptq_version = _is_package_available("vptq", return_version=True)
_av_available = importlib.util.find_spec("av") is not None
_decord_available = importlib.util.find_spec("decord") is not None
_torchcodec_available = importlib.util.find_spec("torchcodec") is not None
_libcst_available = _is_package_available("libcst")
_bitsandbytes_available = _is_package_available("bitsandbytes")
_eetq_available = _is_package_available("eetq")
_fbgemm_gpu_available = _is_package_available("fbgemm_gpu")
_galore_torch_available = _is_package_available("galore_torch")
_lomo_available = _is_package_available("lomo_optim")
_grokadamw_available = _is_package_available("grokadamw")
_schedulefree_available, _schedulefree_version = _is_package_available("schedulefree", return_version=True)
_torch_optimi_available = importlib.util.find_spec("optimi") is not None
# `importlib.metadata.version` doesn't work with `bs4` but `beautifulsoup4`. For `importlib.util.find_spec`, reversed.
_bs4_available = importlib.util.find_spec("bs4") is not None
_coloredlogs_available = _is_package_available("coloredlogs")
# `importlib.metadata.util` doesn't work with `opencv-python-headless`.
_cv2_available = importlib.util.find_spec("cv2") is not None
_yt_dlp_available = importlib.util.find_spec("yt_dlp") is not None
_datasets_available = _is_package_available("datasets")
_detectron2_available = _is_package_available("detectron2")
# We need to check `faiss`, `faiss-cpu` and `faiss-gpu`.
_faiss_available = importlib.util.find_spec("faiss") is not None
try:
    _faiss_version = importlib.metadata.version("faiss")
    logger.debug(f"Successfully imported faiss version {_faiss_version}")
except importlib.metadata.PackageNotFoundError:
    try:
        _faiss_version = importlib.metadata.version("faiss-cpu")
        logger.debug(f"Successfully imported faiss version {_faiss_version}")
    except importlib.metadata.PackageNotFoundError:
        try:
            _faiss_version = importlib.metadata.version("faiss-gpu")
            logger.debug(f"Successfully imported faiss version {_faiss_version}")
        except importlib.metadata.PackageNotFoundError:
            _faiss_available = False
_ftfy_available = _is_package_available("ftfy")
_g2p_en_available = _is_package_available("g2p_en")
_hadamard_available = _is_package_available("fast_hadamard_transform")
_jieba_available = _is_package_available("jieba")
_jinja_available = _is_package_available("jinja2")
_kenlm_available = _is_package_available("kenlm")
_keras_nlp_available = _is_package_available("keras_nlp")
_levenshtein_available = _is_package_available("Levenshtein")
_librosa_available = _is_package_available("librosa")
_natten_available = _is_package_available("natten")
_nltk_available = _is_package_available("nltk")
_onnx_available = _is_package_available("onnx")
_openai_available = _is_package_available("openai")
_optimum_available = _is_package_available("optimum")
_auto_gptq_available = _is_package_available("auto_gptq")
_gptqmodel_available = _is_package_available("gptqmodel")
_auto_round_available, _auto_round_version = _is_package_available("auto_round", return_version=True)
# `importlib.metadata.version` doesn't work with `awq`
_auto_awq_available = importlib.util.find_spec("awq") is not None
_quark_available = _is_package_available("quark")
_fp_quant_available, _fp_quant_version = _is_package_available("fp_quant", return_version=True)
_qutlass_available = _is_package_available("qutlass")
_is_optimum_quanto_available = False
try:
    importlib.metadata.version("optimum_quanto")
    _is_optimum_quanto_available = True
except importlib.metadata.PackageNotFoundError:
    _is_optimum_quanto_available = False
# For compressed_tensors, only check spec to allow compressed_tensors-nightly package
_compressed_tensors_available = importlib.util.find_spec("compressed_tensors") is not None
_pandas_available = _is_package_available("pandas")
_peft_available = _is_package_available("peft")
_phonemizer_available = _is_package_available("phonemizer")
_uroman_available = _is_package_available("uroman")
_psutil_available = _is_package_available("psutil")
_py3nvml_available = _is_package_available("py3nvml")
_pyctcdecode_available = _is_package_available("pyctcdecode")
_pygments_available = _is_package_available("pygments")
_pytesseract_available = _is_package_available("pytesseract")
_pytest_available = _is_package_available("pytest")
_pytorch_quantization_available = _is_package_available("pytorch_quantization")
_rjieba_available = _is_package_available("rjieba")
_sacremoses_available = _is_package_available("sacremoses")
_safetensors_available = _is_package_available("safetensors")
_scipy_available = _is_package_available("scipy")
_sentencepiece_available = _is_package_available("sentencepiece")
_is_seqio_available = _is_package_available("seqio")
_is_gguf_available, _gguf_version = _is_package_available("gguf", return_version=True)
_sklearn_available = importlib.util.find_spec("sklearn") is not None
if _sklearn_available:
    try:
        importlib.metadata.version("scikit-learn")
    except importlib.metadata.PackageNotFoundError:
        _sklearn_available = False
_smdistributed_available = importlib.util.find_spec("smdistributed") is not None
_soundfile_available = _is_package_available("soundfile")
_spacy_available = _is_package_available("spacy")
_sudachipy_available, _sudachipy_version = _is_package_available("sudachipy", return_version=True)
_tensorflow_probability_available = _is_package_available("tensorflow_probability")
_tensorflow_text_available = _is_package_available("tensorflow_text")
_tf2onnx_available = _is_package_available("tf2onnx")
_timm_available = _is_package_available("timm")
_tokenizers_available = _is_package_available("tokenizers")
_torchaudio_available = _is_package_available("torchaudio")
_torchao_available, _torchao_version = _is_package_available("torchao", return_version=True)
_torchdistx_available = _is_package_available("torchdistx")
_torchvision_available, _torchvision_version = _is_package_available("torchvision", return_version=True)
_mlx_available = _is_package_available("mlx")
_num2words_available = _is_package_available("num2words")
_hqq_available, _hqq_version = _is_package_available("hqq", return_version=True)
_tiktoken_available = _is_package_available("tiktoken")
_blobfile_available = _is_package_available("blobfile")
_liger_kernel_available = _is_package_available("liger_kernel")
_spqr_available = _is_package_available("spqr_quant")
_rich_available = _is_package_available("rich")
_kernels_available = _is_package_available("kernels")
_matplotlib_available = _is_package_available("matplotlib")
_mistral_common_available = _is_package_available("mistral_common")
_triton_available, _triton_version = _is_package_available("triton", return_version=True)

_torch_available = False
if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES:
    _torch_available, _ = _is_package_available("torch", return_version=True)
    if not _torch_available:
        logger.info("Disabling PyTorch because it is not installed.")
else:
    logger.info("Disabling PyTorch because USE_TORCH is not set.")


_tf_version = "N/A"
_tf_available = False
if FORCE_TF_AVAILABLE in ENV_VARS_TRUE_VALUES:
    _tf_available = True
else:
    if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
        _tf_available = importlib.util.find_spec("tensorflow") is not None
        if _tf_available:
            candidates = (
                "tensorflow",
                "tensorflow-cpu",
                "tensorflow-gpu",
                "tf-nightly",
                "tf-nightly-cpu",
                "tf-nightly-gpu",
                "tf-nightly-rocm",
                "intel-tensorflow",
                "intel-tensorflow-avx512",
                "tensorflow-rocm",
                "tensorflow-macos",
                "tensorflow-aarch64",
            )
            _tf_version = None
            for pkg in candidates:
                try:
                    _tf_version = importlib.metadata.version(pkg)
                    break
                except importlib.metadata.PackageNotFoundError:
                    pass
            _tf_available = _tf_version is not None
        if _tf_available:
            if version.parse(_tf_version) < version.parse("2"):
                logger.info(f"TensorFlow found but with version {_tf_version}. Transformers requires version 2 minimum.")
                _tf_available = False
    else:
        logger.info("Disabling Tensorflow because USE_TORCH is set or USE_TF is not set.")


_essentia_available = importlib.util.find_spec("essentia") is not None
try:
    _essentia_version = importlib.metadata.version("essentia")
    logger.debug(f"Successfully imported essentia version {_essentia_version}")
except importlib.metadata.PackageNotFoundError:
    _essentia_version = False


_pydantic_available = importlib.util.find_spec("pydantic") is not None
try:
    _pydantic_version = importlib.metadata.version("pydantic")
    logger.debug(f"Successfully imported pydantic version {_pydantic_version}")
except importlib.metadata.PackageNotFoundError:
    _pydantic_available = False


_fastapi_available = importlib.util.find_spec("fastapi") is not None
try:
    _fastapi_version = importlib.metadata.version("fastapi")
    logger.debug(f"Successfully imported pydantic version {_fastapi_version}")
except importlib.metadata.PackageNotFoundError:
    _fastapi_available = False


_uvicorn_available = importlib.util.find_spec("uvicorn") is not None
try:
    _uvicorn_version = importlib.metadata.version("uvicorn")
    logger.debug(f"Successfully imported pydantic version {_uvicorn_version}")
except importlib.metadata.PackageNotFoundError:
    _uvicorn_available = False


_pretty_midi_available = importlib.util.find_spec("pretty_midi") is not None
try:
    _pretty_midi_version = importlib.metadata.version("pretty_midi")
    logger.debug(f"Successfully imported pretty_midi version {_pretty_midi_version}")
except importlib.metadata.PackageNotFoundError:
    _pretty_midi_available = False


_jax_version = "N/A"
_flax_version = "N/A"
_jax_available = False
_flax_available = False
if USE_JAX in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _jax_available, _jax_version = _is_package_available("jax", return_version=True)
    if _jax_available:
        _flax_available, _flax_version = _is_package_available("flax", return_version=True)
        if _flax_available:
            logger.info(f"JAX version {_jax_version}, Flax version {_flax_version} available.")
        else:
            _jax_available = False
            _jax_version = _flax_version = "N/A"
else:
    logger.info("Disabling JAX because USE_TF is set or USE_JAX is not set.")


def is_kenlm_available():
    return _kenlm_available


def is_kernels_available():
    return _kernels_available


def is_cv2_available():
    return _cv2_available


def is_yt_dlp_available():
    return _yt_dlp_available


def is_torch_available():
    return _torch_available


def is_libcst_available():
    return _libcst_available


def is_accelerate_available(min_version: str = ACCELERATE_MIN_VERSION):
    return _accelerate_available and version.parse(_accelerate_version) >= version.parse(min_version)


def is_triton_available(min_version: str = TRITON_MIN_VERSION):
    return _triton_available and version.parse(_triton_version) >= version.parse(min_version)


def is_hadamard_available():
    return _hadamard_available


def is_hqq_available(min_version: str = HQQ_MIN_VERSION):
    return _hqq_available and version.parse(_hqq_version) >= version.parse(min_version)


def is_pygments_available():
    return _pygments_available


def get_jax_version():
    return _jax_version


def get_flax_version():
    return _flax_version


def get_jax_major_and_minor_version() -> str:
    if _jax_version == "N/A":
        return "N/A"
    parsed_version = version.parse(_jax_version)
    return str(parsed_version.major) + "." + str(parsed_version.minor)


def is_torchvision_available():
    return _torchvision_available


def is_galore_torch_available():
    return _galore_torch_available


def is_apollo_torch_available():
    return _apollo_torch_available


def is_torch_optimi_available():
    return _torch_optimi_available


def is_lomo_available():
    return _lomo_available


def is_grokadamw_available():
    return _grokadamw_available


def is_schedulefree_available(min_version: str = SCHEDULEFREE_MIN_VERSION):
    return _schedulefree_available and version.parse(_schedulefree_version) >= version.parse(min_version)


def is_pyctcdecode_available():
    return _pyctcdecode_available


def is_librosa_available():
    return _librosa_available


def is_essentia_available():
    return _essentia_available


def is_pydantic_available():
    return _pydantic_available


def is_fastapi_available():
    return _fastapi_available


def is_uvicorn_available():
    return _uvicorn_available


def is_openai_available():
    return _openai_available


def is_pretty_midi_available():
    return _pretty_midi_available


@lru_cache
def is_gpu_available():
    if not is_jax_available():
        return False
    import jax

    return len(jax.devices("gpu")) > 0


@lru_cache
def is_tpu_available():
    if not is_jax_available():
        return False
    import jax

    return len(jax.devices("tpu")) > 0


@lru_cache
def is_mps_available():
    if not is_jax_available():
        return False
    import jax

    try:
        return len(jax.devices("mps")) > 0
    except RuntimeError:
        return False


@lru_cache
def is_bf16_available():
    if not is_jax_available():
        return False
    import jax
    import jax.numpy as jnp

    try:
        # Try to create a bfloat16 array on the default device
        _ = jnp.zeros((2, 2), dtype=jnp.bfloat16)
        return True
    except (TypeError, RuntimeError):
        return False


def is_bf16_gpu_available():
    if not is_gpu_available():
        return False
    return is_bf16_available()


def is_bf16_cpu_available():
    if not is_jax_available() or is_gpu_available() or is_tpu_available():
        return is_bf16_available()
    return False


@lru_cache
def is_fp16_available_on_device(device_str: str):
    if not is_jax_available():
        return False
    import jax
    import jax.numpy as jnp

    try:
        device = jax.devices(device_str)[0]
        with jax.default_device(device):
            _ = jnp.zeros((2, 2), dtype=jnp.float16)
        return True
    except (IndexError, RuntimeError, TypeError):
        return False


def is_tf32_available():
    # TF32 is on by default in JAX on Ampere+ GPUs.
    # A simple check for GPU availability is a reasonable proxy.
    return is_gpu_available()


def is_peft_available():
    return _peft_available


def is_bs4_available():
    return _bs4_available


def is_tf_available():
    return _tf_available


def is_coloredlogs_available():
    return _coloredlogs_available


def is_tf2onnx_available():
    return _tf2onnx_available


def is_onnx_available():
    return _onnx_available


def is_jax_available():
    return _jax_available


def is_flax_available():
    return _flax_available


def is_flute_available():
    try:
        return importlib.util.find_spec("flute") is not None and importlib.metadata.version("flute-kernel") >= "0.4.1"
    except importlib.metadata.PackageNotFoundError:
        return False


def is_ftfy_available():
    return _ftfy_available


def is_g2p_en_available():
    return _g2p_en_available


@lru_cache
def is_neuron_available(check_device=True):
    if importlib.util.find_spec("jax_neuron") is None:
        return False
    if check_device:
        try:
            import jax

            return len(jax.devices("neuron")) > 0
        except (RuntimeError, ImportError):
            return False
    return True


@lru_cache
def is_npu_available(check_device=False):
    "Checks if `jax_plugins.npu` is installed and potentially if a NPU is in the environment"
    if not is_jax_available() or importlib.util.find_spec("jax_plugins.npu") is None:
        return False
    if check_device:
        try:
            import jax

            return len(jax.devices("npu")) > 0
        except RuntimeError:
            return False
    return True


def is_jax_tracer(x: Any) -> bool:
    if is_jax_available():
        import jax

        return isinstance(x, jax.core.Tracer)
    return False


def is_datasets_available():
    return _datasets_available


def is_detectron2_available():
    return _detectron2_available


def is_rjieba_available():
    return _rjieba_available


def is_psutil_available():
    return _psutil_available


def is_py3nvml_available():
    return _py3nvml_available


def is_sacremoses_available():
    return _sacremoses_available


def is_apex_available():
    return _apex_available


def is_aqlm_available():
    return _aqlm_available


def is_vptq_available(min_version: str = VPTQ_MIN_VERSION):
    return _vptq_available and version.parse(_vptq_version) >= version.parse(min_version)


def is_av_available():
    return _av_available


def is_decord_available():
    return _decord_available


def is_torchcodec_available():
    return _torchcodec_available


def is_ninja_available():
    r"""
    Code comes from *torch.utils.cpp_extension.is_ninja_available()*. Returns `True` if the
    [ninja](https://ninja-build.org/) build system is available on the system, `False` otherwise.
    """
    try:
        subprocess.check_output(["ninja", "--version"])
    except Exception:
        return False
    else:
        return True


@lru_cache
def is_bitsandbytes_available(check_library_only=False) -> bool:
    if not _bitsandbytes_available:
        return False

    if check_library_only:
        return True

    # bitsandbytes requires a GPU
    return is_gpu_available()


def is_bitsandbytes_multi_backend_available() -> bool:
    if not is_bitsandbytes_available():
        return False

    import bitsandbytes as bnb

    return "multi_backend" in getattr(bnb, "features", set())


def is_flash_attn_2_available():
    if not _is_package_available("flash_attn"):
        return False

    # Let's add an extra check to see if a GPU or TPU is available
    if not (is_gpu_available() or is_tpu_available()):
        return False

    return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.1.0")


@lru_cache
def is_flash_attn_3_available():
    if not _is_package_available("flash_attn_3"):
        return False

    if not is_gpu_available():
        return False

    return True


@lru_cache
def is_flash_attn_greater_or_equal_2_10():
    if not _is_package_available("flash_attn"):
        return False

    return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.1.0")


@lru_cache
def is_flash_attn_greater_or_equal(library_version: str):
    if not _is_package_available("flash_attn"):
        return False

    return version.parse(importlib.metadata.version("flash_attn")) >= version.parse(library_version)


@lru_cache
def is_jax_greater_or_equal(library_version: str, accept_dev: bool = False):
    """
    Accepts a library version and returns True if the current version of JAX is greater than or equal to the
    given version. If `accept_dev` is True, it will also accept development versions.
    """
    if not _is_package_available("jax"):
        return False

    current_version = version.parse(importlib.metadata.version("jax"))
    if accept_dev:
        current_version = version.parse(current_version.base_version)

    return current_version >= version.parse(library_version)


@lru_cache
def is_jax_less_or_equal(library_version: str, accept_dev: bool = False):
    """
    Accepts a library version and returns True if the current version of JAX is less than or equal to the
    given version. If `accept_dev` is True, it will also accept development versions.
    """
    if not _is_package_available("jax"):
        return False

    current_version = version.parse(importlib.metadata.version("jax"))
    if accept_dev:
        current_version = version.parse(current_version.base_version)

    return current_version <= version.parse(library_version)


@lru_cache
def is_huggingface_hub_greater_or_equal(library_version: str, accept_dev: bool = False):
    if not _is_package_available("huggingface_hub"):
        return False

    if accept_dev:
        return version.parse(
            version.parse(importlib.metadata.version("huggingface_hub")).base_version
        ) >= version.parse(library_version)
    else:
        return version.parse(importlib.metadata.version("huggingface_hub")) >= version.parse(library_version)


@lru_cache
def is_quanto_greater(library_version: str, accept_dev: bool = False):
    """
    Accepts a library version and returns True if the current version of the library is greater than or equal to the
    given version. If `accept_dev` is True, it will also accept development versions.
    """
    if not _is_package_available("optimum.quanto"):
        return False

    if accept_dev:
        return version.parse(version.parse(importlib.metadata.version("optimum-quanto")).base_version) > version.parse(
            library_version
        )
    else:
        return version.parse(importlib.metadata.version("optimum-quanto")) > version.parse(library_version)


def is_torchdistx_available():
    return _torchdistx_available


def is_faiss_available():
    return _faiss_available


def is_scipy_available():
    return _scipy_available


def is_sklearn_available():
    return _sklearn_available


def is_sentencepiece_available():
    return _sentencepiece_available


def is_seqio_available():
    return _is_seqio_available


def is_gguf_available(min_version: str = GGUF_MIN_VERSION):
    return _is_gguf_available and version.parse(_gguf_version) >= version.parse(min_version)


def is_protobuf_available():
    if importlib.util.find_spec("google") is None:
        return False
    return importlib.util.find_spec("google.protobuf") is not None


def is_optimum_available():
    return _optimum_available


def is_auto_awq_available():
    return _auto_awq_available


def is_auto_round_available(min_version: str = AUTOROUND_MIN_VERSION):
    return _auto_round_available and version.parse(_auto_round_version) >= version.parse(min_version)


def is_optimum_quanto_available():
    return _is_optimum_quanto_available


def is_quark_available():
    return _quark_available


def is_fp_quant_available():
    return _fp_quant_available and version.parse(_fp_quant_version) >= version.parse("0.1.6")


def is_qutlass_available():
    return _qutlass_available


def is_compressed_tensors_available():
    return _compressed_tensors_available


def is_auto_gptq_available():
    return _auto_gptq_available


def is_gptqmodel_available():
    return _gptqmodel_available


def is_eetq_available():
    return _eetq_available


def is_fbgemm_gpu_available():
    return _fbgemm_gpu_available


def is_levenshtein_available():
    return _levenshtein_available


def is_optimum_neuron_available():
    return _optimum_available and _is_package_available("optimum.neuron")


def is_safetensors_available():
    return _safetensors_available


def is_tokenizers_available():
    return _tokenizers_available


@lru_cache
def is_vision_available():
    _pil_available = importlib.util.find_spec("PIL") is not None
    if _pil_available:
        try:
            package_version = importlib.metadata.version("Pillow")
        except importlib.metadata.PackageNotFoundError:
            try:
                package_version = importlib.metadata.version("Pillow-SIMD")
            except importlib.metadata.PackageNotFoundError:
                return False
        logger.debug(f"Detected PIL version {package_version}")
    return _pil_available


def is_pytesseract_available():
    return _pytesseract_available


def is_pytest_available():
    return _pytest_available


def is_spacy_available():
    return _spacy_available


def is_tensorflow_text_available():
    return is_tf_available() and _tensorflow_text_available


def is_keras_nlp_available():
    return is_tensorflow_text_available() and _keras_nlp_available


def is_in_notebook():
    try:
        # Check if we are running inside Marimo
        if "marimo" in sys.modules:
            return True
        # Test adapted from tqdm.autonotebook: https://github.com/tqdm/tqdm/blob/master/tqdm/autonotebook.py
        get_ipython = sys.modules["IPython"].get_ipython
        if "IPKernelApp" not in get_ipython().config:
            raise ImportError("console")
        # Removed the lines to include VSCode
        if "DATABRICKS_RUNTIME_VERSION" in os.environ and os.environ["DATABRICKS_RUNTIME_VERSION"] < "11.0":
            # Databricks Runtime 11.0 and above uses IPython kernel by default so it should be compatible with Jupyter notebook
            # https://docs.microsoft.com/en-us/azure/databricks/notebooks/ipython-kernel
            raise ImportError("databricks")

        return importlib.util.find_spec("IPython") is not None
    except (AttributeError, ImportError, KeyError):
        return False


def is_pytorch_quantization_available():
    return _pytorch_quantization_available


def is_tensorflow_probability_available():
    return _tensorflow_probability_available


def is_pandas_available():
    return _pandas_available


def is_sagemaker_dp_enabled():
    # Get the sagemaker specific env variable.
    sagemaker_params = os.getenv("SM_FRAMEWORK_PARAMS", "{}")
    try:
        # Parse it and check the field "sagemaker_distributed_dataparallel_enabled".
        sagemaker_params = json.loads(sagemaker_params)
        if not sagemaker_params.get("sagemaker_distributed_dataparallel_enabled", False):
            return False
    except json.JSONDecodeError:
        return False
    # Lastly, check if the `smdistributed` module is present.
    return _smdistributed_available


def is_sagemaker_mp_enabled():
    # Get the sagemaker specific mp parameters from smp_options variable.
    smp_options = os.getenv("SM_HP_MP_PARAMETERS", "{}")
    try:
        # Parse it and check the field "partitions" is included, it is required for model parallel.
        smp_options = json.loads(smp_options)
        if "partitions" not in smp_options:
            return False
    except json.JSONDecodeError:
        return False

    # Get the sagemaker specific framework parameters from mpi_options variable.
    mpi_options = os.getenv("SM_FRAMEWORK_PARAMS", "{}")
    try:
        # Parse it and check the field "sagemaker_distributed_dataparallel_enabled".
        mpi_options = json.loads(mpi_options)
        if not mpi_options.get("sagemaker_mpi_enabled", False):
            return False
    except json.JSONDecodeError:
        return False
    # Lastly, check if the `smdistributed` module is present.
    return _smdistributed_available


def is_training_run_on_sagemaker():
    return "SAGEMAKER_JOB_NAME" in os.environ


def is_soundfile_available():
    return _soundfile_available


def is_timm_available():
    return _timm_available


def is_natten_available():
    return _natten_available


def is_nltk_available():
    return _nltk_available


def is_torchaudio_available():
    return _torchaudio_available


def is_torchao_available(min_version: str = TORCHAO_MIN_VERSION):
    return _torchao_available and version.parse(_torchao_version) >= version.parse(min_version)


def is_speech_available():
    # For now this depends on torchaudio but the exact dependency might evolve in the future.
    return _torchaudio_available


def is_spqr_available():
    return _spqr_available


def is_phonemizer_available():
    return _phonemizer_available


def is_uroman_available():
    return _uroman_available


def jax_only_method(fn):
    def wrapper(*args, **kwargs):
        if not _flax_available:
            raise ImportError(
                "You need to install JAX and Flax to use this method or class, "
                "or activate it with environment variables USE_JAX=1 and USE_TF=0."
            )
        else:
            return fn(*args, **kwargs)

    return wrapper


def is_sudachi_available():
    return _sudachipy_available


def get_sudachi_version():
    return _sudachipy_version


def is_sudachi_projection_available():
    if not is_sudachi_available():
        return False

    # NOTE: We require sudachipy>=0.6.8 to use projection option in sudachi_kwargs for the constructor of BertJapaneseTokenizer.
    # - `projection` option is not supported in sudachipy<0.6.8, see https://github.com/WorksApplications/sudachi.rs/issues/230
    return version.parse(_sudachipy_version) >= version.parse("0.6.8")


def is_jumanpp_available():
    return (importlib.util.find_spec("rhoknp") is not None) and (shutil.which("jumanpp") is not None)


def is_cython_available():
    return importlib.util.find_spec("pyximport") is not None


def is_jieba_available():
    return _jieba_available


def is_jinja_available():
    return _jinja_available


def is_mlx_available():
    return _mlx_available


def is_num2words_available():
    return _num2words_available


def is_tiktoken_available():
    return _tiktoken_available and _blobfile_available


def is_liger_kernel_available():
    if not _liger_kernel_available:
        return False

    return version.parse(importlib.metadata.version("liger_kernel")) >= version.parse("0.3.0")


def is_rich_available():
    return _rich_available


def is_matplotlib_available():
    return _matplotlib_available


def is_mistral_common_available():
    return _mistral_common_available


# docstyle-ignore
AV_IMPORT_ERROR = """
{0} requires the PyAv library but it was not found in your environment. You can install it with:
TIMM_IMPORT_ERROR = """
{0} requires the timm library but it was not found in your environment. You can install it with pip:
`pip install timm`. Please note that you may need to restart your runtime after installation.
"""
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Import utilities: Utilities related to imports and our lazy inits.
"""

# Assuming _datasets_available is defined in the same module, similar to the PyTorch version.
# This variable would be set based on checking for the 'datasets' package availability.
# For example:
#
# import importlib.util
# _datasets_available = importlib.util.find_spec("datasets") is not None


def is_datasets_available() -> bool:
  """Check if the `datasets` library is available."""
  return _datasets_available

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Import utilities: Utilities related to imports and our lazy inits.
"""


def is_essentia_available() -> bool:
  return _essentia_available

import importlib.util
import importlib.metadata
from MaxText import max_logging

logger = max_logging.get_logger(__name__)

_fastapi_available = importlib.util.find_spec("fastapi") is not None
if _fastapi_available:
  try:
    _fastapi_version = importlib.metadata.version("fastapi")
    logger.debug(f"Successfully imported fastapi version {_fastapi_version}")
  except importlib.metadata.PackageNotFoundError:
    _fastapi_available = False


def is_fastapi_available() -> bool:
  """Check if FastAPI is available."""
  return _fastapi_available

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Import utilities: Utilities related to imports and our lazy inits.
"""


def is_ftfy_available() -> bool:
  """Check if ftfy is available."""
  return _ftfy_available

# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Import utilities: Utilities related to imports and our lazy inits.
"""


def is_levenshtein_available() -> bool:
  return _levenshtein_available

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Import utilities: Utilities related to imports and our lazy inits.
"""


def is_openai_available() -> bool:
  """Check if openai is available."""
  return _openai_available

# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Import utilities: Utilities related to imports and our lazy inits.
"""
# Note: This function relies on a global variable `_peft_available`
# which is not part of this code block. It is assumed to be defined
# elsewhere in the file, for instance:
# from . import _is_package_available
# _peft_available = _is_package_available("peft")


def is_peft_available() -> bool:
  """Check if the peft library is available."""
  return _peft_available

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Import utilities: Utilities related to imports and our lazy inits.
"""


def is_pyctcdecode_available() -> bool:
  """Check if pyctcdecode is available."""
  return _pyctcdecode_available

# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Import utilities: Utilities related to imports and our lazy inits.
"""

# Assuming _rich_available is defined at the module level,
# similar to the original PyTorch file.
# e.g.,
# import importlib.util
# _rich_available = importlib.util.find_spec("rich") is not None


def is_rich_available() -> bool:
  """Checks if the 'rich' library is available."""
  return _rich_available

import importlib.util
import importlib.metadata

# Global variable to cache the availability of sacremoses
_sacremoses_available: bool

def _is_package_available(pkg_name: str) -> bool:
  """
  Check if a package is available and installed, not just a local directory.
  This is a simplified version of the utility in the source file.
  """
  package_exists = importlib.util.find_spec(pkg_name) is not None
  if package_exists:
    try:
      importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
      package_exists = False
  return package_exists

_sacremoses_available = _is_package_available("sacremoses")

def is_sacremoses_available() -> bool:
  """Returns True if sacremoses is available, False otherwise."""
  return _sacremoses_available

def is_torchvision_available() -> bool:
  """Returns True if torchvision is available."""
  return _torchvision_available
PYTORCH_IMPORT_ERROR_WITH_FLAX = """
{0} requires the PyTorch library but it was not found in your environment.
However, we were able to find a JAX/Flax installation. Flax classes begin
with "Flax", but are otherwise identically named to our PyTorch classes. This
means that the Flax equivalent of the class you tried to import would be "Flax{0}".
If you want to use Flax, please use Flax classes instead!

If you really do want to use PyTorch please go to
https://pytorch.org/get-started/locally/ and follow the instructions that
match your environment.
"""
# Copyright 2024 The MaxText Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Import utilities: Utilities related to imports and our lazy inits.
"""

import importlib.machinery
import importlib.metadata
import importlib.util
import logging
from typing import Union


logger = logging.getLogger(__name__)


# TODO: This doesn't work for all packages (`bs4`, `faiss`, etc.)
def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[tuple[bool, str], bool]:
  """
  Checks if a package is available and optionally returns its version.

  Args:
    pkg_name: The name of the package to check.
    return_version: Whether to return the package version along with the availability.

  Returns:
    If return_version is True, a tuple of (bool, str) representing availability and version.
    Otherwise, a bool representing availability.
  """
  # Check if the package spec exists and grab its version to avoid importing a local directory
  package_exists = importlib.util.find_spec(pkg_name) is not None
  package_version = "N/A"
  if package_exists:
    try:
      # TODO: Once python 3.9 support is dropped, `importlib.metadata.packages_distributions()`
      # should be used here to map from package name to distribution names
      # e.g. PIL -> Pillow, Pillow-SIMD; quark -> amd-quark; onnxruntime -> onnxruntime-gpu.
      # `importlib.metadata.packages_distributions()` is not available in Python 3.9.

      # Primary method to get the package version
      package_version = importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
      # Fallback method: Only for "torch" and versions containing "dev"
      if pkg_name == "torch":
        try:
          package = importlib.import_module(pkg_name)
          temp_version = getattr(package, "__version__", "N/A")
          # Check if the version contains "dev"
          if "dev" in temp_version:
            package_version = temp_version
            package_exists = True
          else:
            package_exists = False
        except ImportError:
          # If the package can't be imported, it's not available
          package_exists = False
      elif pkg_name == "quark":
        # TODO: remove once `importlib.metadata.packages_distributions()` is supported.
        try:
          package_version = importlib.metadata.version("amd-quark")
        except Exception:
          package_exists = False
      elif pkg_name == "triton":
        try:
          # import triton works for both linux and windows
          package = importlib.import_module(pkg_name)
          package_version = getattr(package, "__version__", "N/A")
        except Exception:
          try:
            package_version = importlib.metadata.version("pytorch-triton")  # pytorch-triton
          except Exception:
            package_exists = False
      else:
        # For packages other than "torch", don't attempt the fallback and set as not available
        package_exists = False
    logger.debug(f"Detected {pkg_name} version: {package_version}")
  if return_version:
    return package_exists, package_version
  else:
    return package_exists


_phonemizer_available = _is_package_available("phonemizer")

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Import utilities: Utilities related to imports and our lazy inits.
"""

# Assuming _phonemizer_available is defined at the module level,
# similar to the original PyTorch file.
# from .import_utils import _phonemizer_available


def is_phonemizer_available() -> bool:
  """Checks if the 'phonemizer' library is installed and available."""
  return _phonemizer_available

import importlib.machinery
import importlib.metadata
import importlib.util
import logging
from typing import Union

logger = logging.getLogger(__name__)


def _is_package_available(
    pkg_name: str, return_version: bool = False
) -> Union[tuple[bool, str], bool]:
  """
  Checks if a package is available and optionally returns its version.

  This function checks for the presence of a package and its version without
  importing a local directory. It includes specific fallbacks for packages
  like torch, quark, and triton.

  Args:
      pkg_name: The name of the package to check.
      return_version: If True, returns a tuple of (bool, version_str).
                      Otherwise, returns a bool.

  Returns:
      A boolean indicating if the package is available, or a tuple of
      (availability, version).
  """
  # Check if the package spec exists and grab its version to avoid importing a local directory
  package_exists = importlib.util.find_spec(pkg_name) is not None
  package_version = "N/A"
  if package_exists:
    try:
      # Primary method to get the package version
      package_version = importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
      # Fallback method: Only for "torch" and versions containing "dev"
      if pkg_name == "torch":
        try:
          package = importlib.import_module(pkg_name)
          temp_version = getattr(package, "__version__", "N/A")
          # Check if the version contains "dev"
          if "dev" in temp_version:
            package_version = temp_version
            package_exists = True
          else:
            package_exists = False
        except ImportError:
          # If the package can't be imported, it's not available
          package_exists = False
      elif pkg_name == "quark":
        try:
          package_version = importlib.metadata.version("amd-quark")
        except Exception:
          package_exists = False
      elif pkg_name == "triton":
        try:
          # import triton works for both linux and windows
          package = importlib.import_module(pkg_name)
          package_version = getattr(package, "__version__", "N/A")
        except Exception:
          try:
            package_version = importlib.metadata.version(
                "pytorch-triton"
            )  # pytorch-triton
          except Exception:
            package_exists = False
      else:
        # For packages other than "torch", don't attempt the fallback and set as not available
        package_exists = False
    logger.debug(f"Detected {pkg_name} version: {package_version}")
  if return_version:
    return package_exists, package_version
  else:
    return package_exists


_scipy_available = _is_package_available("scipy")

import importlib.util
import importlib.metadata


# A JAX-equivalent helper function to check for package availability.
def _is_package_available(pkg_name: str) -> bool:
  """Check if a package is available."""
  package_exists = importlib.util.find_spec(pkg_name) is not None
  if package_exists:
    try:
      importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
      package_exists = False
  return package_exists


_scipy_available: bool = _is_package_available("scipy")


def is_scipy_available() -> bool:
  """Returns True if SciPy is available."""
  return _scipy_available

import importlib
import importlib.metadata
import importlib.util
from typing import Union

from packaging import version

# Assuming a MaxText-style logging import
from MaxText import max_logging as logging


logger = logging.get_logger(__name__)


def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[tuple[bool, str], bool]:
  """
  Checks if a package is available and optionally returns its version.
  This is a utility function that does not rely on any specific framework (JAX/PyTorch)
  and can be used in a general Python environment.
  """
  # Check if the package spec exists and grab its version to avoid importing a local directory
  package_exists = importlib.util.find_spec(pkg_name) is not None
  package_version = "N/A"
  if package_exists:
    try:
      # Primary method to get the package version
      package_version = importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
      # Fallback method for specific cases
      if pkg_name == "torch":
        try:
          package = importlib.import_module(pkg_name)
          temp_version = getattr(package, "__version__", "N/A")
          # Check if the version contains "dev"
          if "dev" in temp_version:
            package_version = temp_version
            package_exists = True
          else:
            package_exists = False
        except ImportError:
          # If the package can't be imported, it's not available
          package_exists = False
      elif pkg_name == "quark":
        try:
          package_version = importlib.metadata.version("amd-quark")
        except Exception:
          package_exists = False
      elif pkg_name == "triton":
        try:
          # import triton works for both linux and windows
          package = importlib.import_module(pkg_name)
          package_version = getattr(package, "__version__", "N/A")
        except Exception:
          try:
            package_version = importlib.metadata.version("pytorch-triton")  # pytorch-triton
          except Exception:
            package_exists = False
      else:
        # For other packages, don't attempt the fallback and set as not available
        package_exists = False
    logger.debug(f"Detected {pkg_name} version: {package_version}")
  if return_version:
    return package_exists, package_version
  else:
    return package_exists


_torchaudio_available = _is_package_available("torchaudio")

from typing import Any

from .import_utils import is_tf_available


def is_tf_tensor(x: Any) -> bool:
  """Tests if `x` is a tensorflow tensor or not.

  Safe to call even if tensorflow is not installed.
  """
  return False if not is_tf_available() else _is_tensorflow(x)

import importlib.util

# Simplified check for package availability.
# The original Hugging Face code includes complex logic for version checks and
# environment variables (USE_TF, USE_TORCH) which are not relevant in a
# pure JAX context.
_tf_available = importlib.util.find_spec("tensorflow") is not None
_tensorflow_text_available = importlib.util.find_spec("tensorflow_text") is not None


def is_tf_available() -> bool:
  """Check if TensorFlow is available."""
  return _tf_available


def is_tensorflow_text_available() -> bool:
  """Check if TensorFlow Text is available."""
  return is_tf_available() and _tensorflow_text_available

from collections import OrderedDict

# It is assumed that the following functions and constants are defined elsewhere in the JAX file,
# mirroring the structure of the original PyTorch file.
from . import (
    ACCELERATE_IMPORT_ERROR,
    AV_IMPORT_ERROR,
    BS4_IMPORT_ERROR,
    CCL_IMPORT_ERROR,
    CV2_IMPORT_ERROR,
    CYTHON_IMPORT_ERROR,
    DATASETS_IMPORT_ERROR,
    DECORD_IMPORT_ERROR,
    DETECTRON2_IMPORT_ERROR,
    ESSENTIA_IMPORT_ERROR,
    FAISS_IMPORT_ERROR,
    FASTAPI_IMPORT_ERROR,
    FLAX_IMPORT_ERROR,
    FTFY_IMPORT_ERROR,
    G2P_EN_IMPORT_ERROR,
    JIEBA_IMPORT_ERROR,
    JINJA_IMPORT_ERROR,
    KERAS_NLP_IMPORT_ERROR,
    LEVENSHTEIN_IMPORT_ERROR,
    LIBROSA_IMPORT_ERROR,
    MISTRAL_COMMON_IMPORT_ERROR,
    NATTEN_IMPORT_ERROR,
    NLTK_IMPORT_ERROR,
    OPENAI_IMPORT_ERROR,
    PANDAS_IMPORT_ERROR,
    PEFT_IMPORT_ERROR,
    PHONEMIZER_IMPORT_ERROR,
    PRETTY_MIDI_IMPORT_ERROR,
    PROTOBUF_IMPORT_ERROR,
    PYCTCDECODE_IMPORT_ERROR,
    PYDANTIC_IMPORT_ERROR,
    PYTESSERACT_IMPORT_ERROR,
    PYTORCH_IMPORT_ERROR,
    PYTORCH_QUANTIZATION_IMPORT_ERROR,
    RICH_IMPORT_ERROR,
    SACREMOSES_IMPORT_ERROR,
    SCIPY_IMPORT_ERROR,
    SENTENCEPIECE_IMPORT_ERROR,
    SKLEARN_IMPORT_ERROR,
    SPEECH_IMPORT_ERROR,
    TENSORFLOW_IMPORT_ERROR,
    TENSORFLOW_PROBABILITY_IMPORT_ERROR,
    TENSORFLOW_TEXT_IMPORT_ERROR,
    TIMM_IMPORT_ERROR,
    TOKENIZERS_IMPORT_ERROR,
    TORCHAUDIO_IMPORT_ERROR,
    TORCHCODEC_IMPORT_ERROR,
    TORCHVISION_IMPORT_ERROR,
    UROMAN_IMPORT_ERROR,
    UVICORN_IMPORT_ERROR,
    VISION_IMPORT_ERROR,
    YT_DLP_IMPORT_ERROR,
    is_accelerate_available,
    is_av_available,
    is_bs4_available,
    is_ccl_available,
    is_cv2_available,
    is_cython_available,
    is_datasets_available,
    is_decord_available,
    is_detectron2_available,
    is_essentia_available,
    is_faiss_available,
    is_fastapi_available,
    is_flax_available,
    is_ftfy_available,
    is_g2p_en_available,
    is_jieba_available,
    is_jinja_available,
    is_keras_nlp_available,
    is_levenshtein_available,
    is_librosa_available,
    is_mistral_common_available,
    is_natten_available,
    is_nltk_available,
    is_openai_available,
    is_pandas_available,
    is_peft_available,
    is_phonemizer_available,
    is_pretty_midi_available,
    is_protobuf_available,
    is_pyctcdecode_available,
    is_pydantic_available,
    is_pytesseract_available,
    is_pytorch_quantization_available,
    is_rich_available,
    is_sacremoses_available,
    is_scipy_available,
    is_sentencepiece_available,
    is_sklearn_available,
    is_speech_available,
    is_tensorflow_probability_available,
    is_tensorflow_text_available,
    is_tf_available,
    is_timm_available,
    is_tokenizers_available,
    is_torch_available,
    is_torchaudio_available,
    is_torchcodec_available,
    is_torchvision_available,
    is_uroman_available,
    is_uvicorn_available,
    is_vision_available,
    is_yt_dlp_available,
)


BACKENDS_MAPPING = OrderedDict(
    [
        ("av", (is_av_available, AV_IMPORT_ERROR)),
        ("bs4", (is_bs4_available, BS4_IMPORT_ERROR)),
        ("cv2", (is_cv2_available, CV2_IMPORT_ERROR)),
        ("datasets", (is_datasets_available, DATASETS_IMPORT_ERROR)),
        ("decord", (is_decord_available, DECORD_IMPORT_ERROR)),
        ("detectron2", (is_detectron2_available, DETECTRON2_IMPORT_ERROR)),
        ("essentia", (is_essentia_available, ESSENTIA_IMPORT_ERROR)),
        ("faiss", (is_faiss_available, FAISS_IMPORT_ERROR)),
        ("flax", (is_flax_available, FLAX_IMPORT_ERROR)),
        ("ftfy", (is_ftfy_available, FTFY_IMPORT_ERROR)),
        ("g2p_en", (is_g2p_en_available, G2P_EN_IMPORT_ERROR)),
        ("pandas", (is_pandas_available, PANDAS_IMPORT_ERROR)),
        ("phonemizer", (is_phonemizer_available, PHONEMIZER_IMPORT_ERROR)),
        ("uroman", (is_uroman_available, UROMAN_IMPORT_ERROR)),
        ("pretty_midi", (is_pretty_midi_available, PRETTY_MIDI_IMPORT_ERROR)),
        ("levenshtein", (is_levenshtein_available, LEVENSHTEIN_IMPORT_ERROR)),
        ("librosa", (is_librosa_available, LIBROSA_IMPORT_ERROR)),
        ("protobuf", (is_protobuf_available, PROTOBUF_IMPORT_ERROR)),
        ("pyctcdecode", (is_pyctcdecode_available, PYCTCDECODE_IMPORT_ERROR)),
        ("pytesseract", (is_pytesseract_available, PYTESSERACT_IMPORT_ERROR)),
        ("sacremoses", (is_sacremoses_available, SACREMOSES_IMPORT_ERROR)),
        (
            "pytorch_quantization",
            (
                is_pytorch_quantization_available,
                PYTORCH_QUANTIZATION_IMPORT_ERROR,
            ),
        ),
        (
            "sentencepiece",
            (is_sentencepiece_available, SENTENCEPIECE_IMPORT_ERROR),
        ),
        ("sklearn", (is_sklearn_available, SKLEARN_IMPORT_ERROR)),
        ("speech", (is_speech_available, SPEECH_IMPORT_ERROR)),
        (
            "tensorflow_probability",
            (
                is_tensorflow_probability_available,
                TENSORFLOW_PROBABILITY_IMPORT_ERROR,
            ),
        ),
        ("tf", (is_tf_available, TENSORFLOW_IMPORT_ERROR)),
        (
            "tensorflow_text",
            (is_tensorflow_text_available, TENSORFLOW_TEXT_IMPORT_ERROR),
        ),
        ("timm", (is_timm_available, TIMM_IMPORT_ERROR)),
        ("torchaudio", (is_torchaudio_available, TORCHAUDIO_IMPORT_ERROR)),
        ("natten", (is_natten_available, NATTEN_IMPORT_ERROR)),
        ("nltk", (is_nltk_available, NLTK_IMPORT_ERROR)),
        ("tokenizers", (is_tokenizers_available, TOKENIZERS_IMPORT_ERROR)),
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        ("torchvision", (is_torchvision_available, TORCHVISION_IMPORT_ERROR)),
        ("torchcodec", (is_torchcodec_available, TORCHCODEC_IMPORT_ERROR)),
        ("vision", (is_vision_available, VISION_IMPORT_ERROR)),
        ("scipy", (is_scipy_available, SCIPY_IMPORT_ERROR)),
        ("accelerate", (is_accelerate_available, ACCELERATE_IMPORT_ERROR)),
        ("oneccl_bind_pt", (is_ccl_available, CCL_IMPORT_ERROR)),
        ("cython", (is_cython_available, CYTHON_IMPORT_ERROR)),
        ("jieba", (is_jieba_available, JIEBA_IMPORT_ERROR)),
        ("peft", (is_peft_available, PEFT_IMPORT_ERROR)),
        ("jinja", (is_jinja_available, JINJA_IMPORT_ERROR)),
        ("yt_dlp", (is_yt_dlp_available, YT_DLP_IMPORT_ERROR)),
        ("rich", (is_rich_available, RICH_IMPORT_ERROR)),
        ("keras_nlp", (is_keras_nlp_available, KERAS_NLP_IMPORT_ERROR)),
        ("pydantic", (is_pydantic_available, PYDANTIC_IMPORT_ERROR)),
        ("fastapi", (is_fastapi_available, FASTAPI_IMPORT_ERROR)),
        ("uvicorn", (is_uvicorn_available, UVICORN_IMPORT_ERROR)),
        ("openai", (is_openai_available, OPENAI_IMPORT_ERROR)),
        (
            "mistral-common",
            (is_mistral_common_available, MISTRAL_COMMON_IMPORT_ERROR),
        ),
    ]
)
