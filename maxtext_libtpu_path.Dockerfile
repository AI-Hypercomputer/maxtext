ARG BASEIMAGE=maxtext_base_image
FROM $BASEIMAGE

ENV MAXTEXT_ASSETS_ROOT=/deps/src/MaxText/assets
ENV MAXTEXT_PKG_DIR=/deps/src/MaxText
ENV MAXTEXT_REPO_ROOT=/deps
ENV MAXTEXT_TEST_ASSETS_ROOT=/deps/src/MaxText/test_assets
ENV MAXTEXT_VENV=/deps/venvs/maxtext_venv

ENV VIRTUAL_ENV="${MAXTEXT_VENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

#FROM maxtext_base_image
# Set the TPU_LIBRARY_PATH
ENV TPU_LIBRARY_PATH='/root/custom_libtpu/libtpu.so'

WORKDIR /deps
