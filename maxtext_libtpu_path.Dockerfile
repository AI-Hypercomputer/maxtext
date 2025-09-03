ARG BASEIMAGE=src/MaxText_base_image
FROM $BASEIMAGE

ENV MAXTEXT_ASSETS_ROOT=/deps/assets
ENV MAXTEXT_PKG_DIR=/deps/src/MaxText
ENV MAXTEXT_REPO_ROOT=/deps

#FROM src/MaxText_base_image
# Set the TPU_LIBRARY_PATH
ENV TPU_LIBRARY_PATH='/root/custom_libtpu/libtpu.so'

WORKDIR /deps
