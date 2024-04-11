ARG BASEIMAGE=maxtext_base_image
FROM $BASEIMAGE

#FROM maxtext_base_image
# Set the TPU_LIBRARY_PATH
ENV TPU_LIBRARY_PATH='/root/custom_libtpu/libtpu.so'

WORKDIR /deps