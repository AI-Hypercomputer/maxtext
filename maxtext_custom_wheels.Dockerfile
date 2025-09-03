ARG BASEIMAGE=src/MaxText_base_image
FROM $BASEIMAGE

# Requires wheels be in /deps. This means any custom wheels should be placed
# in the src/MaxText directory.
RUN python3 -m pip install --force-reinstall /deps/*.whl
