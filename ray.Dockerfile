FROM rayproject/ray:2.37.0-py310

USER root

WORKDIR /maxtext
COPY requirements.txt /maxtext
RUN pip install -r /maxtext/requirements.txt
RUN pip install cryptography gitpython memray py-spy jupyter
RUN pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
