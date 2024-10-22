From gcr.io/tpu-prod-env-multipod/maxtext_jax_stable_stack_0.4.33
RUN pip3 uninstall -y grain-nightly orbax-checkpoint && \
    gsutil cp gs://mlperf-exp-us-east1-cp0/grain-wheel/grain-0.2.2-cp310-cp310-linux_x86_64.whl ./ && \
    pip3 install grain-0.2.2-cp310-cp310-linux_x86_64.whl orbax-checkpoint==0.6.0
