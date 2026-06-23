FROM gcr.io/tpu-prod-env-multipod/maxtext_post_training_nightly:2026-06-23
COPY . /app
WORKDIR /app
