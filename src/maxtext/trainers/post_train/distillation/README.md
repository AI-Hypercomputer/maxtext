# MaxText distillation trainer

Distillation from a large teacher LLM into a (optionally pruned) student on
TPU, via JAX/MaxText on top of Tunix's PEFT trainer. This README is
operational — for concepts, see
[`knowledge_distillation.md`](../../../../../docs/tutorials/posttraining/knowledge_distillation.md)
and [`post_training_index.md`](../../../../../docs/tutorials/post_training_index.md).

Canonical launcher: [`scripts/run_distill_xpk.sh`](scripts/run_distill_xpk.sh)
(see its header for all env vars).


## 1. Pick a config

Configs live in [`src/maxtext/configs/post_train/`](../../../configs/post_train/):

| File | Student | Teacher | Notes |
|---|---|---|---|
| `distillation.yml` | llama3.1-8b | llama3.1-8b | Baseline |
| `distillation-sft.yml` | llama3.1-8b | llama3.1-8b | Distillation + SFT mix |

**Override `num_epoch` to a value > 1 if you want to train for more than one
epoch** (e.g. `num_epoch=10`) — the base default is 1, and the input pipeline
(Grain) iterates the dataset only once before stopping, so longer runs run
out of data mid-training. Pass it as a CLI override (`… distillation.yml num_epoch=10`) or
edit the YAML directly.

## 2. Single-host smoke test

Validate your config + checkpoint paths on a single TPU VM (no xpk, no GKE)
before scaling to a cluster. The default `distillation.yml` (llama3.1-8b
student + 8b teacher) needs a slice large enough to hold both models in
HBM — ≥ v5p-16 in practice; a v5p-8 only fits with bf16 weights or a
shrunken student (see below):

```bash
source <your-venv>/bin/activate
PYTHONPATH=$PWD/src python -m maxtext.trainers.post_train.distillation.train_distill \
  src/maxtext/configs/post_train/distillation.yml \
  run_name=local_smoke \
  base_output_directory=gs://<bucket>/distill_smoke \
  steps=5
```

Smaller TPU? Shrink the model with overrides like `base_emb_dim=...
base_num_decoder_layers=... base_num_query_heads=...` — pass
`override_model_config=True` so the CLI overrides actually take effect
(default is `False`).


## 3. Cluster auth

```bash
pip install git+https://github.com/AI-Hypercomputer/xpk.git

# Kubeconfig (use --dns-endpoint; IP endpoints are often stale).
# Use --zone for zonal clusters, --region for regional ones.
gcloud container clusters get-credentials <cluster> \
  --zone=<zone> --project=<project> --dns-endpoint

# Verify RBAC in the default namespace:
kubectl auth can-i create roles --namespace=default   # must print: yes
```

If `can-i` prints `no`, ask a cluster admin to bind
`roles/container.admin` to your user.

## 4. Build + push the image (one time)

The flow is: build the MaxText base → `prep_image` rebuilds `$XPK_BASE_IMAGE`
**in place** (same local tag) with tunix layered on top → `docker_upload_runner.sh`
pushes that modified local tag to GCR under `$CLOUD_IMAGE_NAME`.

```bash
# Local tag prep_image rebuilds; registry name docker_upload_runner pushes to.
export XPK_BASE_IMAGE=maxtext_base_image:stable
export CLOUD_IMAGE_NAME=gcr.io/<your-project>/maxtext_base_image:stable

# Base image with MaxText + TPU deps.
sudo bash src/dependencies/scripts/docker_build_dependency_image.sh \
  MODE=stable WORKFLOW=post-training

# Layer tunix + re-pin jax/libtpu for libtpu compat. Rebuilds $XPK_BASE_IMAGE in place.
bash src/maxtext/trainers/post_train/distillation/scripts/run_distill_xpk.sh prep_image

# Push the modified local tag to GCR so later submits pull from the registry (no buildx).
sudo bash src/dependencies/scripts/docker_upload_runner.sh \
  CLOUD_IMAGE_NAME=${CLOUD_IMAGE_NAME}
```

## 5. Submit

```bash
export XPK_CLUSTER=<cluster>
export XPK_PROJECT=<project>
export XPK_ZONE=<zone>
export XPK_DEVICE_TYPE=tpu7x-4x4x4
export XPK_BASE_IMAGE=${CLOUD_IMAGE_NAME} # slash in name → --docker-image auto-selected
export XPK_BASE_OUTPUT_DIR=gs://<bucket>/distillation
export XPK_RUN_NAME=<experiment>          # default: distill_run; set per experiment
                                          # to scope checkpoints + TB under
                                          # ${XPK_BASE_OUTPUT_DIR}/${XPK_WORKLOAD}/${XPK_RUN_NAME}/

bash src/maxtext/trainers/post_train/distillation/scripts/run_distill_xpk.sh submit
```

The launcher writes the workload name to `~/.xpk_last_workload` for §6/§7.
Override `XPK_WORKLOAD` if you want; keep it ≲16 chars — some clusters
cap derived resource names at 49. See the script header for other env
vars (`DISTILL_ALPHA`, `DISTILL_BETA`, `STEPS_OVERRIDE`, etc.).

## 6. Monitor

```bash
WL=$(cat ~/.xpk_last_workload)
POD=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=$WL,batch.kubernetes.io/job-completion-index=0 -o name | head -1)
kubectl logs -f ${POD} -c jax-tpu-1 | grep "Train step"
```

## 7. Resume

Submit again with the **same `XPK_BASE_OUTPUT_DIR` + `XPK_WORKLOAD` + `XPK_RUN_NAME`** —
checkpoints live at `${XPK_BASE_OUTPUT_DIR}/${XPK_WORKLOAD}/${XPK_RUN_NAME}/checkpoints/`,
and `maybe_restore` picks up the latest one. All three must match the
previous submit (the launcher writes the workload name to `~/.xpk_last_workload`).
For auto-retry:

```bash
STEPS_OVERRIDE=108000 \
  bash src/maxtext/trainers/post_train/distillation/scripts/run_distill_xpk.sh resume_until_done
```

## 8. Checkpoint retention

| Field | Default | Description |
|---|---|---|
| `checkpoint_period` | 2000 | Save a checkpoint every N training steps |
| `max_num_checkpoints_to_keep` | `None` | Keep the most recent N checkpoints. `None` keeps all |

