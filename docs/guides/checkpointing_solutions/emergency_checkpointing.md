# Emergency checkpointing

Emergency checkpointing is a vital feature for large-scale, multi-slice training. It enables rapid saving and restoration of model state from local, in-memory checkpoints in response to hardware failures, host errors, or preemptions. This feature becomes increasingly critical as the number of hosts and devices grows, which raises the probability of a failure.

## Assumptions

- **GKE Environment**: A **Google Kubernetes Engine (GKE)** cluster must be used. GCE infrastructure solutions like QueuedResources are not supported.
- **Multi-Tier Checkpointing Enabled on GKE cluster level**: The Multi-Tier Checkpointing feature must be enabled and configured on your GKE cluster. This involves setting up the necessary CSI drivers and configurations as outlined in the [Google Cloud Checkpointing Documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/training/multi-tier-checkpointing).
- **Multi-Slice Workload**: The training job must be a [multi-slice environment](https://cloud.google.com/kubernetes-engine/docs/how-to/tpu-multislice), meaning it utilizes more than one node pool.
- **Orbax Checkpointer**: The [Orbax library](https://orbax.readthedocs.io) must be used for checkpointing in your training script.
- **Ramdisk Mounted via Jobset**: Each workload pod must have a [ramdisk directory mounted by Jobset](https://cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/training/multi-tier-checkpointing#update-jobset) using the Multi-Tier Checkpointing CSI driver. This provides a high-speed, in-memory storage location for checkpoints.
- **Supported TPU types**: [v4](https://cloud.google.com/tpu/docs/v4), [v5e](https://cloud.google.com/tpu/docs/v5e), [v5p](https://cloud.google.com/tpu/docs/v5p), and [v6e](https://cloud.google.com/tpu/docs/v6e)
- **Cluster version**: GKE cluster version needs to be later than [1.32.3-gke.1170000](https://cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/training/multi-tier-checkpointing#existing-cluster).

## Configure the GKE cluster

To run workloads with Emergency Checkpointing, use a Google Kubernetes Engine (GKE) cluster with the necessary drivers and features enabled. Follow the [Google Cloud Checkpointing Documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/training/multi-tier-checkpointing) to configure Multi-Tier Checkpointing at the cluster level.

The cluster-level settings that must be enabled are:

| Setting                       | Description                                                              |
| :---------------------------- | :----------------------------------------------------------------------- |
| **Multi-Tier Checkpointing**  | Enables the required cluster-level checkpointing feature (`HighScaleCheckpointing` addon). |
| **GCS FUSE CSI driver**       | Installs the required GCS FUSE CSI driver (`GcsFuseCsiDriver` addon). |
| **Ramdisk allocation**        | Dynamically provisioned by the MTC CSI driver on each workload pod when `--gke-mtc-enabled` and `--gke-mtc-ramdisk-dir` are passed to `gcluster job submit`. |
| **GCS bucket access**         | Grants the GKE checkpointing service account IAM permissions on your Cloud Storage bucket. |

### Calculating ramdisk size per host

The total size of a full training checkpoint (including model weights and optimizer state) can be estimated based on the number of model parameters.
A good rule of thumb:
**Total Checkpoint Size ≈ Number of Parameters × 12 bytes**

For example, a 1 billion parameter model would require approximately **1B × 12 bytes = 12 GB** for a full checkpoint.

In a distributed training environment, the checkpoint is **sharded**, or split, across all the hosts in a slice. Each host is only responsible for saving its portion of the total checkpoint. Therefore, the ramdisk on a single pod only needs to be large enough for its local shard.

The formula is:
**Required Ramdisk Size per Pod ≈ 2 × (Total Checkpoint Size / Number of Hosts in the Slice)**

It's a good practice to add a **10-15% buffer**.

### Example calculation

Let's walk through an example for a large model.

- **Model**: A 70 billion parameter language model.
- **Training Slice**: A nodepool with **32 hosts**.

1. **Estimate Total Checkpoint Size**:
   `70,000,000,000 parameters × 12 bytes/parameter = 840,000,000,000 bytes`
   `840,000,000,000 bytes ≈ 840 GB`

2. **Calculate Per-Host Checkpoint shard**:
   `(Total Checkpoint Size / 32 hosts) = 26.25 GB per host`

3. **Calculate Per-Host Ramdisk Size**:
   `(Per-Host Checkpoint shard) × 2 = 52.50 GB per host`

4. **Add a Safety Buffer (e.g., 15%)**:
   `(Per-Host Ramdisk Size) × 1.15 ≈ 60.3 GB`

In this scenario, you should configure each pod in that slice with a ramdisk of at least **60 GB**.

### Cluster configuration values

1. **Set up environment variables:**
   ```bash
   PROJECT_ID="<project-id>"
   CLUSTER_LOCATION="<cluster-location>" # example: europe-west4 (region) or us-central1-a (zone)
   BUCKET_LOCATION="<bucket-location>"   # example: europe-west4 or us-central1 (must be a region or multi-region, not a zone)
   TPU_ZONE="<tpu-zone>"                 # example: europe-west4-a
   CLUSTER_NAME="<cluster-name>"
   NODE_POOL_NAME="<tpu-node-pool-name>" # example: v6e-pool
   COMPUTE_TYPE="<tpu-machine-type>"     # example: ct6e-standard-4t
   TOPOLOGY="<tpu-topology>"             # example: 8x16 (for 32 hosts with ct6e-standard-4t) or 4x8 (8 hosts)
   GKE_VERSION="<gke-version>"           # example: 1.32.4-gke.1415000 (minimum for new clusters)
   GCS_BUCKET="<gcs-bucket-name>"        # example: my-checkpoint-bucket
   OUTPUT_PATH="gs://${GCS_BUCKET}/checkpoints"
   ```
2. **Configure gcloud and Cloud Storage:**
   Configure `gcloud` defaults and create a Cloud Storage bucket with **Hierarchical Namespace (HNS)** enabled. HNS provides fast atomic folder renames required for checkpoint finalization:
   ```bash
   gcloud config set project ${PROJECT_ID?}
   gcloud config set compute/zone ${TPU_ZONE?}

   gcloud storage buckets create gs://${GCS_BUCKET?} \
     --location=${BUCKET_LOCATION?} \
     --hierarchical-namespace
   ```
3. **Configure the cluster:** Multi-Tier Checkpointing requires the `HighScaleCheckpointing` and `GcsFuseCsiDriver` addons, as well as Workload Identity Federation, to be enabled on your GKE cluster.

   - **For an existing cluster**, update the cluster workload pool and addons in two separate commands (since `--workload-pool` and `--update-addons` cannot be specified together in `gcloud`):
     ```bash
     # Step 1: Enable Workload Identity Federation
     gcloud container clusters update ${CLUSTER_NAME?} \
       --workload-pool=${PROJECT_ID?}.svc.id.goog \
       --location=${CLUSTER_LOCATION?}

     # Step 2: Enable MTC and GCS FUSE addons
     gcloud container clusters update ${CLUSTER_NAME?} \
       --update-addons=HighScaleCheckpointing=ENABLED,GcsFuseCsiDriver=ENABLED \
       --location=${CLUSTER_LOCATION?}
     ```

   - **For a new cluster**, include the addons and workload pool during cluster creation:
     ```bash
     gcloud container clusters create ${CLUSTER_NAME?} \
       --workload-pool=${PROJECT_ID?}.svc.id.goog \
       --addons=HighScaleCheckpointing,GcsFuseCsiDriver \
       --location=${CLUSTER_LOCATION?} \
       --cluster-version=${GKE_VERSION?}
     ```

   - **Verify that HighScaleCheckpointing is enabled**:
     ```bash
     gcloud container clusters describe ${CLUSTER_NAME?} \
       --location=${CLUSTER_LOCATION?} \
       --format="yaml(addonsConfig.highScaleCheckpointingConfig)"
     ```
     The output should confirm `enabled: true`.

     > **Note:** If `HighScaleCheckpointing` is not enabled, Cluster Toolkit (`gcluster job submit`) will reject the workload submission with:
     > `Error: Multi-Tier Checkpointing (MTC) requires the HighScaleCheckpointing addon to be enabled on the target GKE cluster.`

   - **Create the TPU node pool**: If your cluster does not already have a TPU node pool, create one with the target compute type and topology:
     ```bash
     gcloud container node-pools create ${NODE_POOL_NAME?} \
       --cluster=${CLUSTER_NAME?} \
       --location=${CLUSTER_LOCATION?} \
       --node-locations=${TPU_ZONE?} \
       --machine-type=${COMPUTE_TYPE?} \
       --tpu-topology=${TOPOLOGY?}
     ```

   - **Authenticate kubectl**: Authenticate `kubectl` with the cluster credentials:
     ```bash
     gcloud container clusters get-credentials ${CLUSTER_NAME?} \
       --location=${CLUSTER_LOCATION?} \
       --project=${PROJECT_ID?}
     ```

4. **Grant access to Cloud Storage buckets:**
   - **GKE Checkpointing Service Account**: Grant `roles/storage.objectUser` on the bucket to the GKE checkpointing daemon:
     ```bash
     PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID?} --format="value(projectNumber)")
     gcloud storage buckets add-iam-policy-binding gs://${GCS_BUCKET?} \
       --member="principal://iam.googleapis.com/projects/${PROJECT_NUMBER?}/locations/global/workloadIdentityPools/${PROJECT_ID?}.svc.id.goog/subject/ns/gke-managed-checkpointing/sa/gke-checkpointing-multitier-node" \
       --role="roles/storage.objectUser"
     ```
   - **Workload Service Account**: Under `enable_emergency_checkpoint=True`, persistent checkpoints are written directly by the MaxText training pod. Grant `roles/storage.objectUser` to the Kubernetes service account (KSA) used by your training workload pod (e.g. `default` in namespace `default`):
     ```bash
     WORKLOAD_NAMESPACE="default"
     WORKLOAD_KSA="default"
     gcloud storage buckets add-iam-policy-binding gs://${GCS_BUCKET?} \
       --member="principal://iam.googleapis.com/projects/${PROJECT_NUMBER?}/locations/global/workloadIdentityPools/${PROJECT_ID?}.svc.id.goog/subject/ns/${WORKLOAD_NAMESPACE?}/sa/${WORKLOAD_KSA?}" \
       --role="roles/storage.objectUser"
     ```

## MaxText configuration

MaxText provides a set of configuration flags to control checkpointing options. This configuration manages a `two-tiered checkpointing` system designed for both durability and rapid recovery.

- **Local Emergency Checkpoints**: It saves checkpoints much more frequently to a fast, local directory on each host (i.e., a ramdisk). If a preemption or failure occurs, the job can restore from this recent local copy, minimizing lost work without needing to download from slower persistent storage. This feature is enabled by setting `enable_checkpointing`, `enable_emergency_checkpoint`, `local_checkpoint_directory` and a non-zero `local_checkpoint_period`.

- **Persistent Checkpoints**: These are standard checkpoints saved periodically and much more rarely to durable storage (i.e., GCS bucket). They ensure that you can recover your training state even after a complete cluster failure. This is controlled by `enable_checkpointing`, and `checkpoint_period`.

| Flag                                   | Description                                                                                                                                                                                                                                                                                              | Type      | Default |
| :------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------- | :------ |
| `enable_checkpointing`                 | A master switch to enable (`True`) or disable (`False`) saving checkpoints during the training run.                                                                                                                                                                                                      | `boolean` | `True`  |
| `enable_emergency_checkpoint`          | When set to (`True`), this flag enables the two-tiered emergency checkpointing feature.                                                                                                                                                                                                                  | `boolean` | `False` |
| `async_checkpointing`                  | When set to (`True`), this flag makes checkpoint saving asynchronous. The training step is only blocked for the minimal time needed to capture the model's state, and the actual writing to storage happens in a background thread. This is highly recommended for performance. It's enabled by default. | `boolean` | `True`  |
| `local_checkpoint_directory`           | The high-speed local filesystem path (i.e., ramdisk) where **emergency checkpoints** are saved. Setting this path, along with a non-zero `local_checkpoint_period`, enables the emergency checkpointing feature.                                                                                          | `string`  | `""`    |
| `local_checkpoint_period`              | The interval, in training steps, for how often a **local checkpoint** is saved. This should be set to a much smaller value than `checkpoint_period` for frequent, low-overhead saves.                                                                                                                    | `integer` | `0`     |
| `checkpoint_period`                    | The interval, in training steps, for how often a checkpoint is saved to **persistent storage**.                                                                                                                                                                                                          | `integer` | `10000` |
| `enable_single_replica_ckpt_restoring` | If `True`, one replica reads the checkpoint from storage and then broadcasts it to all other replicas. This can significantly speed up restoration on multi-host systems by reducing redundant reads from storage.                                                                                       | `boolean` | `False` |
| `enable_autocheckpoint`                | If `True`, enables saving a checkpoint when a preemption signal (SIGTERM) is received. This is a reactive mechanism that saves to persistent storage.                                                                                                                                                    | `boolean` | `False` |

### Autocheckpoint vs. Emergency Checkpointing

While both features aim to protect against progress loss, they operate differently:

- **Autocheckpoint (`enable_autocheckpoint`)**: A **reactive** mechanism. When the infrastructure sends a `SIGTERM` signal (indicating imminent preemption or maintenance), MaxText immediately attempts to save a checkpoint to persistent storage (GCS). It is best for handling planned maintenance or preemptions where a short grace period is provided.
- **Emergency Checkpointing (`enable_emergency_checkpoint`)**: A **proactive** mechanism. It saves checkpoints very frequently to local, high-speed storage (ramdisk). If a failure occurs *without* warning, the job can recover from the most recent local checkpoint. It is best for handling sudden hardware failures.

For maximum reliability, both features can be enabled simultaneously.

## Workload submission using Cluster Toolkit

The Cluster Toolkit workload must mount the ramdisk so the training process can access it:

| Flag                  | Description                                                                                                                                                                     |
| :-------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--gke-mtc-enabled`     | Enables the Multi-Tier Checkpointing feature for the workload pods.                                                                      |
| `--gke-mtc-ramdisk-dir` | Specifies the mount path inside each pod where the high-speed ramdisk will be accessible.                                                   |

### Example Cluster Toolkit workload submission

1. **Set up environment variables:**

   ```bash
   PROJECT_ID="<project-id>"
   CLUSTER_NAME="<cluster-name>"
   CLUSTER_LOCATION="<cluster-location>" # example: europe-west4 (region) or us-central1-a (zone)
   RAMDISK_DIRECTORY="<your-ramdisk-directory>" # example: /tmp/ramdisk
   WORKLOAD_NAME="<workload-name>"
   NUM_SLICES=1 # number of slices
   LOCAL_CHECKPOINT_PERIOD=10
   CHECKPOINT_PERIOD="<checkpoint-period>"
   STEPS="<steps>"
   OUTPUT_PATH="<gcs-bucket-output-path>"
   COMPUTE_TYPE="<compute-type>" # example: ct6e-standard-4t
   TOPOLOGY="<tpu-topology>"     # example: 8x16 or 4x8
   DATA_PATH="<dataset-path>"    # optional: only required if dataset_type is not synthetic
   ```

2. **Define the Docker image:**

   ```bash
   # Official release pre-training image (recommended)
   DOCKER_IMAGE="us-docker.pkg.dev/cloud-tpu-images/maxtext-images/tpu_pre_training:0.2.4"
   # Or your custom runner image:
   # DOCKER_IMAGE="${CLUSTER_LOCATION}-docker.pkg.dev/${PROJECT_ID}/<repo>/${USER}_mtc_runner:latest"
   ```

3. **Run the workload creation command:**

   ```bash
   gcloud container clusters get-credentials ${CLUSTER_NAME?} \
     --location=${CLUSTER_LOCATION?} \
     --project=${PROJECT_ID?}

   gcluster config set project ${PROJECT_ID?}
   gcluster config set cluster ${CLUSTER_NAME?}
   gcluster config set location ${CLUSTER_LOCATION?}

   gcluster job submit \
     --image=${DOCKER_IMAGE?} \
     --name=${WORKLOAD_NAME?} \
     --compute-type=${COMPUTE_TYPE?} \
     --topology=${TOPOLOGY?} \
     --num-slices=${NUM_SLICES?} \
     --gke-mtc-enabled \
     --gke-mtc-ramdisk-dir=${RAMDISK_DIRECTORY?} \
     --command "python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml run_name=${WORKLOAD_NAME?} base_output_directory=${OUTPUT_PATH?} model_name=default dataset_type=synthetic steps=${STEPS?} per_device_batch_size=6 checkpoint_period=${CHECKPOINT_PERIOD?} enable_emergency_checkpoint=True local_checkpoint_period=${LOCAL_CHECKPOINT_PERIOD?} local_checkpoint_directory=${RAMDISK_DIRECTORY?} num_slices=${NUM_SLICES?}"
   ```
