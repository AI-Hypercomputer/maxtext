# Emergency Checkpointing

Emergency checkpointing is a vital feature for large-scale, multi-slice training. It enables rapid saving and restoration of model state from local, in-memory checkpoints in response to hardware failures, host errors, or preemptions. This feature becomes increasingly critical as the number of hosts and devices grows, which raises the probability of a failure.

## Assumptions

* **GKE Environment**: A **Google Kubernetes Engine (GKE)** cluster must be used. GCE infrastructure solutions like QueuedResources are not supported.
* **Multi-Tier Checkpointing Enabled on GKE cluster level**: The Multi-Tier Checkpointing feature must be enabled and configured on your GKE cluster. This involves setting up the necessary CSI drivers and configurations as outlined in the [Google Cloud Checkpointing Documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/training/multi-tier-checkpointing).
* **Multi-Slice Workload**: The training job must be a [multi-slice environment](https://cloud.google.com/kubernetes-engine/docs/how-to/tpu-multislice), meaning it utilizes more than one node pool.
* **Orbax Checkpointer**: The [Orbax library](https://orbax.readthedocs.io) must be used for checkpointing in your training script.
* **Ramdisk Mounted via Jobset**: Each workload pod must have a [ramdisk directory mounted by Jobset](https://cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/training/multi-tier-checkpointing#update-jobset) using the Multi-Tier Checkpointing CSI driver. This provides a high-speed, in-memory storage location for checkpoints.
* **Supported TPU types**: [v4](https://cloud.google.com/tpu/docs/v4), [v5e](https://cloud.google.com/tpu/docs/v5e), [v5p](https://cloud.google.com/tpu/docs/v5p), and [v6e](https://cloud.google.com/tpu/docs/v6e)

## Cluster Creation using XPK

To run workloads with Emergency Checkpointing, you need a Google Kubernetes Engine (GKE) cluster with the necessary drivers and features enabled. You can create a properly configured cluster using the **XPK** or by setting it up manually with `gcloud` commands following [Google Cloud Checkpointing Documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/training/multi-tier-checkpointing).


The `xpk` script provides a streamlined way to create a GKE cluster with all the required MTC settings. The key flags used are:

* `--enable-mtc`: Enables the Multi-Tier Checkpointing feature.
* `--enable-gcsfuse-csi-driver`: Installs the required GCS FUSE CSI driver.
* `--mtc-ramdisk-size`: Allocates an in-memory ramdisk on each node for fast, local checkpoints.
* `--mtc-gcs-bucket`: Specifies the GCS bucket. It is not utilized in emergency checkpointing, but is needed to deploy checkpointing configurations.


### Calculating Ramdisk Size Per Host 

The total size of a full training checkpoint (including model weights and optimizer state) can be estimated based on the number of model parameters.
A good rule of thumb:
**Total Checkpoint Size ≈ Number of Parameters × 12 bytes**

For example, a 1 billion parameter model would require approximately **1B × 12 bytes = 12 GB** for a full checkpoint.

In a distributed training environment, the checkpoint is **sharded**, or split, across all the hosts in a slice. Each host is only responsible for saving its portion of the total checkpoint. Therefore, the ramdisk on a single pod only needs to be large enough for its local shard.

The formula is:
**Required Ramdisk Size per Pod ≈ 2 * ( Total Checkpoint Size / Number of Hosts in the Slice**)

It's a good practice to add a **10-15% buffer** .

### Example Calculation

Let's walk through an example for a large model.

* **Model**: A 70 billion parameter language model.
* **Training Slice**: A nodepool with **32 hosts**.

1.  **Estimate Total Checkpoint Size**:
    `70,000,000,000 parameters × 12 bytes/parameter = 840,000,000,000 bytes`
    `840,000,000,000 bytes ≈ 840 GB`
2.  **Calculate Per-Host Checkpoint shard**:
    `(Total Checkpoint Size / 32 hosts) = 26.25 GB per host`

3.  **Calculate Per-Host Ramdisk Size**:
    `(Per-Host Checkpoint shard) * 2 = 52.50 GB per host`

4.  **Add a Safety Buffer (e.g., 15%)**:
    `(Per-Host Ramdisk Size) × 1.15 ≈ 60.3 GB`

In this scenario, you should configure each pod in that slice with a ramdisk of at least **60 GB**.

### Example XPK cluster creation command

1.  **Set up environment variables:**
    ```bash
    OUTPUT_PATH=<gcs bucket output path>
    PROJECT_ID=<project id>
    ZONE=<your zone>
    CLUSTER_NAME=<cluster name>
    TPU_TYPE=<tpu-type> #example: v6e-256
    MACHINE_TYPE=<cpu machine-type>
    NUM_SLICES=<number of slices>
    RAMDISK_SIZE=<ramdisk size> #example: 60000Mi
    GKE_VERSION=<gke version> #example: 1.32.3-gke.1785000
    ```
2.  **Configure gcloud:**
    ```bash
    gcloud config set project ${PROJECT_ID}
    gcloud config set compute/zone ${ZONE}
    ```
3.  **Clone the XPK repository:**
    ```bash
    git clone [https://github.com/AI-Hypercomputer/xpk.git](https://github.com/AI-Hypercomputer/xpk.git)
    ```
4.  **Run the cluster creation command:**
    ```bash
    python3 xpk/xpk.py cluster create \
    --cluster ${CLUSTER_NAME} \
    --cluster-cpu-machine-type=${MACHINE_TYPE} \
    --num-slices=${NUM_SLICES} \
    --tpu-type=${TPU_TYPE} \
    --enable-mtc \
    --enable-gcsfuse-csi-driver \
    --mtc-ramdisk-size=${RAMDISK_SIZE} \
    --mtc-gcs-bucket=${OUTPUT_PATH} \
    --gke-version=${GKE_VERSION}
    ```

## MaxText Configuration

MaxText provides a set of configuration flags to control checkpointing options. This configuration manages a `two-tiered checkpointing` system designed for both durability and rapid recovery.

* **Local Emergency Checkpoints**: It saves checkpoints much more frequently to a fast, local directory on each host (i.e. a ramdisk). If a preemption or failure occurs, the job can restore from this recent local copy, minimizing lost work without needing to download from slower persistent storage. This feature is enabled by setting `enable_checkpointing`, `enable_emergency_checkpoint`, `local_checkpoint_directory` and a non-zero `local_checkpoint_period`.

* **Persistent Checkpoints**: These are standard checkpoints saved periodically and much more rarely to durable storage(i.e. GCS bucket). They ensure that you can recover your training state even after a complete cluster failure. This is controlled by `enable_checkpointing`, and `checkpoint_period`.


| Flag | Description | Type | Default |
| :--- | :--- | :--- | :--- |
| `enable_checkpointing` | A master switch to enable (`True`) or disable (`False`) saving checkpoints during the training run. | `boolean` | `False` |
| `enable_emergency_checkpoint` | When set to (`True`), this flag enables the two-tiered emergency checkpointing feature. | `boolean` | `False` |
| `async_checkpointing` | When set to (`True`), this flag makes checkpoint saving asynchronous. The training step is only blocked for the minimal time needed to capture the model's state, and the actual writing to storage happens in a background thread. This is highly recommended for performance. It's enabled by default. | `boolean` | `True` |
| `local_checkpoint_directory` | The high-speed local filesystem path(i.e. ramdisk) where **emergency checkpoints** are saved. Setting this path, along with a non-zero `local_checkpoint_period`, enables the emergency checkpointing feature. | `string` | `""` |
| `local_checkpoint_period` | The interval, in training steps, for how often a **local checkpoint** is saved. This should be set to a much smaller value than `checkpoint_period` for frequent, low-overhead saves. | `integer` | `0` |
| `checkpoint_period` | The interval, in training steps, for how often a checkpoint is saved to **persistent storage**. | `integer` | `10000` |
| `enable_single_replica_ckpt_restoring` | If `True`, one replica reads the checkpoint from storage and then broadcasts it to all other replicas. This can significantly speed up restoration on multi-host systems by reducing redundant reads from storage. | `boolean` | `False` |

## Workload Creation using XPK

The flags below would give the user access to the ramdisk in their workload:

| Flag | Description |
| :--- | :--- |
| `--mtc-enabled` | Enables the Multi-Tier Checkpointing feature, by mounting ramdisk to the workload pods, using csi drivers. |
| `--ramdisk-directory` | Specifies the mount path inside each pod where the high-speed ramdisk will be accessible. Your training application should write its local, emergency checkpoints to this path. |


### Example XPK workload creation command

1.  **Set up environment variables:**
    ```bash
    RAMDISK_DIRECTORY=<your ramdisk directory>
    WORKLOAD_NAME=<YOUR WORKLOAD NAME>
    TPU_TYPE=<tpu-type>
    NUM_SLICES=<number of slices>
    PROJECT_ID=<project-id>
    LOCAL_CHECKPOINT_PERIOD=<>
    CHECKPOINT_PEROID=<checkpoint_period>
    STEPS=<steps>
    DATA_PATH=<dataset path>
    OUTPUT_PATH=<gcs bucket output path>
    ```

2.  **Define the Docker image:**
    ```bash
    DOCKER_IMAGE=gcr.io/${PROJECT_ID}/${USER}_mtc_runner:latest
    ```

3.  **Run the workload creation command:**
    ```bash
    python3 xpk/xpk.py workload create \
    --cluster ${CLUSTER_NAME} \
    --docker-image ${DOCKER_IMAGE} \
    --workload ${WORKLOAD_NAME} \
    --tpu-type=${TPU_TYPE} \
    --num-slices=${NUM_SLICES} \
    --ramdisk-directory=${RAMDISK_DIRECTORY} \
    --mtc-enabled \
    --command "python3 MaxText/train.py MaxText/configs/base.yml base_output_directory=$OUTPUT_PATH dataset_path=$DATA_PATH steps=120 per_device_batch_size=6 enable_checkpoint_cloud_logger=True checkpoint_period=${CHECKPOINT_PEROID} enable_emergency_checkpoint=True local_checkpoint_period=${LOCAL_CHECKPOINT_PERIOD} local_checkpoint_directory=/${RAMDISK_DIRECTORY}"
    ```
