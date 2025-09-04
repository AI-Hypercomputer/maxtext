# Multi-Tier Checkpointing

Multi-tier checkpointing is a solution designed to optimize the storage and management of checkpoints for large-scale machine learning (ML) training jobs, particularly those utilizing thousands of nodes. It aims to increase **"Goodput"** (the time spent making progress) and decrease costs by reducing wasted progress and the **mean-time-to-recovery (MTTR)** from failures.

## Purpose and Benefits

* **Addresses frequent interruptions**: Large-scale ML training jobs are prone to frequent interruptions (potentially hourly), and recovery from these can be slow.
* **Improves Goodput**: By saving checkpoints more frequently and efficiently, multi-tier checkpointing reduces the amount of lost progress when a failure occurs, thereby increasing the overall Goodput of the training process.
* **Reduces MTTR**: The multi-tiered approach allows for faster restoration of training progress after a disruption.
* **Optimized restore**: During the ML training workload's startup, available checkpoint shards are asynchronously copied to the local ramdisk. These shards are pulled from the fastest available source, whether from local peer nodes or the backup on GCS persistent storage. This process ensures the data is ready to be picked up by Orbax from the ramdisk, minimizing startup delays.

## Architecture and Tiers

Multi-tier checkpointing stores checkpoints across multiple tiers of storage:

* **RAM (in-memory)**: Checkpoints are stored in each node's RAM for the fastest access and lowest latency. This is used for frequent, local saves.
* **In-cluster (peer replication)**: Checkpoints are replicated to other nodes or slices within the cluster.
* **GCS (persistent storage)**: Checkpoints are backed up to GCS for long-term durability and global accessibility. This tier is used for less frequent, but more robust, saves.

## Implementation Details

* **GKE Component**: A managed GKE component is involved in handling high-scale checkpointing, including controllers, daemonsets, worker discovery, and rank assignment.
* **Local Storage**: For multi-tier checkpointing, local storage (such as ramdisk provided by a CSI ephemeral driver) is used for checkpoints, persisting across workload pod deletions.
* **Replication Service**: A replication service in a managed GKE component replicates checkpoints in-cluster and backs up local checkpoint files to GCS at certain intervals and is responsible for fetching latest checkpoint files to nodes without local checkpoints during restoration.

## Comparison with Other Checkpointing Methods

* **GCS Checkpointing**: This involves saving the model state directly to durable storage like GCS. However, this can be slow at larger model/cluster scales, blocking training, and leading to redundant data copies.
* **Emergency/Ramdisk Checkpointing**: While this method uses a low-latency ramdisk for checkpointing, Orbax manages the GCS save and restore operations at the workload level. As a result, saving to GCS blocks the training process during the device-to-host data transfer.
* **Multi-tier Checkpointing (Ramdisk + GCS)**: This approach combines the speed of ramdisk, the resilience of in-cluster replication, and the durability of GCS to offer a more robust and efficient solution. With Multi-Tier Checkpointing, above blocking issue is resolved because the GCS save is handled at the service level, operating on the checkpoint already saved locally.

## Assumptions

* **GKE Environment**: A **Google Kubernetes Engine (GKE)** cluster must be used. GCE infrastructure solutions like QueuedResources are not supported.
* **Multi-Tier Checkpointing Enabled on GKE cluster level**: The Multi-Tier Checkpointing feature must be enabled and configured on your GKE cluster. This involves setting up the necessary CSI drivers and configurations as outlined in the [Google Cloud Checkpointing Documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/training/multi-tier-checkpointing).
* **Multi-Slice Workload**: The training job must be a [multi-slice environment](https://cloud.google.com/kubernetes-engine/docs/how-to/tpu-multislice), meaning it utilizes more than one node pool.
* **Orbax Checkpointer**: The [Orbax library](https://orbax.readthedocs.io) must be used for checkpointing in your training script.
* **Ramdisk Mounted via Jobset**: Each workload pod must have a [ramdisk directory mounted by Jobset](https://cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/training/multi-tier-checkpointing#update-jobset) using the Multi-Tier Checkpointing CSI driver. This provides a high-speed, in-memory storage location for checkpoints.
* **Supported TPU types**: [v4](https://cloud.google.com/tpu/docs/v4), [v5e](https://cloud.google.com/tpu/docs/v5e), [v5p](https://cloud.google.com/tpu/docs/v5p), and [v6e](https://cloud.google.com/tpu/docs/v6e)
* **Cluster version**: Gke cluster version needs to be later than [1.32.3-gke.1170000](https://cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/training/multi-tier-checkpointing#existing-cluster).

## Cluster Creation using XPK

To run workloads with Multi-Tier Checkpointing (MTC), you need a Google Kubernetes Engine (GKE) cluster with the necessary drivers and features enabled. You can create a properly configured cluster using the **XPK** or by setting it up manually with `gcloud` commands following [Google Cloud Checkpointing Documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/training/multi-tier-checkpointing).

The [xpk script](https://github.com/AI-Hypercomputer/xpk/blob/develop/xpk.py) provides a streamlined way to create a GKE cluster with all the required MTC settings. The key flags used are:

| Flag | Description |
| :--- | :--- |
| `--enable-mtc` | Enables the Multi-Tier Checkpointing feature. |
| `--enable-gcsfuse-csi-driver` | Installs the required GCS FUSE CSI driver. |
| `--mtc-ramdisk-size` | Allocates an in-memory ramdisk on each node for fast, local checkpoints. |
| `--mtc-gcs-bucket` | Specifies the GCS bucket. |


### Calculating Ramdisk Size Per Host 

The total size of a full training checkpoint (including model weights and optimizer state) can be estimated based on the number of model parameters.
A good rule of thumb:
**Total Checkpoint Size ≈ Number of Parameters × 12 bytes**

For example, a 1 billion parameter model would require approximately **1B × 12 bytes = 12 GB** for a full checkpoint.

In a distributed training environment, the checkpoint is **sharded**, or split, across all the hosts in a slice. Each host is only responsible for saving its portion of the total checkpoint. Therefore, the ramdisk on a single pod only needs to be large enough for its local shard.

The formula is:
**Required Ramdisk Size per Pod ≈ 2 * (Total Checkpoint Size / Number of Hosts in the Slice)**

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

This configuration manages a `multi-tiered checkpointing` system designed for both durability and rapid recovery.

* **Local checkpointing**: Saves checkpoints much more frequently to a fast, local directory on each host (i.e. a ramdisk). If a preemption or failure occurs, the job can restore from this recent local copy almost instantly, minimizing lost work without needing to download from slower persistent storage. This feature is enabled by setting `enable_checkpointing`, `enable_multi_tier_checkpointing`, `local_checkpoint_directory`, and a non-zero `local_checkpoint_period` flags.

* **Backup checkpointing**: These are checkpoints saved periodically to persistent storage(i.e. GCS bucket). They ensure that you can recover your training state even after a complete job failure(repair of all nodepools). From User's perspective all restoration is from local ramdisk, its replicator service responsibility to make the checkpointing available to local storage in case of job restart. The interval for backup can be enabled by setting a non-zero `multi_tier_checkpointing_backup_interval_minutes` flags.

| Flag | Description | Type | Default |
| :--- | :--- | :--- | :--- |
| `enable_checkpointing` | A master switch to enable (`True`) or disable (`False`) saving checkpoints during the training run. | `boolean` | `False` |
| `enable_multi_tier_checkpointing` | When set to (`True`), this flag enables the multi-tier checkpointing feature on maxtext level. | `boolean` | `False` |
| `local_checkpoint_directory` | The high-speed local filesystem path(i.e. ramdisk) where **Multi-tier checkpoints** are saved. Setting this path, along with a non-zero `local_checkpoint_period`, enables the Multi-tier Checkpointing feature. | `string` | `""` |
| `local_checkpoint_period` | The interval, in training steps, for how often a **Multi-tier checkpoint** is saved in local ramdisks. | `integer` | `0` |
| `multi_tier_checkpointing_backup_interval_minutes`| The interval, in minutes, for how often a **Multi-tier checkpoint** is saved to backup from local ramdisks. | `integer` | `0` |

### Workload Creation using XPK

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
    MULTI_TIER_CHECKPOINTING_BACKUP_INT_MIN=<multi_tier_checkpointing_backup_interval_minutes>
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
    --mtc-enabled  \
    --command "python3 MaxText/train.py MaxText/configs/base.yml base_output_directory=$OUTPUT_PATH dataset_path=$DATA_PATH steps=120 per_device_batch_size=6 enable_checkpoint_cloud_logger=True checkpoint_period=${CHECKPOINT_PEROID} enable_multi_tier_checkpointing=True local_checkpoint_period=${LOCAL_CHECKPOINT_PERIOD} local_checkpoint_directory=/${RAMDISK_DIRECTORY} multi_tier_checkpointing_backup_interval_minutes=${MULTI_TIER_CHECKPOINTING_BACKUP_INT_MIN}"
    ```