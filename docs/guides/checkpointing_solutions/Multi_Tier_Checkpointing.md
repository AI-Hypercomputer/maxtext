# Multi-Tier Checkpointing

Multi-tier checkpointing is a solution designed to optimize the storage and management of checkpoints for large-scale machine learning (ML) training jobs, particularly those utilizing thousands of nodes. It aims to increase **"Goodput"** (the time spent making progress) and decrease costs by reducing wasted progress and the **mean-time-to-recovery (MTTR)** from failures.

---

## Purpose and Benefits

* **Addresses frequent interruptions**: Large-scale ML training jobs are prone to frequent interruptions (potentially hourly), and recovery from these can be slow.
* **Improves Goodput**: By saving checkpoints more frequently and efficiently, multi-tier checkpointing reduces the amount of lost progress when a failure occurs, thereby increasing the overall Goodput of the training process.
* **Reduces MTTR**: The multi-tiered approach allows for faster restoration of training progress after a disruption.
* **Asynchronous restore**: During the ML training workload's startup, available checkpoint shards are asynchronously copied to the local ramdisk. These shards are pulled from the fastest available source, whether from local peer nodes or the backup on GCS persistent storage. This process ensures the data is ready to be picked up by Orbax from the ramdisk, minimizing startup delays.
---

## Architecture and Tiers

Multi-tier checkpointing stores checkpoints across multiple tiers of storage:

* **RAM (in-memory)**: Checkpoints are stored in each node's RAM for the fastest access and lowest latency. This is used for frequent, local saves.
* **In-cluster (peer replication)**: Checkpoints can be replicated to other nodes or slices within the cluster.
* **GCS (persistent storage)**: Checkpoints are backed up to GCS for long-term durability and global accessibility. This tier is used for less frequent, but more robust, saves.

---

## Implementation Details

* **GKE Component**: A managed GKE component is involved in handling high-scale checkpointing, including controllers, daemonsets, worker discovery, and rank assignment.
* **Local Storage**: For multi-tier checkpointing, local storage (such as ramdisk provided by a CSI ephemeral driver) is used for checkpoints, persisting across workload pod deletions.
* **Replication Service**: A replication service in a managed GKE component backs up local checkpoint files to GCS at certain intervals and is responsible for fetching latest checkpoint files to nodes without local checkpoints during restoration.
---

## Comparison with Other Checkpointing Methods

* **GCS Checkpointing**: This involves saving the model state directly to durable storage like GCS. However, this can be slow at larger model/cluster scales, blocking training, and leading to redundant data copies.
* **Emergency/Ramdisk Checkpointing**: This method utilizes low-latency ramdisk for checkpoint storage, but lacks robustness of GCS storage.
* **Multi-tier Checkpointing (Ramdisk + GCS)**: This combines the speed of ramdisk with the durability of GCS, offering a more robust and efficient solution.
.

---

## Assumptions

* **GKE Environment**: A **Google Kubernetes Engine (GKE)** cluster must be used. GCE infrastructure solutions like QueuedResources are not supported.
* **Multi-Tier Checkpointing Enabled on GKE cluster level**: The Multi-Tier Checkpointing feature must be enabled and configured on your GKE cluster. This involves setting up the necessary CSI drivers and configurations as outlined in the [official documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/training/multi-tier-checkpointing). 
* **Multi-Slice Workload**: The training job must be a **multi-slice environment**, meaning it utilizes more than one node pool.
* **Orbax Checkpointer**: The **Orbax** library must be used for checkpointing in your training script.
* **Ramdisk Mounted via Jobset**: Each workload pod must have a **ramdisk directory mounted by Jobset using the Multi-Tier Checkpointing CSI driver**. This
  provides a high-speed, in-memory storage location for checkpoints.

---

## Cluster Creation using XPK

To run workloads with Multi-Tier Checkpointing (MTC), you need a Google Kubernetes Engine (GKE) cluster with the necessary drivers and features enabled. You can create a properly configured cluster using the **XPK** or by setting it up manually with `gcloud` commands following [official documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/training/multi-tier-checkpointing).


The `xpk` script provides a streamlined way to create a GKE cluster with all the required MTC settings. The key flags used are:

* `--enable-mtc`: Enables the Multi-Tier Checkpointing feature.
* `--enable-gcsfuse-csi-driver`: Installs the required GCS FUSE CSI driver.
* `--mtc-ramdisk-size`: Allocates an in-memory ramdisk on each node for fast, local checkpoints.
* `--mtc-gcs-bucket`: Specifies the GCS bucket.


#### Calculating Ramdisk Size Per Host 

The total size of a full training checkpoint (including model weights and optimizer state) can be estimated based on the number of model parameters.
A good rule of thumb for a model trained with:
**Total Checkpoint Size ≈ Number of Parameters × 12 bytes**

For example, a 1 billion parameter model would require approximately **1B × 12 bytes = 12 GB** for a full checkpoint.

In a distributed training environment, the checkpoint is **sharded**, or split, across all the hosts in a slice. Each host is only responsible for saving its portion of the total checkpoint. Therefore, the ramdisk on a single pod only needs to be large enough for its local shard.

The formula is:
**Required Ramdisk Size per Pod ≈ Total Checkpoint Size / Number of Hosts in the Slice**

It's a good practice to add a **10-15% buffer** .

#### Example Calculation

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

#### Example XPK cluster creation command

```
OUTPUT_PATH=<gcs bucket output path>
PROJECT_ID=<project id>
ZONE=<your zone>
CLUSTER_NAME=<cluster name>
TPU_TYPE=<tpu-type>
MACHINE_TYPE=<cpu machine-type>
NUM_SLICES=<number of slices>
RAMDISK_SIZE="60000Mi"
GKE_VERSION=1.32.3-gke.1785000

gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}

git clone https://github.com/AI-Hypercomputer/xpk.git

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

---
### Maxtext Configuration

This configuration manages a `multi-tiered checkpointing` system designed for both durability and rapid recovery.

* **local checkpointing**: It saves checkpoints much more frequently to a fast, local directory on each host (i.e. a ramdisk). If a preemption or failure occurs, the job can restore from this recent local copy almost instantly, minimizing lost work without needing to download from slower persistent storage. This feature is enabled by setting `enable_checkpointing`, `enable_emergency_checkpoint`, `local_checkpoint_directory`, `use_replicator_service`, and a non-zero `local_checkpoint_period`.

* **backup checkpointing**: These are checkpoints saved periodically to persistent storage(i.e. GCS bucket). They ensure that you can recover your training state even after a complete job failure(repair of all nodepools). From User's perspective all restoration is from local ramdisk, its replicator service responsibility to make the checkpointing available to local storage in case of job restart. The interval for backup can be enabled by setting a non-zero `replicator_backup_interval_minutes`.

#### `enable_checkpointing`
A master switch to enable (`True`) or disable (`False`) saving checkpoints during the training run.
* **Type**: `boolean`
* **Default**: `False`

#### `enable_emergency_checkpoint`
When set to (`True`), this flag enables the  emergency checkpointing feature. 
* **Type**: `boolean`
* **Default**: `False`

#### `use_replicator_service`
When set to (`True`), along side with  `enable_emergency_checkpoint`, this flag Multi-Tier checkpointing feature. 
* **Type**: `boolean`
* **Default**: `False`

#### `local_checkpoint_directory`
The high-speed local filesystem path(i.e. ramdisk) where **MTC checkpoints** are saved. Setting this path, along with a non-zero `local_checkpoint_period`, enables the Multi-tier Checkpointing feature.
* **Type**: `string`
* **Default**: `""`

#### `local_checkpoint_period`
The interval, in training steps, for how often a **Multi-tier checkpoint** is saved in local ramdiks.
* **Type**: `integer`
* **Default**: `0`

#### `replicator_backup_interval_minutes`
The interval, in minutes, for how often a **Multi-tier checkpoint** is saved to backup from local ramdiks.
* **Type**: `integer`
* **Default**: `0`
---
### Workload Creation using XPK

Flags below would give the user access to the Ramdisk in their workload:

* `--mtc-enabled`: Enables the Multi-Tier Checkpointing feature, by mounting ramdisk to the workload pods, using csi drivers.
* `--ramdisk-directory`: Specifies the mount path inside each pod where the high-speed ramdisk will be accessible. Your training application should write its local checkpointing to this path.

#### Example XPK workload creation command

```
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

DOCKER_IMAGE=gcr.io/${PROJECT_ID}/${USER}_mtc_runner:latest

COMMAND="python3 MaxText/train.py MaxText/configs/base.yml base_output_directory=$OUTPUT_PATH dataset_path=$DATA_PATH steps=120 per_device_batch_size=6 enable_checkpoint_cloud_logger=True checkpoint_period=${CHECKPOINT_PEROID} enable_emergency_checkpoint=True local_checkpoint_period=${LOCAL_CHECKPOINT_PERIOD} local_checkpoint_directory=/${RAMDISK_DIRECTORY} use_replicator_service=True replicator_backup_interval_minutes=${REPLICATOR_BACKUP_INT_MIN}"

python3 xpk/xpk.py workload create \
--cluster ${CLUSTER_NAME} \
--docker-image ${DOCKER_IMAGE} \
--workload ${WORKLOAD_NAME} \
--tpu-type=${TPU_TYPE} \
--num-slices=${NUM_SLICES} \
--ramdisk-directory=${RAMDISK_DIRECTORY} \
--mtc-enabled  \
--command $COMMAND

```

