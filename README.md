Some backlogged needs:

* Does it work/converge? Who knows?
* Lightweight testing:
-  Unit testing strategy and first test
-  Linting and linting controls
* First XLML test
* Managing configs for experiments
* Multihost/Multipod
-   Needs multihost data loading (https://github.com/sholtodouglas/multihost_dataloading)
-   Data parallelism across pods / weight sharding within Pods
* Write metrics, probably to tensorboard gcs (MFU, Perplexity, Iteration Time)
-   Needs to figure out an approved way to estimate TFLOPs.
* Bigger dataset (might be able to use an internal datset for now)
* Checkpointing to GCS

Longer term:
* OSS work to one day open source
* README for users
* Open source friendly bigger dataset
* Hosted pretrained parameters
