### Getting starting with using the maxtest tool

*Note: This is currently in preview stage and only supported for v5p and v6e*

The main intent of this tool is to quickly verify whether the provisioned TPU
nodepool can run a Maxtext workload **without having to
use XPK**. This is useful for debugging provisioned nodepools for potential hardware issues.

```shell
bash maxtest.sh \
    --project tpu-test-project \
    --cluster tpu-cluster \
    --region asia-east1 \
    --nodepool tpu-nodepool \
    --num_workers 8 # the nodepool here is a v5p-128 so 8 workers

Fetching cluster endpoint and auth data.
kubeconfig entry generated for bodaborg-v6e-32-td-c.
Applying generated configuration to the cluster:
service/headless-svc-87e0b-maxtest created job.batch/87e0b-maxtest created

Job started, visit https://console.cloud.google.com/logs/query;query=resource.labels.pod_name%3D~%2287e0b-maxtest%22;duration=PT1H?project=tpu-test-project to view logs
or run: 'kubectl logs job.batch/87e0b-maxtest'
```

-   After running the job, you can check the logs and the exit code to determine
    the outcome. A successful job will end with `EXIT_CODE=0` as well as the
    TFLOPs and step time of each step.

```
I0731 09:47:48.898259     514 isa_program_util_common.cc:493] (HLO module jit__where): Host transfer fingerprint:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
I0731 09:47:48.898413     139 2a886c8_compiler_base.cc:7540] END_TO_END stage duration: 10.6394605ms
completed step: 0, seconds: 1.353, TFLOP/s/device: 1674.038, Tokens/s/device: 36325.072, total_weights: 1572864, loss: 10.874
completed step: 1, seconds: 12.444, TFLOP/s/device: 182.028, Tokens/s/device: 3949.842, total_weights: 1572864, loss: 10.410
...
completed step: 2, seconds: 0.002, TFLOP/s/device: 917442.589, Tokens/s/device: 19907654.921, total_weights: 1572864, loss: 10.043
...
completed step: 3, seconds: 11.715, TFLOP/s/device: 193.354, Tokens/s/device: 4195.595, total_weights: 1572864, loss: 9.774
completed step: 4, seconds: 11.645, TFLOP/s/device: 194.526, Tokens/s/device: 4221.032, total_weights: 1572864, loss: 9.610
Unified tuning params for model are: {'per_device_batch_size': 12, 'ici_fsdp_parallelism': -1, 'remat_policy': 'full', 'max_target_length': 4096, 'attention': 'flash', 'gcs_metrics': False, 'use_iota_embed': True, 'dataset_path': 'gs://max-datasets-rogue', 'dataset_type': 'synthetic', 'reuse_example_batch': 1, 'enable_checkpointing': False, 'profiler': 'xplane', 'sa_block_q': 1024, 'sa_block_q_dkv': 2048, 'sa_block_q_dq': 2048}
Job End: Thu Jul 31 09:48:56 UTC 2025
EXIT_CODE=0
```

-   maxtest.sh will generate a YAML file in the directory that is passed to kubectl. This file can be modified and reused by running `kubectl apply -f maxtest.yaml`

### Passing custom libtpu or XLA flags ###

If we want to pass custom flags this is also possible by specifying
`--libtpu_args`.


#### Setting flags for SDC checking ####

Useful checking for the existence of SDC on TPU hardware.

```
bash maxtest.sh --project $TPU_PROJECT --cluster $CLUSTER --region $REGION --nodepool $NODEPOOL_NAME --num_workers $NUM_WORKERS --libtpu_args '--xla_tpu_enable_sdc_checker'
```


### Debugging common job errors ###

If the job does not exit with `EXIT_CODE=0`, there is a failure among one of
the workers or the coordinator VM, this can be debugged by checking the specific
worker logs.

Modify the `pod_name` in Cloud Logs Explorer to the specific worker to view its logs
by hyphenating the `JOB_NAME` with the index of the worker:

```
resource.labels.pod_name=~"87e0b-maxtest-0" #Here 0 indicates the coordinator VM (VM 0) logs and '87e0b-maxtext' is the job name
```

#### Slice Failure / Chip Driver Error ####

A `SLICE_FAILURE_CHIP_DRIVER_ERROR` indicates a problem with one of the TPU workers. The logs from `worker_0` (the coordinator) will show which worker missed the health check.

```
I0511 18:18:30.779521     915 session_master.cc:211] Worker Address: tpu-job-jax-v6e-8-5-53.headless-svc:8471                                                                                                                                        
I0511 18:18:30.779522     915 session_master.cc:211] Worker Address: tpu-job-jax-v6e-8-5-38.headless-svc:8471                                                                                                                                        
W0511 18:18:50.796322     914 session_master.cc:539] The task with address tpu-job-jax-v6e-8-5-38.headless-svc:8471 missed 1 health checks.The most recent health check failed with the following error: UNAVAILABLE: failed to connect to all address
ses; last error: UNKNOWN: ipv4:10.200.16.4:8471: Failed to connect to remote host: Timeout occurred: FD Shutdown                                                                                                                                     
=== Source Location Trace: ===                                                                                                                                                                                                                       
third_party/grpc/include/grpcpp/impl/status.h:112                                                                                                                                                                                                    
E0511 18:18:50.796514     915 session_master.cc:505] Session is failing with the following status: Session master detected TPU session error from worker tpu-job-jax-v6e-8-5-15.headless-svc:8471. The reporting TPU chip locations are:tpu31:pe4:3, 
tpu31:pe4:2                                                                                                                                                                                                                                          
E0511 18:18:50.796852     915 tpunetd_client.cc:311] Detected failure SLICE_FAILURE_CHIP_DRIVER_ERROR in session e6c885828b538167; Start failure handling                                                                                            
W0511 18:18:50.796885     915 session_master.cc
```

Use Logs Explorer to view the logs for that specific worker. For example, for worker 38, use the query `resource.labels.pod_name=~"tpu-job-jax-v6e-8-5-38"`. Where `38` is the VM that missed the health check and `tpu-job-jax-v6e-8-5` is the name of the job.

#### An XLA compiler bug or your job hit a bad TPU hardware ####

```
6x2x0_SC1: *** A program or fatal error occurred, all pending programs will fail, and results may be corrupted. This is likely an XLA compiler bug or your job hit a bad TPU hardware. If you suspect a compiler bug please file a bug to xla-tpu@ and CC tfrt-devs@. If the error is repeatedly happening on the same machine and you suspect faulty hardware, consider reporting a bad machine. Physical location: tpu3:pe2:2
```

Action: If you see this error, likely you have a bad TPU hardware issue, attempt to retry the workload a few times to confirm. The bad hardware will be on the same VM as the log is reported on. Delete and recreate the nodepool in this situation.



