import os
from jinja2 import Template
import argparse

def generate_rl_config(
    metadata_name, 
    batch_size, 
    rollout_data_parallelism, 
    rollout_tensor_parallelism, 
    rollout_expert_parallelism, 
    trainer_devices_fraction, 
    subslice_shape, 
    enable_single_controller, 
    sampler_devices_fraction, 
    base_output_directory, 
    run_name,
    hf_token
):
    yaml_template = """apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  labels:
    kueue.x-k8s.io/queue-name: multislice-queue
  name: {{ metadata_name }}
  namespace: default
spec:
  coordinator:
    replicatedJob: pathways-head
  failurePolicy:
    maxRestarts: 1
    restartStrategy: Recreate
  network:
    enableDNSHostnames: true
    publishNotReadyAddresses: true
  replicatedJobs:
  - name: pathways-head
    replicas: 1
    template:
      metadata:
        annotations:
          kueue.x-k8s.io/safe-to-forcefully-terminate: "true"
      spec:
        backoffLimit: 0
        completionMode: Indexed
        completions: 1
        parallelism: 1
        template:
          spec:
            containers:
            - command:
              - bash
              - -c
              - |
                echo XPK Start: $(date);
                _sigterm() (kill -SIGTERM $! 2>/dev/null;);
                trap _sigterm SIGTERM;

                (pip install --no-deps git+https://github.com/AI-Hypercomputer/pathways-utils.git@v0.1.4 && \\
                pip install src/maxtext/integration/vllm && \\
                HF_TOKEN={{ hf_token }} JAX_RANDOM_WEIGHTS=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 NEW_MODEL_DESIGN=1 TPU_MIN_LOG_LEVEL=0 TF_CPP_MIN_LOG_LEVEL=0 TPU_STDERR_LOG_LEVEL=0 JAX_PLATFORMS=proxy,cpu JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 ENABLE_PATHWAYS_PERSISTENCE=1 \\
                python3 -m src.maxtext.trainers.post_train.rl.reshard_debug src/maxtext/configs/post_train/rl.yml \\
                model_name=qwen3-30b-a3b \\
                tokenizer_path=Qwen/Qwen3-30B-A3B \\
                run_name={{ run_name }} \\
                base_output_directory={{ base_output_directory }} \\
                hf_access_token={{ hf_token }} \\
                batch_size={{ batch_size }} \\
                rl.num_generations={{ batch_size }} \\
                num_batches=4 \\
                rollout_data_parallelism={{ rollout_data_parallelism }} \\
                rollout_tensor_parallelism={{ rollout_tensor_parallelism }} \\
                rollout_expert_parallelism={{ rollout_expert_parallelism }} \\
                hbm_utilization_vllm=0.4 \\
                scan_layers=True \\
                allow_split_physical_axes=True \\
                vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \\
                vllm_additional_config='{maxtext_config: {model_name: qwen3-30b-a3b, allow_split_physical_axes: true, log_config: false, weight_dtype: bfloat16}}' \\
                trainer_devices_fraction={{ trainer_devices_fraction }} \\
                subslice_shape='{{ subslice_shape }}' \\
                enable_single_controller={{ enable_single_controller }} \\
                sampler_devices_fraction={{ sampler_devices_fraction }}) & PID=$!;

                while kill -0 $PID 2>/dev/null;
                    do sleep 5;
                done;
                wait $PID;
                EXIT_CODE=$?;

                echo XPK End: $(date);
                echo EXIT_CODE=$EXIT_CODE;

                exit $EXIT_CODE
              env:
              - name: PATHWAYS_HEAD
                valueFrom:
                  fieldRef:
                    fieldPath: metadata.labels['jobset.sigs.k8s.io/coordinator']
              - name: JAX_PLATFORMS
                value: proxy
              - name: XCLOUD_ENVIRONMENT
                value: GCP
              - name: JAX_BACKEND_TARGET
                value: grpc://$(PATHWAYS_HEAD):29000
              image: gcr.io/cloud-tpu-multipod-dev/sanbao/maxtext_reshard_image:latest
              imagePullPolicy: Always
              name: jax-tpu
              resources:
                limits:
                  cpu: "24"
                  memory: 100G
              securityContext:
                privileged: true
              volumeMounts:
              - mountPath: /tmp
                name: shared-tmp
            dnsPolicy: ClusterFirstWithHostNet
            hostNetwork: true
            initContainers:
            - args:
              - --server_port=29001
              - --gcs_scratch_location=gs://cloud-pathways-staging/tmp
              - --node_type=resource_manager
              - --instance_count=1
              - --instance_type=tpu7x:4x4x4
              env:
              - name: REPLICATED_JOB_NAME
                valueFrom:
                  fieldRef:
                    fieldPath: metadata.annotations['jobset.sigs.k8s.io/replicatedjob-name']
              - name: JOBSET_NAME
                valueFrom:
                  fieldRef:
                    fieldPath: metadata.annotations['jobset.sigs.k8s.io/jobset-name']
              - name: HOST_ADDRESS
                valueFrom:
                  fieldRef:
                    fieldPath: metadata.labels['jobset.sigs.k8s.io/coordinator']
              - name: TPU_SKIP_MDS_QUERY
                value: "true"
              image: us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest
              imagePullPolicy: Always
              name: pathways-rm
              ports:
              - containerPort: 29001
                protocol: TCP
              - containerPort: 29002
                protocol: TCP
              resources:
                limits:
                  cpu: "8"
                  memory: 32G
              restartPolicy: Always
            - args:
              - --server_port=29000
              - --resource_manager_address=$(PATHWAYS_HEAD):29001
              - --gcs_scratch_location=gs://cloud-pathways-staging/tmp
              env:
              - name: PATHWAYS_HEAD
                valueFrom:
                  fieldRef:
                    fieldPath: metadata.labels['jobset.sigs.k8s.io/coordinator']
              image: us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:latest
              imagePullPolicy: Always
              name: pathways-proxy
              ports:
              - containerPort: 29000
                protocol: TCP
              resources:
                limits:
                  cpu: "16"
                  memory: 100G
              restartPolicy: Always
            nodeSelector:
              cloud.google.com/gke-nodepool: cpu-np
            restartPolicy: Never
            volumes:
            - hostPath:
                path: /tmp
                type: DirectoryOrCreate
              name: shared-tmp
  - name: worker
    replicas: 1
    template:
      metadata:
        annotations:
          cloud.google.com/gke-tpu-slice-topology: 4x4x4
      spec:
        backoffLimit: 32
        completionMode: Indexed
        completions: 16
        parallelism: 16
        template:
          metadata:
            annotations:
              cloud.google.com/gke-tpu-slice-topology: 4x4x4
          spec:
            tolerations:
              - key: "google.com/tpu"
                operator: "Equal"
                value: "present"
                effect: "NoSchedule"
            affinity:
              nodeAffinity:
                requiredDuringSchedulingIgnoredDuringExecution:
                  nodeSelectorTerms:
                  - matchExpressions:
                    - key: cloud.google.com/gke-tpu-partition-4x4x4-state
                      operator: In
                      values:
                      - HEALTHY
                      - DEGRADED
            containers:
            - args:
              - --server_port=29005
              - --resource_manager_address=$(PATHWAYS_HEAD):29001
              - --gcs_scratch_location=gs://cloud-pathways-staging/tmp
              env:
              - name: TPU_MIN_LOG_LEVEL
                value: "0"
              - name: TF_CPP_MIN_LOG_LEVEL
                value: "0"
              - name: XCLOUD_ENVIRONMENT
                value: GCP
              - name: MEGASCALE_GRPC_ENABLE_XOR_TRACER
                value: "false"
              - name: MEGASCALE_NUM_SLICES
                valueFrom:
                  fieldRef:
                    fieldPath: metadata.labels['jobset.sigs.k8s.io/replicatedjob-replicas']
              - name: JOBSET_NAME
                valueFrom:
                  fieldRef:
                    fieldPath: metadata.annotations['jobset.sigs.k8s.io/jobset-name']
              - name: REPLICATED_JOB_NAME
                valueFrom:
                  fieldRef:
                    fieldPath: metadata.annotations['jobset.sigs.k8s.io/replicatedjob-name']
              - name: MEGASCALE_SLICE_ID
                valueFrom:
                  fieldRef:
                    fieldPath: metadata.labels['jobset.sigs.k8s.io/job-index']
              - name: PATHWAYS_HEAD
                valueFrom:
                  fieldRef:
                    fieldPath: metadata.labels['jobset.sigs.k8s.io/coordinator']
              - name: MEGASCALE_COORDINATOR_ADDRESS
                valueFrom:
                  fieldRef:
                    fieldPath: metadata.labels['jobset.sigs.k8s.io/coordinator']
              image: us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest
              imagePullPolicy: Always
              name: pathways-worker
              ports:
              - containerPort: 29005
                protocol: TCP
              - containerPort: 29006
                protocol: TCP
              - containerPort: 8471
                protocol: TCP
              - containerPort: 8080
                protocol: TCP
              resources:
                limits:
                  google.com/tpu: "4"
              volumeMounts:
              - mountPath: /tmp
                name: shared-tmp
            dnsPolicy: ClusterFirstWithHostNet
            hostNetwork: true
            nodeSelector:
              cloud.google.com/gke-tpu-accelerator: tpu7x
            priorityClassName: medium
            restartPolicy: OnFailure
            terminationGracePeriodSeconds: 30
            volumes:
            - hostPath:
                path: /tmp
                type: DirectoryOrCreate
              name: shared-tmp
  startupPolicy:
    startupPolicyOrder: InOrder
  successPolicy:
    operator: All
    targetReplicatedJobs:
    - pathways-head"""

    t = Template(yaml_template)
    rendered_yaml = t.render(
        metadata_name=metadata_name,
        batch_size=batch_size,
        rollout_data_parallelism=rollout_data_parallelism,
        rollout_tensor_parallelism=rollout_tensor_parallelism,
        rollout_expert_parallelism=rollout_expert_parallelism,
        trainer_devices_fraction=trainer_devices_fraction,
        subslice_shape=subslice_shape,
        enable_single_controller=enable_single_controller,
        sampler_devices_fraction=sampler_devices_fraction,
        base_output_directory=base_output_directory,
        run_name=run_name
    )
    
    return rendered_yaml

# Example Usage:
if __name__ == "__main__":
    # add args for metadat_name, trainer_chips, sampler_chips, rollout_data_parallelism, rollout_tensor_parallelism, rollout_expert_parallelism

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_name", type=str, required=True)
    parser.add_argument("--trainer_chips", type=int, required=True)
    parser.add_argument("--number_of_sampler_chips_per_replica", type=int, required=True)
    parser.add_argument("--sampler_sharding_per_replica", type=int, required=True)
    parser.add_argument("--sampler_replicas", type=int, required=True)
    parser.add_argument("--base_output_directory", type=str, required=True)
    parser.add_argument("--hf_token", type=str, required=True)
    args = parser.parse_args()

    # for v7x-128
    number_of_chips = 64
    batch_size = args.trainer_chips * 2
    trainer_devices_fraction = args.trainer_chips / number_of_chips
    rollout_data_parallelism = args.sampler_replicas
    sampler_chips = args.number_of_sampler_chips_per_replica * args.sampler_sharding_per_replica
    rollout_tensor_parallelism = sampler_chips // batch_size

    result = generate_rl_config(
        metadata_name=args.metadata_name,
        batch_size=batch_size,
        rollout_data_parallelism=args.rollout_data_parallelism,
        rollout_tensor_parallelism=args.rollout_tensor_parallelism,
        rollout_expert_parallelism=args.rollout_expert_parallelism,
        trainer_devices_fraction=0.0625,
        subslice_shape="2,2,1",
        enable_single_controller="true",
        sampler_devices_fraction=0.0625,
        base_output_directory=args.base_output_directory,
        run_name=args.metadata_name
        hf_token=args.hf_token
    )
    
    with open("qwen3-30b-v7x-temp.yaml", "w") as f:
        f.write(result)