#!/bin/bash

######### OVERVIEW OF SCRIPT #########
## This script is intended to guide one on the steps needed to create large scale
## (>5k VMs) with v5e-256 with xpk and GKE.
## This script was run by manually copying commands per step and verifying the
## output of each step.
## We recommend you manually copy commands per step and verify the outputs of
## each step.

## Step Summary is:
## Step 1: Cluster Networking setup.
## Step 2: Create your cluster with xpk.
## Step 3: Move from KubeDNS to CoreDNS. This is necessary past 1000 VMs.
## Step 4: Pass Cluster name and Project ID to Google POCs to setup your cluster
##         for large scale and high throughput. This is necessary past 5000 VMs.
## Step 5: Scale up your cluster.
## Step 6 (OPTIONAL):  Run a simple sample job on all slices in the cluster.
##                     Shows how to run jobs and view jobs with xpk.
## Step 7 (OPTIONAL):  Run maxtext training on slices in the cluster.
##                     Shows how to create custom docker images and use cacheimage to improve start up time.
##                     XPK offers cacheimage as a wrapper around daemonset.
######### OVERVIEW OF SCRIPT #########

### USER VARIABLES:
# TODO(USER): ADJUST PROJECT_NAME, CLUSTER NAME at least.
export PREFIX=${USER}
export PROJECT=PROJECT_NAME
export REGION=us-central2
export ZONE=us-central2-b
export CLUSTER=CLUSTER-NAME

gcloud config set project $PROJECT
gcloud config set compute/zone $ZONE
gcloud config set compute/region $REGION

### INTERNAL VARIABLES:
export NETWORK_NAME=${PREFIX}-privatenetwork
export SUBNET_NAME=${PREFIX}-privatesubnet
export FIREWALL_RULE_NAME=${PREFIX}-privatefirewall
export ROUTER_NAME=${PREFIX}-network
export NAT_CONFIG=${PREFIX}-natconfig
export NUMSLICES=4

##### STEP 1 #################
##### Cluster Networking #####
##############################

##### 1A #####################
# Create network for cluster.
##### 1A #####################

gcloud compute networks create "${NETWORK_NAME}" --mtu=8896 --bgp-routing-mode=regional --subnet-mode=custom

# Created [https://www.googleapis.com/compute/v1/projects/PROJECT/global/networks/PREFIX-privatenetwork].
# NAME                  SUBNET_MODE  BGP_ROUTING_MODE  IPV4_RANGE  GATEWAY_IPV4
# PREFIX-privatenetwork  CUSTOM       REGIONAL

# Instances on this network will not be reachable until firewall rules
# are created. As an example, you can allow all internal traffic between
# instances as well as SSH, RDP, and ICMP by running:

##### 1B #####################
# Create subnetwork for cluster.
##### 1B #####################

gcloud compute networks subnets create "${SUBNET_NAME}" --network="${NETWORK_NAME}" --range=10.10.0.0/18 --region="${REGION}"

# Created [https://www.googleapis.com/compute/v1/projects/PROJECT/regions/us-central2/subnetworks/PREFIX-privatesubnet].
# NAME                 REGION    NETWORK               RANGE         STACK_TYPE  IPV6_ACCESS_TYPE  INTERNAL_IPV6_PREFIX  EXTERNAL_IPV6_PREFIX
# PREFIX-privatesubnet  us-central2  PREFIX-privatenetwork  10.10.0.0/18  IPV4_ONLY

##### 1C #####################
# Create firewall rules for private network.
##### 1C #####################

gcloud compute firewall-rules create "${FIREWALL_RULE_NAME}" --network "${NETWORK_NAME}" --allow tcp,icmp,udp --project="${PROJECT}"

# Creating firewall...⠹Created [https://www.googleapis.com/compute/v1/projects/PROJECT/global/firewalls/PREFIX-privatefirewall].
# Creating firewall...done.
# NAME                   NETWORK               DIRECTION  PRIORITY  ALLOW         DENY  DISABLED
# PREFIX-privatefirewall  PREFIX-privatenetwork  INGRESS    1000      tcp,icmp,udp        False

##### 1D #####################
# Routers for network and region.
##### 1D #####################

gcloud compute routers create "${ROUTER_NAME}" \
  --project="${PROJECT}" \
  --network="${NETWORK_NAME}" \
  --region="${REGION}"

# Creating router [PREFIX-network]...done.
# NAME           REGION    NETWORK
# PREFIX-network  us-central2  PREFIX-privatenetwork

##### 1E #####################
# Router nats for the region
##### 1E #####################

gcloud compute routers nats create "${NAT_CONFIG}" \
  --router="${ROUTER_NAME}" \
  --region="${REGION}" \
  --auto-allocate-nat-external-ips \
  --nat-all-subnet-ip-ranges \
  --enable-logging

# Creating NAT [PREFIX-natconfig] in router [PREFIX-network]...done.

##### STEP 2 ############################
##### Create your cluster with xpk. #####
#########################################

##### 2A #####################
# Export cluster and node pool arguments
##### 2A #####################

export CLUSTER_ARGUMENTS=" \
 --network=${NETWORK_NAME} \
 --subnetwork=${SUBNET_NAME} \
 --scopes=storage-full,gke-default \
 --enable-ip-alias \
 --enable-private-nodes \
 --master-ipv4-cidr 172.16.0.32/28 \
 --cluster-ipv4-cidr=10.224.0.0/12 \
 --no-enable-master-authorized-networks \
"

export TPU_NODEPOOL_ARGUMENTS=" \
 --scopes=storage-full,gke-default \
 --enable-gvnic \
 --max-pods-per-node 15 \
 --disk-size=50 \
"

##### 2B #####################
# Git clone and go to the correct directory.
##### 2B #####################

git clone https://github.com/google/maxtext.git && cd maxtext


##### 2C #####################
# Confirm that variables are correctly set:
##### 2C #####################
echo python3 xpk/xpk.py cluster create \
  --cluster "${CLUSTER}" --tpu-type=v5litepod-256 \
  --num-slices="${NUMSLICES}" \
  --host-maintenance-interval=PERIODIC \
  --custom-cluster-arguments="${CLUSTER_ARGUMENTS}" \
  --custom-tpu-nodepool-arguments="${TPU_NODEPOOL_ARGUMENTS}"

# python3 xpk/xpk.py cluster create --cluster NAME \
#  --tpu-type=v5litepod-256 --num-slices=4 \
#  --host-maintenance-interval=PERIODIC \
# --custom-cluster-arguments=  --network=NETWORK  --subnetwork=SUBNET  --scopes=storage-full,gke-default  --enable-ip-alias  --enable-private-nodes  --master-ipv4-cidr 172.16.0.32/28  --cluster-ipv4-cidr=10.224.0.0/12  --no-enable-master-authorized-networks
# --custom-tpu-nodepool-arguments=  --scopes=storage-full,gke-default  --enable-gvnic  --max-pods-per-node 15  --disk-size=50


##### 2D #####################
# Run Cluster Create.
##### 2D #####################

# Rerun create command to update the cluster (with a new slice size) or if the create command fails.
python3 xpk/xpk.py cluster create \
  --cluster "${CLUSTER}" --tpu-type=v5litepod-256 \
  --num-slices="${NUMSLICES}" \
  --host-maintenance-interval=PERIODIC \
  --custom-cluster-arguments="${CLUSTER_ARGUMENTS}" \
  --custom-tpu-nodepool-arguments="${TPU_NODEPOOL_ARGUMENTS}"

# This process takes around 4 minutes with 4 slices of v5e-256.

###############################
##### 2D - TIPS ###############
###############################

# 1) View other examples of xpk.py here: https://github.com/google/maxtext/blob/main/xpk/README.md
# 2) xpk create command will update the cluster. If you adjust the num-slices and call create again,
#    xpk will intelligently adjust the # number of node pools and execute the number of create / delete commands.
# 3) If xpk create command fails, the first step is to try running create again.

##### STEP 3 ##############################
##### MOVE From KubeDNS to CoreDNS ########
###########################################

##### 3A #####################
# Install jq command-line JSON processor
##### 3A #####################

sudo apt install jq -y

##### 3B #####################
# git clone coredns deployment repo.
##### 3B #####################

git clone https://github.com/coredns/deployment.git

##### 3C #####################
# Go to repo and deploy coredns.
##### 3C #####################

cd deployment/kubernetes
./deploy.sh | kubectl apply -f -

# serviceaccount/coredns created
# clusterrole.rbac.authorization.k8s.io/system:coredns created
# clusterrolebinding.rbac.authorization.k8s.io/system:coredns created
# configmap/coredns created
# deployment.apps/coredns created
# Warning: resource services/kube-dns is missing the kubectl.kubernetes.io/last-applied-configuration annotation which is required by kubectl apply. kubectl apply should only be used on resources created declaratively by either kubectl create --save-config or kubectl apply. The missing annotation will be patched automatically.
# service/kube-dns configured

##### 3D #####################
# Scale down kube-dns-autoscaler
##### 3D #####################

kubectl scale deployment --replicas=0 kube-dns-autoscaler --namespace=kube-system

# deployment.apps/kube-dns-autoscaler scaled

##### 3E #####################
# Scale down kube-dns
##### 3E #####################

kubectl scale deployment --replicas=0 kube-dns --namespace=kube-system

# Warning: spec.template.metadata.annotations[scheduler.alpha.kubernetes.io/critical-pod]: non-functional in v1.16+; use the "priorityClassName" field instead
# Warning: spec.template.metadata.annotations[seccomp.security.alpha.kubernetes.io/pod]: non-functional in v1.27+; use the "seccompProfile" field instead
# deployment.apps/kube-dns scaled

##### 3F #####################
# Scale up core-dns
##### 3F #####################

# We recommend 15+ replicas
kubectl scale deployment coredns --replicas=15 -n kube-system

# deployment.apps/coredns scaled

##### 3G #####################
# Verify that kubedns pods have stopped.
# Verify that coredns pods have started.
##### 3G #####################

watch 'kubectl get pods -n kube-system -o=wide | grep dns | grep -i kube'
# These should be terminated / disappear soon.
watch 'kubectl get pods -n kube-system -o=wide | grep dns | grep -i core'
# These should create at least one coredns pod.

##### 3H #####################
# Rerun xpk cluster create to plumb coredns changes to the cluster.
##### 3H #####################

# Go to the correct directory.
cd ../..

# Cluster create is the same command as run previously in step 2D. It will
# not recreate the cluster but just update it.
python3 xpk/xpk.py cluster create \
  --cluster "${CLUSTER}" --tpu-type=v5litepod-256 \
  --num-slices="${NUMSLICES}" \
  --host-maintenance-interval=PERIODIC \
  --custom-cluster-arguments="${CLUSTER_ARGUMENTS}" \
  --custom-tpu-nodepool-arguments="${TPU_NODEPOOL_ARGUMENTS}"

##### STEP 4 ###################################################
##### PASS Cluster name and Project ID to Google POCs ##########
################################################################

# 4A. Tell Google POCs to setup GKE Cluster for large scale.
echo -e "\nTell Google POCS: We want $CLUSTER in $PROJECT to be set up for high throughput scheduling and large scale.\n"

##### STEP 5 ###############################
##### Begin Scale Up of GKE Cluster ########
############################################

##### 5A #####################
# TODO(USER): Set NUMSLICES to what you wish to scale to
##### 5A #####################

# Remember it is ok to incrementally scale if you wish. You can run cluster create
# repeatedly and adjust `--num-slices`.
export NUMSLICES=64

### USER VARIABLES:
# TODO(USER): ADJUST PROJECT_NAME AND CLUSTER NAME at least.
export PREFIX=${USER}
export PROJECT=PROJECT_NAME
export REGION=us-central2
export ZONE=us-central2-b
export CLUSTER=CLUSTER-NAME

gcloud config set project $PROJECT
gcloud config set compute/zone $ZONE
gcloud config set compute/region $REGION

### INTERNAL VARIABLES:
export NETWORK_NAME=${PREFIX}-privatenetwork
export SUBNET_NAME=${PREFIX}-privatesubnet
export FIREWALL_RULE_NAME=${PREFIX}-privatefirewall
export ROUTER_NAME=${PREFIX}-network
export NAT_CONFIG=${PREFIX}-natconfig

export CLUSTER_ARGUMENTS=" \
 --network=${NETWORK_NAME} \
 --subnetwork=${SUBNET_NAME} \
 --scopes=storage-full,gke-default \
 --enable-ip-alias \
 --enable-private-nodes \
 --master-ipv4-cidr 172.16.0.32/28 \
 --cluster-ipv4-cidr=10.224.0.0/12 \
 --no-enable-master-authorized-networks \
"

export TPU_NODEPOOL_ARGUMENTS=" \
 --scopes=storage-full,gke-default \
 --enable-gvnic \
 --max-pods-per-node 15 \
 --disk-size=50 \
"

##### 5B #####################
# Confirm that variables are correctly set:
##### 5B #####################

echo python3 xpk/xpk.py cluster create \
  --cluster "${CLUSTER}" --tpu-type=v5litepod-256 \
  --num-slices="${NUMSLICES}" \
  --host-maintenance-interval=PERIODIC \
  --custom-cluster-arguments="${CLUSTER_ARGUMENTS}" \
  --custom-tpu-nodepool-arguments="${TPU_NODEPOOL_ARGUMENTS}"

# python3 xpk/xpk.py cluster create --cluster NAME \
#  --tpu-type=v5litepod-256 --num-slices=64 \
#  --host-maintenance-interval=PERIODIC \
# --custom-cluster-arguments=  --network=NETWORK  --subnetwork=SUBNET  --scopes=storage-full,gke-default  --enable-ip-alias  --enable-private-nodes  --master-ipv4-cidr 172.16.0.32/28  --cluster-ipv4-cidr=10.224.0.0/12  --no-enable-master-authorized-networks
# --custom-tpu-nodepool-arguments=  --scopes=storage-full,gke-default  --enable-gvnic  --max-pods-per-node 15  --disk-size=50

##### 5C #####################
# Scale up to NUMSLICES (64 in the provided case) V5e-256s.
##### 5C #####################

python3 xpk/xpk.py cluster create \
  --cluster "${CLUSTER}" --tpu-type=v5litepod-256 \
  --num-slices="${NUMSLICES}" \
  --host-maintenance-interval=PERIODIC \
  --custom-cluster-arguments="${CLUSTER_ARGUMENTS}" \
  --custom-tpu-nodepool-arguments="${TPU_NODEPOOL_ARGUMENTS}"

###############################
##### 5C - POTENTIAL ERRORS ###
###############################
# If you see failures, the first step is to retry the cluster create command.
# [XPK] Terminating all Create and Delete Nodepools processes since at least one failed.
# [XPK] Failure is NodepoolCreate-PREFIX-CLUSTER-np-53 and logfile /tmp/NodepoolCreate-PREFIX-CLUSTER-np-53-pqfbm5nl
# [XPK] Create and Delete Nodepools returned ERROR 1
# [XPK] XPK failed, error code 1

# Auto-repairing nodepools on creation.
# Node pools can go into auto repair if there is an issue with their creation. For example,
# in the DEADLINE error, the node pool will automatically repair itself. This took ~20 minutes for the
# node pool to repair in the above example. You can continue with rerun cluster create
# commands while it is auto-repairing so that the rest of the cluster continues to
# be created while the repair occurs.

# It took 20 minutes for the above internal example to go from 4 to 64 NPs with
# a series of increment cluster create steps.

##### (OPTIONAL) STEP 6 (OPTIONAL) #########################################
##### Run a simple multislice sample job ###################################
############################################################################

##### 6A #####################
# Verify in Cloud Console that you have NUMSLICES node pools.
##### 6A #####################

# Verify in Cloud Console that you have NUMSLICES node pools in the NODES tab of Cloud Console.
echo "https://console.cloud.google.com/kubernetes/clusters/details/us-central2/${CLUSTER}/details?project=${PROJECT}"

##### 6B #####################
# Run a multislice workload on all slices.
##### 6B #####################

# Set  --scheduler=gke.io/high-throughput-scheduler to use the high throughput scheduler.

python3 xpk/xpk.py workload create \
 --scheduler=gke.io/high-throughput-scheduler \
 --workload xpk-test-workload --command "echo hello world" --cluster ${CLUSTER} \
 --tpu-type=v5litepod-256 --num-slices=${NUMSLICES}

# [XPK] Starting xpk
# [XPK] Working on args.project='PROJECT' and us-central2-b
# [XPK] Task: `Set Cluster` is implemented by `gcloud container clusters get-credentials CLUSTER --region=us-central2 --project=PROJECT && kubectl config view`, hiding output unless there is an error.
# [XPK] Task: `Set Cluster` succeeded.
# [XPK] Task: `Check if Workload Already Exists` is implemented by `kubectl get workloads -o=custom-columns='Jobset:.metadata.ownerReferences[0].name'`, hiding output unless there is an error.
# [XPK] Starting workload create
# [XPK] Task: `Creating Workload` is implemented by `kubectl apply -f /tmp/tmpk7599zd9`, streaming output live.
# [XPK] Waiting for `Creating Workload`, for 0 seconds
# [XPK] Waiting for `Creating Workload`, for 1 seconds
# jobset.jobset.x-k8s.io/xpk-test-workload created
# [XPK] Task: `Creating Workload` terminated with code `0`
# [XPK] Follow your workload here: WORKLOAD_LOGS_LINK
# [XPK] Exiting XPK cleanly

# ###########################################
# #### Logs expected from the above link ####
# ###########################################
# 2023-10-03 11:34:54.621 PDT
# XPK Start: Tue Oct 3 18:34:54 UTC 2023
# 2023-10-03 11:34:54.622 PDT
# hello world
# 2023-10-03 11:34:54.622 PDT
# XPK End: Tue Oct 3 18:34:54 UTC 2023
# 2023-10-03 11:34:54.622 PDT
# EXIT_CODE=0
# 2023-10-03 11:34:54.779 PDT
# XPK Start: Tue Oct 3 18:34:54 UTC 2023
# 2023-10-03 11:34:54.779 PDT
# hello world
# ...

##### 6C #####################
# Verify workload.
# Use the link in the above "WORKLOAD_LOGS_LINK" view logs
# Run xpk workload list to view all workloads on the cluster.
##### 6C #####################

# Use the link in the above "WORKLOAD_LOGS_LINK" view logs. You should see
# the echo command in cloud logs.
python3 xpk/xpk.py workload list \
 --cluster ${CLUSTER}

###############################
##### 6C - TIPS ###############
###############################
# If you see `Not all pods are ready or succeeded` then the workload is still running.
# If you see `JobSet finished successfully` then the workload is finished successfully.

# [XPK] Starting xpk
# Namespace(xpk_subcommands='workload', func=<function workload_list at 0x7fa23a059da0>, xpk_workload_subcommands='list', cluster='CLUSTER', project=None, zone=None, dry_run=False)
# [XPK] Starting workload list
# [XPK] Working on args.project='PROJECT' and us-central2-b
# [XPK] Task: `Set Cluster` is implemented by `gcloud container clusters get-credentials CLUSTER --region=us-central2 --project=PROJECT && kubectl config view`, hiding output unless there is an error.
# [XPK] Task: `Set Cluster` succeeded.
# [XPK] Task: `List Jobs` is implemented by `kubectl get workloads -o=custom-columns='Jobset:.metadata.ownerReferences[0].name,Created Time:.metadata.creationTimestamp,Priority:.spec.priorityClassName,TPU VMs Needed:.spec.podSets[0].count,Last Status Verbose:.status.conditions[-1].message,Last Status:.status.conditions[-1].status,Last Transition:.status.conditions[-1].lastTransitionTime,Current Queue:.status.admission.clusterQueue,All Done:.status.reclaimablePods[0].count'`, streaming output live.
# [XPK] Waiting for `List Jobs`, for 0 seconds
# [XPK] Waiting for `List Jobs`, for 1 seconds
# Jobset              Created Time           Priority   TPU VMs Needed   Last Status Verbose                   Last Status   Last Transition        Current Queue   All Done
# xpk-test-workload   2023-09-27T18:30:47Z   medium     5120             Not all pods are ready or succeeded   False         2023-09-27T18:30:47Z   cluster-queue   192
# [XPK] Task: `List Jobs` terminated with code `0`
# [XPK] Exiting XPK cleanly

##### (OPTIONAL) STEP 7 (OPTIONAL) ###############################
##### XPK with Maxtext ###########################################
##################################################################
# These instructions show you how to build a docker image related to Maxtext
# and cache the docker image for faster job start times.

##### 7A #####################
# Build local Maxtext docker image.
##### 7A #####################

git clone https://github.com/google/maxtext.git
cd maxtext
bash docker_build_dependency_image.sh

##### 7B #####################
# Upload image to gcp project.
##### 7B #####################

bash docker_upload_runner.sh CLOUD_IMAGE_NAME="${USER}"_runner

###############################
##### 7B - TIPS ###############
###############################
# Potential permissions you need for your account or service account:
# Storage Admin
# Kubernetes Engine Admin

##### 7C #####################
# Cluster cacheimage to enable faster start times.
# XPK offers cacheimage as a wrapper around daemonset.
##### 7C #####################
python3 xpk/xpk.py cluster cacheimage \
 --cluster ${CLUSTER} --docker-image gcr.io/"${PROJECT}"/"${USER}"_runner

# [XPK] Starting xpk
# [XPK] Starting cluster cacheimage for cluster: xpk-test
# [XPK] Working on args.project='PROJECT' and us-central2-b
# [XPK] Task: `Deleting Cached Image` is implemented by `kubectl delete -f /tmp/tmpypvl4dn_ --ignore-not-found=true`, streaming output live.
# [XPK] Waiting for `Deleting Cached Image`, for 0 seconds
# [XPK] Waiting for `Deleting Cached Image`, for 1 seconds
# daemonset.apps "containerimage" deleted
# [XPK] Task: `Deleting Cached Image` terminated with code `0`
# [XPK] Task: `Creating Cached Image` is implemented by `kubectl apply -f /tmp/tmpypvl4dn_`, streaming output live.
# [XPK] Waiting for `Creating Cached Image`, for 0 seconds
# daemonset.apps/containerimage created
# [XPK] Task: `Creating Cached Image` terminated with code `0`
# [XPK] Exiting XPK cleanly

##### 7D #####################
# Run workload in the cluster.
##### 7D #####################

export NUMSLICES=64

# Make sure you are in the maxtext github root directory when running this command
python3 xpk/xpk.py workload create \
 --cluster ${CLUSTER} \
 --docker-image gcr.io/${PROJECT}/"${USER}"_runner \
 --workload "${USER}"-first-job \
 --tpu-type=v5litepod-256 \
 --num-slices=${NUMSLICES}  \
 --scheduler=gke.io/high-throughput-scheduler \
 --command "python3 MaxText/train.py MaxText/configs/base.yml base_output_directory=gs://maxtext-experiments-tpem/ dataset_path=gs://max-datasets-rogue/ steps=100 per_device_batch_size=1"

# [XPK] Starting xpk
# [XPK] Working on args.project='PROJECT' and us-central2-b
# [XPK] Task: `Set Cluster` is implemented by `gcloud container clusters get-credentials CLUSTER --region=us-central2 --project=PROJECT && kubectl config view`, hiding output unless there is an error.
# [XPK] Task: `Set Cluster` succeeded.
# [XPK] Task: `Check if Workload Already Exists` is implemented by `kubectl get workloads -o=custom-columns='Jobset:.metadata.ownerReferences[0].name'`, hiding output unless there is an error.
# [XPK] Starting workload create
# [XPK] Task: `Validate Docker Image` is implemented by `gcloud container images describe gcr.io/PROJECT/docker_build_dependency_image --project PROJECT`, hiding output unless there is an error.
# [XPK] Task: `Validate Docker Image` succeeded.
# [XPK] Task: `Creating Workload` is implemented by `kubectl apply -f /tmp/tmpiwxlv270`, streaming output live.
# [XPK] Waiting for `Creating Workload`, for 0 seconds
# jobset.jobset.x-k8s.io/JOB_NAME created
# [XPK] Task: `Creating Workload` terminated with code `0`
# [XPK] Follow your workload here: https://console.cloud.google.com/kubernetes/service/us-central2/CLUSTER/default/JOB_NAME/details?project=PROJECT
# [XPK] Exiting XPK cleanly
#
# ####################################################
# # Logs expected from the above link showcasing the #
# ### model is training and losses are decreasing  ###
# ####################################################
# ...
# 2023-10-03 11:48:40.025 PDT
# completed step: 1, seconds: 8.693, TFLOP/s: 1.731, loss: 9.206
# 2023-10-03 11:48:40.308 PDT
# completed step: 2, seconds: 0.206, TFLOP/s: 72.958, loss: 9.352
# 2023-10-03 11:48:40.589 PDT
# completed step: 3, seconds: 0.282, TFLOP/s: 53.464, loss: 9.768
# 2023-10-03 11:48:40.871 PDT
# completed step: 4, seconds: 0.285, TFLOP/s: 52.876, loss: 8.903
# 2023-10-03 11:48:41.152 PDT
# completed step: 5, seconds: 0.278, TFLOP/s: 54.058, loss: 9.663
# ...
# ...
# completed step: 95, seconds: 0.281, TFLOP/s: 53.562, loss: 6.490
# 2023-10-03 11:49:06.768 PDT
# completed step: 96, seconds: 0.282, TFLOP/s: 53.430, loss: 6.532
# 2023-10-03 11:49:07.049 PDT
# completed step: 97, seconds: 0.281, TFLOP/s: 53.572, loss: 6.578
# 2023-10-03 11:49:07.330 PDT
# completed step: 98, seconds: 0.281, TFLOP/s: 53.596, loss: 6.731
# 2023-10-03 11:49:07.612 PDT
# completed step: 99, seconds: 0.281, TFLOP/s: 53.532, loss: 6.880
# ...
# ...
