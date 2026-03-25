.ONESHELL:
SHELL       := bash
.SHELLFLAGS := -euo pipefail -c

# ==============================================================================
# Variables  (override on CLI: make <target> VAR=value)
# ==============================================================================
PROJECT_ID    := zenteiq-lxp-1722918338008
REGION        := asia-south1
ZONE          := $(REGION)-b
CLUSTER_NAME  ?= brahmai-v6e-lustre
RESERVATION   := cloudtpu-20260317070000-2123307204
NETWORK       := zenteiq-tpu-vpc
WORKLOAD_NAME ?= brahmai-test

DEVICE     := tpu
TPU_TYPE   := v6e-16
NUM_SLICES := 2

IP_RANGE      := lustre-peering-range
SOURCE_RANGES := 10.0.0.0/8,172.16.0.0/12,192.168.0.0/16

STORAGE_NAME       := ziq-lustre
STORAGE_THROUGHPUT := 1000
STORAGE_CAPACITY   := 36000
STORAGE_FS         := ziqfs
MOUNTPOINT         := /lustre-data
MANIFEST           := ./lustre.yaml

REPOSITORY   := $(REGION)-docker.pkg.dev/$(PROJECT_ID)/maxtext-training
DOCKER_IMAGE := $(REPOSITORY)/brahmai:latest
MAXTEXT_DIR  := src

# Positional args from CLI goals:
#   make workload-deploy  my-script.sh          (ARG1=my-script.sh)
#   make workload-delete  my-job                (ARG1=my-job)
#   make cluster-create-manual 4x4 ct6e-std-4t  (ARG1=4x4, ARG2=ct6e-std-4t)
#   make workload-deploy  SCRIPT=my-script.sh   (explicit override still works)
ARG1 := $(word 2,$(MAKECMDGOALS))
ARG2 := $(word 3,$(MAKECMDGOALS))

SCRIPT        ?= $(ARG1)
TOPOLOGY      ?= $(ARG1)
MACHINE       ?= $(ARG2)

# Absorb extra CLI goals so make doesn't error on unknown targets
%:
	@:

.PHONY: help login setup \
  vpc-ip vpc-peering vpc-firewall auto-vpc \
  lustre-create lustre-list lustre-automount lustre-manifest-generate \
  lustre-status lustre-test auto-lustre \
  cluster-list cluster-create cluster-create-manual cluster-enable-lustre \
  cluster-delete auto-cluster \
  workload-setup workload-build workload-push workload-deploy \
  workload-list workload-delete workload-logs workload-status workload-events \
  auto-workload shell doc

help:
	@echo "Usage: make <target> [VAR=value ...]"
	@echo "       make -n <target>          # dry-run: print commands without executing"
	@echo ""
	@echo "Auth:      login  setup"
	@echo "VPC:       vpc-ip  vpc-peering  vpc-firewall  auto-vpc"
	@echo "Lustre:    lustre-create  lustre-list  lustre-automount"
	@echo "           lustre-manifest-generate  lustre-status  lustre-test  auto-lustre"
	@echo "Cluster:   cluster-list  cluster-create  cluster-create-manual"
	@echo "           cluster-enable-lustre  cluster-delete  auto-cluster"
	@echo "Workload:  workload-setup  workload-build  workload-push  workload-deploy"
	@echo "           workload-list  workload-delete  workload-logs"
	@echo "           workload-status  workload-events  auto-workload"
	@echo ""
	@echo "Params:    SCRIPT= WORKLOAD_NAME= CLUSTER_NAME= TOPOLOGY= MACHINE="
	@echo "Other:     shell  doc"

shell:
	export PROJECT_ID=$(PROJECT_ID) REGION=$(REGION) ZONE=$(ZONE) \
	  CLUSTER_NAME=$(CLUSTER_NAME) RESERVATION=$(RESERVATION) NETWORK=$(NETWORK) \
	  WORKLOAD_NAME=$(WORKLOAD_NAME) DEVICE=$(DEVICE) TPU_TYPE=$(TPU_TYPE) \
	  NUM_SLICES=$(NUM_SLICES) IP_RANGE=$(IP_RANGE) SOURCE_RANGES=$(SOURCE_RANGES) \
	  STORAGE_NAME=$(STORAGE_NAME) STORAGE_THROUGHPUT=$(STORAGE_THROUGHPUT) \
	  STORAGE_CAPACITY=$(STORAGE_CAPACITY) STORAGE_FS=$(STORAGE_FS) \
	  MOUNTPOINT=$(MOUNTPOINT) MANIFEST=$(MANIFEST) \
	  REPOSITORY=$(REPOSITORY) DOCKER_IMAGE=$(DOCKER_IMAGE) MAXTEXT_DIR=$(MAXTEXT_DIR)
	exec $${SHELL:-bash}

doc:
	@echo ""
	@echo "======================================================================"
	@echo "  Cluster: $(CLUSTER_NAME)   Workload: $(WORKLOAD_NAME)"
	@echo "======================================================================"
	@echo ""
	@echo "  -- Credentials --"
	@echo "  gcloud container clusters get-credentials $(CLUSTER_NAME) --zone=$(ZONE) --project=$(PROJECT_ID)"
	@echo ""
	@echo "  -- Pods --"
	@echo "  kubectl get pods -n default -o wide"
	@echo "  kubectl get pods -n default -l xpk.google.com/workload=$(WORKLOAD_NAME) -o wide"
	@echo "  kubectl describe pod <POD> -n default"
	@echo ""
	@echo "  -- Logs --"
	@echo "  kubectl logs -n default <POD> -c jax-tpu -f"
	@echo "  kubectl logs -n default <POD> -c jax-tpu --previous"
	@echo ""
	@echo "  -- Exec --"
	@echo "  kubectl exec -it <POD> -n default -c jax-tpu -- bash"
	@echo ""
	@echo "  -- Jobset / Workload --"
	@echo "  kubectl get jobset -n default"
	@echo "  kubectl describe jobset $(WORKLOAD_NAME) -n default"
	@echo "  kubectl get events -n default --field-selector involvedObject.name=$(WORKLOAD_NAME) --sort-by=.lastTimestamp"
	@echo ""
	@echo "  -- Nodes --"
	@echo "  kubectl get nodes -o wide"
	@echo "  kubectl describe node <NODE>"
	@echo ""
	@echo "  -- Lustre --"
	@echo "  kubectl get pv xpk-lustre-pv"
	@echo "  kubectl get pvc xpk-lustre-pvc -n default"
	@echo "======================================================================"
	@echo ""

# ==============================================================================
# Auth
# ==============================================================================
login:
	gcloud auth login
	gcloud auth application-default login
	gcloud auth configure-docker $(REGION)-docker.pkg.dev --quiet

setup:
	gcloud config set project $(PROJECT_ID)
	gcloud config set compute/zone $(ZONE)

# ==============================================================================
# VPC
# ==============================================================================
vpc-ip:
	gcloud compute addresses create $(IP_RANGE) \
	  --global --purpose=VPC_PEERING --prefix-length=20 \
	  --network=$(NETWORK) --project=$(PROJECT_ID)

vpc-peering:
	gcloud services vpc-peerings connect \
	  --network=$(NETWORK) --ranges=$(IP_RANGE) \
	  --service=servicenetworking.googleapis.com --project=$(PROJECT_ID)

vpc-firewall:
	gcloud compute firewall-rules create allow-lustre-all-internal \
	  --allow=tcp:988,tcp:6988 --network=$(NETWORK) \
	  --source-ranges=$(SOURCE_RANGES) --project=$(PROJECT_ID)

auto-vpc: vpc-ip vpc-peering vpc-firewall

# ==============================================================================
# Lustre
# ==============================================================================
lustre-create:
	gcloud lustre instances create $(STORAGE_NAME) \
	  --per-unit-storage-throughput=$(STORAGE_THROUGHPUT) \
	  --capacity-gib=$(STORAGE_CAPACITY) \
	  --filesystem=$(STORAGE_FS) \
	  --location=$(ZONE) \
	  --network=projects/$(PROJECT_ID)/global/networks/$(NETWORK) \
	  --project=$(PROJECT_ID) --async

lustre-list:
	gcloud lustre instances describe $(STORAGE_NAME) \
	  --location=$(ZONE) --project=$(PROJECT_ID) \
	  --format="value(mountPoint)"

lustre-automount:
	xpk storage attach $(STORAGE_NAME) \
	  --cluster=$(CLUSTER_NAME) --project=$(PROJECT_ID) --zone=$(ZONE) \
	  --type=lustre --mount-point=$(MOUNTPOINT) \
	  --readonly=false --auto-mount=true --manifest=$(MANIFEST)

lustre-manifest-generate:
	LUSTRE_IP=$$(gcloud lustre instances describe $(STORAGE_NAME) \
	  --location=$(ZONE) --project=$(PROJECT_ID) \
	  --format='value(mountPoint)' 2>/dev/null | sed 's/@.*//')
	[[ -n "$$LUSTRE_IP" ]] || { echo "Error: could not get Lustre IP" >&2; exit 1; }
	echo "Resolved LUSTRE_IP=$$LUSTRE_IP"
	cat > $(MANIFEST) <<-EOF
	apiVersion: v1
	kind: PersistentVolume
	metadata:
	  name: xpk-lustre-pv
	spec:
	  storageClassName: ""
	  capacity:
	    storage: "$(STORAGE_CAPACITY)Gi"
	  accessModes:
	    - ReadWriteMany
	  persistentVolumeReclaimPolicy: Retain
	  volumeMode: Filesystem
	  claimRef:
	    namespace: default
	    name: xpk-lustre-pvc
	  csi:
	    driver: lustre.csi.storage.gke.io
	    volumeHandle: "projects/$(PROJECT_ID)/locations/$(ZONE)/instances/$(STORAGE_NAME)"
	    volumeAttributes:
	      ip: $$LUSTRE_IP
	      filesystem: $(STORAGE_FS)
	---
	apiVersion: v1
	kind: PersistentVolumeClaim
	metadata:
	  name: xpk-lustre-pvc
	  namespace: default
	spec:
	  accessModes:
	    - ReadWriteMany
	  storageClassName: ""
	  volumeName: xpk-lustre-pv
	  resources:
	    requests:
	      storage: "$(STORAGE_CAPACITY)Gi"
	EOF
	echo "Written to $(MANIFEST)"

lustre-status:
	gcloud lustre instances describe $(STORAGE_NAME) \
	  --location=$(ZONE) --project=$(PROJECT_ID) \
	  --format="table(name,state,capacityGib,network,mountPoint)"

lustre-test:
	kubectl run lustre-io-test --image=busybox --restart=Never --rm --attach \
	  --overrides='{"spec":{"containers":[{"name":"lustre-io-test","image":"busybox","command":["sh","-c","dd if=/dev/zero of=$(MOUNTPOINT)/test_write bs=1M count=512 oflag=direct && dd if=$(MOUNTPOINT)/test_write of=/dev/null bs=1M iflag=direct && rm -f $(MOUNTPOINT)/test_write && echo done"],"volumeMounts":[{"name":"vol","mountPath":"$(MOUNTPOINT)"}]}],"volumes":[{"name":"vol","persistentVolumeClaim":{"claimName":"xpk-lustre-pvc"}}],"restartPolicy":"Never"}}'

auto-lustre:
	$(MAKE) lustre-create
	echo "Waiting for Lustre instance to become ACTIVE..."
	until gcloud lustre instances describe $(STORAGE_NAME) \
	    --location=$(ZONE) --project=$(PROJECT_ID) \
	    --format="value(state)" 2>/dev/null | grep -q "^ACTIVE$$"; do
	  echo "... not ready yet, retrying in 30s"; sleep 30
	done
	$(MAKE) lustre-manifest-generate
	$(MAKE) lustre-automount

# ==============================================================================
# Cluster
# ==============================================================================
cluster-list:
	xpk cluster list --zone=$(ZONE) --project=$(PROJECT_ID)

cluster-create:
	xpk cluster create \
	  --cluster $(CLUSTER_NAME) \
	  --tpu-type=$(TPU_TYPE) --num-slices=$(NUM_SLICES) \
	  --reservation=$(RESERVATION) --zone=$(ZONE) --project=$(PROJECT_ID) \
	  --enable-lustre-csi-driver --skip-validation \
	  --custom-cluster-arguments="--network=$(NETWORK) --release-channel=None"

cluster-create-manual:
	gcloud beta container node-pools create $(DEVICE)-$(TPU_TYPE) \
	  --cluster=$(CLUSTER_NAME) --location=$(REGION) --node-locations=$(ZONE) \
	  --machine-type=$(MACHINE) --tpu-topology=$(TOPOLOGY) \
	  --reservation-affinity=specific --reservation=$(RESERVATION) \
	  --image-type=cos_containerd --project=$(PROJECT_ID) \
	  --scopes="storage-full,cloud-platform"
	xpk cluster adapt \
	  --cluster=$(CLUSTER_NAME) --tpu-type=$(TPU_TYPE) --num-slices=$(NUM_SLICES) \
	  --reservation=$(RESERVATION) --project=$(PROJECT_ID) --zone=$(ZONE) \
	  --enable-lustre-csi-driver

cluster-enable-lustre:
	gcloud container clusters update $(CLUSTER_NAME) \
	  --location=$(REGION) --project=$(PROJECT_ID) \
	  --update-addons=LustreCsiDriver=ENABLED

cluster-delete:
	xpk cluster delete --cluster $(CLUSTER_NAME)

auto-cluster: cluster-create lustre-automount

# ==============================================================================
# Workload
# ==============================================================================
workload-setup:
	uv venv --python 3.12 --seed
	uv pip install xpk

workload-build:
	docker build -t $(DOCKER_IMAGE) \
	  -f $(MAXTEXT_DIR)/dependencies/dockerfiles/maxtext_tpu_dependencies.Dockerfile \
	  $(MAXTEXT_DIR)

workload-push:
	docker push $(DOCKER_IMAGE)

workload-deploy:
	$(eval COMMAND_STR := $(shell sed 's/\\[[:space:]]*$$//' $(SCRIPT) | tr '\n' ' ' | tr -s ' '))
	xpk workload create \
	  --cluster $(CLUSTER_NAME) --workload $(WORKLOAD_NAME) \
	  --tpu-type=$(TPU_TYPE) --num-slices=$(NUM_SLICES) \
	  --reservation=$(RESERVATION) \
	  --project=$(PROJECT_ID) --zone=$(ZONE) \
	  --docker-image=$(DOCKER_IMAGE) --skip-validation \
	  --command="$(COMMAND_STR)"

workload-list:
	xpk workload list --cluster $(CLUSTER_NAME)

workload-delete:
	xpk workload delete --cluster $(CLUSTER_NAME) --workload $(WORKLOAD_NAME)

workload-logs:
	POD=$$(kubectl get pods \
	  -l "xpk.google.com/workload=$(WORKLOAD_NAME),batch.kubernetes.io/job-completion-index=0" \
	  -o name | head -n 1)
	[[ -n "$$POD" ]] || { echo "Error: no pods found for workload '$(WORKLOAD_NAME)'" >&2; exit 1; }
	kubectl logs "$$POD" -c jax-tpu -f

workload-status:
	kubectl get pods -n default -l "xpk.google.com/workload=$(WORKLOAD_NAME)" -o wide
	echo ""
	kubectl get jobset $(WORKLOAD_NAME) -n default \
	  -o jsonpath='{.status.conditions}' | python3 -m json.tool 2>/dev/null || \
	  kubectl get jobset $(WORKLOAD_NAME) -n default

workload-events:
	kubectl get events -n default \
	  --field-selector "involvedObject.name=$(WORKLOAD_NAME)" \
	  --sort-by='.lastTimestamp'

auto-workload:
	$(MAKE) workload-setup
	$(MAKE) workload-build
	$(MAKE) workload-push
	$(MAKE) workload-deploy SCRIPT=$(SCRIPT)
	$(MAKE) workload-status
