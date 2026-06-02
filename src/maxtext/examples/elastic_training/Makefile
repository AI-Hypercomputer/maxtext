# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -e -c

-include .env
export

# Defaults
REGION          ?= us-central1
ZONE            ?= us-central1-a
CLUSTER_NAME    ?= maxtext-elastic-training
# Empty => let the RAPID release channel pick the current GKE version (recommended).
# Set a specific version (e.g. 1.35.3-gke.1993000) only if you need to pin one.
CLUSTER_VERSION ?=
BUCKET_NAME     ?= maxtext-elastic-training-$(PROJECT_ID)
DATASET         ?= glaive
DEBUGGING       ?= false
RUN_NAME        ?= elastic-qwen3-0.6b
TPU_RESERVATION ?=
# Use Spot (preemptible) TPUs (cheaper, reclaimable). Leave empty for on-demand.
TPU_SPOT        ?= false
# Set CONFIRM=yes to skip the interactive cost prompt before `make infra` (for CI).
CONFIRM         ?=

# TPU config (fixed for this demo)
TPU_MACHINE_TYPE := ct5lp-hightpu-4t
TPU_TOPOLOGY     := 4x4
NUM_TPU_SLICES   := 3
NODES_PER_SLICE  := 4
CPU_MACHINE_TYPE := n2-standard-64
NUM_CPU_NODES    := 1

# Derived
AR_REGION ?= $(REGION)
AR_REPO   ?= maxtext-elastic
IMAGE     := $(AR_REGION)-docker.pkg.dev/$(PROJECT_ID)/$(AR_REPO)/maxtext-runner:latest

# Formatting
B := \033[1m
D := \033[2m
C := \033[36m
G := \033[32m
Y := \033[33m
R := \033[31m
Z := \033[0m
LINE := $(D)──────────────────────────────────────────────────────────────────$(Z)

# Targets
.PHONY: help setup infra deploy demo status logs fail destroy

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(C)%-15s$(Z) %s\n", $$1, $$2}'

# Setup
setup: ## Generate terraform.tfvars from .env
	@test -n "$(PROJECT_ID)" || { echo -e "  $(R)✗$(Z) PROJECT_ID not set. Run: cp .env.example .env"; exit 1; }
	test -n "$(HF_TOKEN)" || { echo -e "  $(R)✗$(Z) HF_TOKEN not set in .env"; exit 1; }
	printf '%s\n' \
		'project_id       = "$(PROJECT_ID)"' \
		'region           = "$(REGION)"' \
		'zone             = "$(ZONE)"' \
		'cluster_name     = "$(CLUSTER_NAME)"' \
		'cluster_version  = "$(CLUSTER_VERSION)"' \
		'network          = "default"' \
		'subnetwork       = "default"' \
		'tpu_machine_type = "$(TPU_MACHINE_TYPE)"' \
		'tpu_topology     = "$(TPU_TOPOLOGY)"' \
		'num_tpu_slices   = $(NUM_TPU_SLICES)' \
		'nodes_per_slice  = $(NODES_PER_SLICE)' \
		'cpu_machine_type = "$(CPU_MACHINE_TYPE)"' \
		'num_cpu_nodes    = $(NUM_CPU_NODES)' \
		'gcs_bucket_name  = "$(BUCKET_NAME)"' \
		'hf_token         = "$(HF_TOKEN)"' \
		'tpu_reservation  = "$(TPU_RESERVATION)"' \
		'tpu_spot         = $(TPU_SPOT)' \
		> terraform/terraform.tfvars
	echo -e "  $(G)✓$(Z) Generated terraform/terraform.tfvars"

# Infrastructure
infra: setup ## Provision GKE cluster, TPU nodes, GCS bucket
	@echo -e "\n$(LINE)"
	echo -e "  $(B)$(C)Provisioning infrastructure$(Z)"
	echo -e "$(LINE)\n"
	# Cost guard: $(NUM_TPU_SLICES) x v5litepod-16 + $(CPU_MACHINE_TYPE) bill ~\$$61/hr while up
	# (on-demand list price; verify current TPU pricing). Skip the prompt with: make infra CONFIRM=yes
	if [ "$(CONFIRM)" != "yes" ]; then
		echo -e "  $(Y)!$(Z) This provisions $(NUM_TPU_SLICES) TPU slices + 1 CPU node, about $(B)\$$61/hr$(Z) until you run $(B)make destroy$(Z)."
		read -p "    Continue? [y/N] " ans
		case "$$ans" in [yY]*) ;; *) echo -e "  $(D)Aborted.$(Z)"; exit 1;; esac
		echo ""
	fi
	cd terraform && terraform init -input=false && terraform apply -auto-approve
	echo ""
	echo -e "  $(G)✓$(Z) Infrastructure ready"
	echo -e "  $(D)Cluster: $(CLUSTER_NAME) | Zone: $(ZONE) | TPU slices: $(NUM_TPU_SLICES)$(Z)"

# Deploy
deploy: ## Run Cloud Build pipeline (build, install APIs, deploy training)
	@echo -e "\n$(LINE)"
	echo -e "  $(B)$(C)Running Cloud Build pipeline$(Z)"
	echo -e "$(LINE)\n"
	# Use the dedicated build SA from Terraform, not the legacy default compute SA
	# (which new GCP projects disable). Falls back gracefully if outputs are absent.
	BUILD_SA=$$(cd terraform && terraform output -raw cloudbuild_service_account 2>/dev/null || true)
	SA_FLAG=""
	if [ -n "$$BUILD_SA" ]; then SA_FLAG="--service-account=projects/$(PROJECT_ID)/serviceAccounts/$$BUILD_SA"; fi
	gcloud builds submit --config=deploy/deploy.yaml $$SA_FLAG \
		--substitutions=_CLUSTER_NAME=$(CLUSTER_NAME),_CLUSTER_ZONE=$(ZONE),_BUCKET_NAME=$(BUCKET_NAME),_DATASET=$(DATASET),_DEBUGGING=$(DEBUGGING),_RUN_NAME=$(RUN_NAME) \
		--project=$(PROJECT_ID)
	echo ""
	echo -e "  $(G)✓$(Z) Cloud Build complete"

# Status & Logs
status: ## Show pod status
	@gcloud container clusters get-credentials "$(CLUSTER_NAME)" \
		--zone="$(ZONE)" --project="$(PROJECT_ID)" --quiet 2>/dev/null
	kubectl get pods

logs: ## Tail training logs
	@gcloud container clusters get-credentials "$(CLUSTER_NAME)" \
		--zone="$(ZONE)" --project="$(PROJECT_ID)" --quiet 2>/dev/null
	POD=$$(kubectl get pods -l job-name=pw-elastic-pathways-head-0 \
		-o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
	kubectl logs "$$POD" -c main --follow

# Simulate failure
SLICE ?= 2
fail: ## Simulate slice failure (SLICE=2 by default)
	@gcloud container clusters get-credentials "$(CLUSTER_NAME)" \
		--zone="$(ZONE)" --project="$(PROJECT_ID)" --quiet 2>/dev/null
	# Wait for a safe window between checkpoint writes.
	# Checkpoint writes take ~10s. Elastic retry resumes cleanly between writes,
	# so we time the termination for that window.
	echo -e "  $(D)Waiting for safe checkpoint window (step 140-395)...$(Z)"
	for i in $$(seq 1 600); do
		POD=$$(kubectl get pods -l job-name=pw-elastic-pathways-head-0 \
			-o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
		if [ -n "$$POD" ]; then
			LAST=$$(kubectl logs "$$POD" -c main --tail=1 2>/dev/null || true)
			if echo "$$LAST" | grep -q "completed step:"; then
				STEP=$$(echo "$$LAST" | grep -o 'step: [0-9]*' | grep -o '[0-9]*')
				MOD=$$((STEP % 100))
				if [ "$$STEP" -ge 140 ] && [ $$MOD -ge 40 ] && [ $$MOD -le 95 ]; then
					echo -e "\n  $(G)✓$(Z) Step $$STEP, safe window"
					break
				fi
				echo -ne "\r  $(D)Step $$STEP, waiting...$(Z)  "
			fi
		fi
		sleep 2
	done
	VICTIM=$$(kubectl get pods -l "job-name=pw-elastic-worker-$(SLICE)" \
		-o jsonpath='{.items[0].metadata.name}')
	echo -e "  $(R)Terminating$(Z) $(B)$$VICTIM$(Z) $(D)(slice $(SLICE))$(Z)"
	# --grace-period=0 --force: SIGKILL immediately. Default 30s grace lets the
	# worker keep serving while "Terminating"; at ~0.2s/step that is ~150 steps
	# before Pathways even notices. A real hardware failure does not drain.
	kubectl delete pod "$$VICTIM" --grace-period=0 --force
	echo -e "  $(G)✓$(Z) Pod deleted, watch recovery with: make logs"

# Destroy
destroy: ## Tear down everything
	@echo -e "\n$(LINE)"
	echo -e "  $(B)$(C)Destroying infrastructure$(Z)"
	echo -e "$(LINE)\n"
	gcloud container clusters get-credentials "$(CLUSTER_NAME)" \
		--zone="$(ZONE)" --project="$(PROJECT_ID)" --quiet 2>/dev/null || true
	kubectl delete jobset pw-elastic --ignore-not-found=true 2>/dev/null || true
	cd terraform && terraform destroy -auto-approve
	echo ""
	echo -e "  $(G)✓$(Z) All resources destroyed"

# Demo (full interactive flow)
demo: ## Run full elastic training demo (deploy → train → fail → recover)
	@case "$(MAKE_VERSION)" in 3.8[01]*) printf '  $(R)✗$(Z) GNU Make $(MAKE_VERSION) is too old (need ≥ 3.82 for .ONESHELL)\n  $(D)On macOS: brew install make, then run: gmake demo$(Z)\n'; exit 1;; esac; set -uo pipefail
	# Formatting helpers
	step_header() {
		echo ""
		echo -e "$(LINE)"
		echo -e "  $(B)$(C)[$$1/5]$(Z) $(B)$$2$(Z)"
		echo -e "$(LINE)"
		echo ""
	}
	ok()     { echo -e "  $(G)✓$(Z) $$*"; }
	info()   { echo -e "  $(D)$$*$(Z)"; }
	detail() { echo -e "    $(D)$$*$(Z)"; }
	warn()   { echo -e "  $(Y)!$(Z) $$*"; }
	head_pod() {
		kubectl get pods -l job-name=pw-elastic-pathways-head-0 \
			-o jsonpath='{.items[0].metadata.name}' 2>/dev/null
	}
	pods_ready() {
		local count
		count=$$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=pw-elastic \
			--no-headers 2>/dev/null | grep -c "Running" || true)
		[ "$$count" -ge 13 ]
	}
	training_started() {
		local pod
		pod=$$(head_pod)
		[ -n "$$pod" ] && kubectl logs "$$pod" -c main --tail=5 2>/dev/null | grep -q "completed step:"
	}
	spin() {
		local msg="$$1" check_fn="$$2" timeout="$${3:-300}" elapsed=0
		local chars='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
		while [ $$elapsed -lt $$timeout ]; do
			for (( i=0; i<$${#chars}; i++ )); do
				echo -ne "\r  $(D)$${chars:$$i:1} $$msg ($${elapsed}s)$(Z)  "
				sleep 0.5
				elapsed=$$((elapsed + 1))
				if [ $$((elapsed % 5)) -eq 0 ] && eval "$$check_fn"; then
					echo -ne "\r\033[K"
					return 0
				fi
			done
		done
		echo -ne "\r\033[K"
		return 1
	}
	# Intro
	echo ""
	echo -e "  $(B)Elastic Training Demo$(Z)"
	echo -e "  $(D)Fault-tolerant training on Cloud TPUs with Pathways$(Z)"
	echo ""
	echo -e "  Dataset     $(B)$(DATASET)$(Z)"
	echo -e "  Debugging   $(B)$(DEBUGGING)$(Z)"
	echo -e "  TPU slices  $(B)3 x v5litepod-16$(Z) (48 chips)"
	echo -e "  Run name    $(D)$(RUN_NAME)$(Z)"
	echo -e "  Bucket      $(D)gs://$(BUCKET_NAME)$(Z)"
	# Step 1: Deploy
	step_header 1 "Deploy JobSet"
	if [ "$(DATASET)" = "glaive" ]; then
		if ! gcloud storage ls "gs://$(BUCKET_NAME)/data/glaive-fc-v2/" --project="$(PROJECT_ID)" >/dev/null 2>&1; then
			echo -e "  $(R)✗$(Z) Glaive data not found. Run 'make deploy' first."
			exit 1
		fi
	fi
	gcloud container clusters get-credentials "$(CLUSTER_NAME)" \
		--zone="$(ZONE)" --project="$(PROJECT_ID)" --quiet 2>/dev/null
	kubectl delete jobset pw-elastic --ignore-not-found --wait=true 2>/dev/null || true
	bash deploy/jobs/training/submit.sh
	ok "JobSet created"
	info "elastic_enabled=true  elastic_timeout=300s  elastic_max_retries=10"
	# Step 2: Wait for pods
	step_header 2 "Wait for pods"
	info "Expecting 13 pods: 1 head + 4 workers x 3 slices"
	echo ""
	if spin "Waiting for all pods to schedule and start" "pods_ready" 300; then
		ok "All pods running"
	else
		echo -e "  $(R)✗$(Z) Timed out waiting for pods"
		kubectl get pods
		exit 1
	fi
	echo ""
	kubectl get pods -l jobset.sigs.k8s.io/jobset-name=pw-elastic --no-headers 2>/dev/null \
		| awk '{printf "    %-45s %s\n", $$1, $$3}'
	echo ""
	# Step 3: Wait for training
	step_header 3 "Wait for training"
	info "XLA compilation + first step takes ~3 minutes"
	echo ""
	if spin "Compiling and starting training" "training_started" 420; then
		ok "Training is running"
	else
		warn "Timed out, training may still be compiling"
	fi
	echo ""
	POD=$$(head_pod)
	if [ -n "$$POD" ]; then
		kubectl logs "$$POD" -c main --tail=10 2>/dev/null \
			| grep "completed step:" \
			| tail -3 \
			| while read -r line; do
				step=$$(echo "$$line" | grep -o 'step: [0-9]*' | grep -o '[0-9]*')
				loss=$$(echo "$$line" | grep -o 'loss: [0-9.]*' | grep -o '[0-9.]*')
				tflops=$$(echo "$$line" | grep -o 'TFLOP/s/device: [0-9.]*' | grep -o '[0-9.]*')
				echo -e "    step $(B)$$step$(Z)  loss $(B)$$loss$(Z)  $(D)$$tflops TFLOP/s/device$(Z)"
			done
	fi
	echo ""
	# Step 4: Simulate failure
	step_header 4 "Simulate slice failure"
	# Wait for a safe window: at least one non-zero checkpoint committed and no
	# write in flight. checkpoint_period=100, write takes ~10s with qwen3-0.6b.
	info "Waiting for checkpoint 100 to commit (safe termination window)"
	echo ""
	STEP_BEFORE="?"
	for i in $$(seq 1 200); do
		POD=$$(head_pod)
		L=$$(kubectl logs "$$POD" -c main --tail=5 2>/dev/null | grep "completed step:" | tail -1 || true)
		if [ -n "$$L" ]; then
			S=$$(echo "$$L" | grep -o 'step: [0-9]*' | grep -o '[0-9]*')
			M=$$((S % 100))
			echo -ne "\r  $(D)step $$S$(Z)  "
			if [ "$$S" -ge 140 ] && [ $$M -ge 40 ] && [ $$M -le 95 ]; then
				STEP_BEFORE=$$S
				echo -ne "\r\033[K"
				break
			fi
		fi
		sleep 1
	done
	ok "Training at step $(B)$$STEP_BEFORE$(Z), checkpoint 100 committed, safe to terminate"
	echo ""
	VICTIM=$$(kubectl get pods -l "job-name=pw-elastic-worker-2" \
		-o jsonpath='{.items[0].metadata.name}')
	echo -e "  $(R)Terminating$(Z) $(B)$$VICTIM$(Z) $(D)(slice 2, --grace-period=0)$(Z)"
	kubectl delete pod "$$VICTIM" --grace-period=0 --force > /dev/null 2>&1
	echo ""
	info "What happens now:"
	detail "1. Pathways heartbeat times out (~10s) → connection lost"
	detail "2. elastic_retry catches the error in-process"
	detail "3. Incomplete checkpoints cleaned up on GCS"
	detail "4. Replacement worker pod schedules on slice 2"
	detail "5. Checkpoint restored, training rewinds and resumes"
	echo ""
	# Step 5: Verify recovery
	step_header 5 "Verify recovery"
	wait_log() {
		kubectl logs "$$(head_pod)" -c main 2>/dev/null | grep -qE "$$1"
	}
	if spin "Waiting for Pathways to detect slice down" "wait_log 'Slice down event detected'" 90; then
		ok "Slice down event detected"
	else
		warn "No slice-down event, check: make logs"
	fi
	if spin "Waiting for checkpoint restore" "wait_log 'Restoring checkpoint from'" 90; then
		ok "Checkpoint restored"
	else
		warn "Restore not seen, check: make logs"
	fi
	# Training resumed = step regression observed (rewind) and tail shows steps
	step_regression() {
		kubectl logs "$$(head_pod)" -c main 2>/dev/null \
			| grep -o 'completed step: [0-9]*' | grep -o '[0-9]*' \
			| awk 'NR>1 && $$1<prev {found=1} {prev=$$1} END{exit !found}'
	}
	if spin "Waiting for training to resume" "step_regression" 120; then
		ok "Training resumed"
	else
		warn "Still recovering, check: make logs"
	fi
	echo ""
	POD=$$(head_pod)
	kubectl logs "$$POD" -c main 2>/dev/null \
		| grep -E "Slice down event|Elastic attempt [2-9]|Keeping gs://|Restoring checkpoint from|Sufficient slices" \
		| sed 's/^.*] //' \
		| while read -r line; do detail "$$line"; done
	echo ""
	REWIND=$$(kubectl logs "$$POD" -c main 2>/dev/null \
		| grep -o 'completed step: [0-9]*' | grep -o '[0-9]*' \
		| awk 'NR>1 && $$1<prev {print prev " → " $$1; exit} {prev=$$1}')
	STEP_AFTER=$$(kubectl logs "$$POD" -c main --tail=20 2>/dev/null \
		| grep "completed step:" | tail -1 | grep -o 'step: [0-9]*' | grep -o '[0-9]*' || echo "?")
	ok "Step rewind: $(B)$$REWIND$(Z) $(D)(restored from last good checkpoint)$(Z)"
	ok "Now training at step $(B)$$STEP_AFTER$(Z), same head pod, no job restart"
	echo ""
	PRX=$$(kubectl get pod "$$POD" -o jsonpath='{.status.initContainerStatuses[?(@.name=="pathways-proxy")].restartCount}' 2>/dev/null)
	JSR=$$(kubectl get jobset pw-elastic -o jsonpath='{.status.restarts}' 2>/dev/null)
	detail "pathways-proxy restartCount = $${PRX:-0}"
	detail "jobset restarts            = $${JSR:-0}"
	echo ""
	ok "GCS checkpoints:"
	{ gcloud storage ls "gs://$(BUCKET_NAME)/output/$(RUN_NAME)/checkpoints/" 2>/dev/null || true; } \
		| grep -E '/[0-9]+/$$' \
		| while read -r dir; do
			detail "$$(basename "$${dir%/}")"
		done || true
	echo ""
	echo -e "$(LINE)"
	echo -e "  $(G)$(B)Demo complete$(Z)"
	echo -e "$(LINE)"
	echo ""
	info "Training is still running. Next:"
	detail "make logs       , tail training output"
	detail "make fail SLICE=1, simulate another failure"
	detail "make destroy    , tear down everything"
