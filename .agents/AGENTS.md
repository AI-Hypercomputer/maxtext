# Behavioral Rules for AI-Hypercomputer/maxtext

* **No direct Docker commands:** Do not execute `docker` CLI commands (like `docker build`, `docker run`) directly or prepend them with `sudo`. If a container building or running step is needed, present the command to the USER and ask them to run it themselves.
* **Never delete other people's jobs:** Do not delete, cancel, or terminate Kubernetes resources (e.g. pods, jobsets, slices) or workloads that do not belong to your active workload. Always let them be preempted/cleaned up naturally or ask the USER.

