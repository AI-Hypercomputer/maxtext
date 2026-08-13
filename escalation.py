"""Escalation Report.

Diagnosis:
Kubernetes pod not found during checkpoint shape validation.

Error Type:
Infrastructure

Evidence:
- pods "checkpoint-shape-validation-pod-45kmf6lu" not found
- code:404

Structured Plan:
Step 1: Escalate to the infrastructure team to investigate the missing Kubernetes pod.
Step 2: Check Kubernetes cluster logs and events for pod eviction or deletion reasons.
Step 3: Rerun the Airflow DAG once the infrastructure issue is resolved.
"""
