# Enable GCP Workload Observabiltiy
This guide provides an overview on how to enable GCP workload observability for your MaxText workload.

## Overview
Google offers a monitoring and alerting feature that is well suited for critical MaxText workloads sensitive to infrastructure changes.
Once enabled, metrics will be automatically sent to [Cloud Monarch](https://research.google/pubs/monarch-googles-planet-scale-in-memory-time-series-database/) for monitoring.
If a metric hits its pre-defined threshold, the Google Cloud on-call team will be alerted to see if any action is needed. 

The feature currently supports heartbeat and performance (training step time in seconds) metrics. In the near future, support for the goodput metric will also be added.
Users should work with their Customer Engineer (CE) and the Google team to define appropriate thresholds for the performance metrics.

This guide layouts how to enable the feature for your MaxText workload.

## Enabling GCP Workload Observabiltiy
User can control which metric they want to report via config:

### Heartbeat metric 
- This metric will be a boolean flag.
- To turn on this metric, set `report_heartbeat_metric_for_gcp_monitoring` to `True`
- To control the frequency of heartbeat reporting (default is every 5 seconds), set `heartbeat_reporting_interval_in_seconds` to your desired value.

### Performance metric
- This metric will be a double, capturing the training step time in seconds.
- To turn on this metric, set `report_performance_metric_for_gcp_monitoring` to `True`

For an example, please refer to [base.yml](../MaxText/configs/base.yml).