import re
import pandas as pd
from google.cloud import logging
from google.cloud.logging import DESCENDING
from datetime import datetime, timedelta, timezone

def get_reshard_data():
    client = logging.Client(project="cloud-tpu-multipod-dev")
    
    # 1. Define a narrow time window (last 24 hours)
    # This prevents the API from searching the entire history of the project
    start_time = (datetime.now(timezone.utc) - timedelta(days=5)).strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # 2. Build the exact filter that worked in your UI
    # We replace SEARCH() with textPayload: which is the API equivalent
    log_filter = (
        f'resource.type="k8s_container" '
        f'resource.labels.location="us-central1" '
        f'resource.labels.cluster_name="zxhe-super-xpk-bid" '
        f'resource.labels.namespace_name="default" '
        f'resource.labels.pod_name:"sanbao-rl-0307-2" '
        f'severity>=DEFAULT '
        f'timestamp >= "{start_time}" '
        f'SEARCH("Reshard finished in")'
    )

    print(f"Querying logs from the last 24 hours (Newest first)...")
    
    # 3. Use order_by=DESCENDING to find recent logs immediately
    entries = client.list_entries(filter_=log_filter, order_by=DESCENDING)
    
    pattern = r"Reshard finished in (\d+\.?\d*)s"
    results = []

    try:
        for entry in entries:
            payload = entry.payload
            payload_str = None
            if isinstance(payload, dict):
                payload_str = payload.get("message") or str(payload)
            else:
                payload_str = str(payload)
            if payload_str:
                match = re.search(pattern, payload_str)
                if match:
                    results.append({
                        "timestamp": entry.timestamp,
                        "reshard_sec": float(match.group(1)),
                        "pod": entry.resource.labels.get("pod_name")
                    })
    except Exception as e:
        print(f"Error during API call: {e}")

    if not results:
        print("Still no logs found. Try this final check:")
        print(f"1. Run: gcloud logging read '{log_filter}' --limit=1")
        print("2. If that returns nothing, your local gcloud credentials don't have permission for this project.")
        return None

    df = pd.DataFrame(results).sort_values("timestamp")
    df.to_csv("reshard_times.csv", index=False)
    
    print(f"Success! Found {len(df)} events.")
    print(df.describe())
    return df

df = get_reshard_data()