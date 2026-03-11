import argparse
import re
import pandas as pd
from google.cloud import logging
from google.cloud.logging import DESCENDING
from datetime import datetime, timedelta, timezone

def get_reshard_data(args):
    client = logging.Client(project="cloud-tpu-multipod-dev")
    
    # 1. Define a narrow time window (last 5 days)
    # This prevents the API from searching the entire history of the project
    start_time = (datetime.now(timezone.utc) - timedelta(days=5)).strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # 2. Build the filter to search for both reshard and weight sync times.
    # We replace SEARCH() with textPayload: which is the API equivalent
    log_filter = (
        f'resource.type="k8s_container" '
        f'resource.labels.location="us-central1" '
        f'resource.labels.cluster_name="zxhe-super-xpk-bid" '
        f'resource.labels.namespace_name="default" '
        f'resource.labels.pod_name:"{args.pod_name}" '
        f'severity>=DEFAULT '
        f'timestamp >= "{start_time}" '
        f'(SEARCH("Reshard finished in") OR SEARCH("Weight Syncing Time taken:"))'
    )

    print(f"Querying logs from the last 5 days (Newest first)...")
    
    # 3. Use order_by=DESCENDING to find recent logs immediately
    entries = client.list_entries(filter_=log_filter, order_by=DESCENDING)
    
    reshard_pattern = r"Reshard finished in (\d+\.?\d*)s"
    weight_sync_pattern = r"Weight Syncing Time taken: (\d+\.?\d*)s"
    reshard_results = []
    weight_sync_results = []

    try:
        for entry in entries:
            payload = entry.payload
            payload_str = None
            if isinstance(payload, dict):
                payload_str = payload.get("message") or str(payload)
            else:
                payload_str = str(payload)
            if payload_str:
                reshard_match = re.search(reshard_pattern, payload_str)
                if reshard_match:
                    reshard_results.append({
                        "timestamp": entry.timestamp,
                        "reshard_sec": float(reshard_match.group(1)),
                        "pod": entry.resource.labels.get("pod_name")
                    })
                
                weight_sync_match = re.search(weight_sync_pattern, payload_str)
                if weight_sync_match:
                    weight_sync_results.append({
                        "timestamp": entry.timestamp,
                        "weight_sync_sec": float(weight_sync_match.group(1)),
                        "pod": entry.resource.labels.get("pod_name")
                    })
    except Exception as e:
        print(f"Error during API call: {e}")

    if not reshard_results and not weight_sync_results:
        print("Still no logs found. Try this final check:")
        print(f"1. Run: gcloud logging read '{log_filter}' --limit=1")
        print("2. If that returns nothing, your local gcloud credentials don't have permission for this project.")
        return None

    mean_reshard_time = float('nan')
    if reshard_results:
        df = pd.DataFrame(reshard_results).sort_values("timestamp")
        # Only keep the third - tenth ones and compute the mean of them
        # Note: iloc[2:min(df.shape[0], args.max_steps)] gets indices 2 through 9 (8 items), corresponding to 3rd through 10th
        selected_df = df.iloc[3:min(df.shape[0], args.max_steps)]
        mean_reshard_time = selected_df["reshard_sec"].mean()

    mean_weight_sync_time = float('nan')
    if weight_sync_results:
        df = pd.DataFrame(weight_sync_results).sort_values("timestamp")
        selected_df = df.iloc[3:min(df.shape[0], args.max_steps)]
        mean_weight_sync_time = selected_df["weight_sync_sec"].mean()

    result_df = pd.DataFrame([{
        "pod_name": args.pod_name, 
        "mean_reshard_time": mean_reshard_time,
        "mean_weight_sync_time": mean_weight_sync_time
    }])
    # If the csv file already exists, append to it instead of overwriting
    try:
        existing_df = pd.read_csv("reshard_stats.csv")
        result_df = pd.concat([existing_df, result_df], ignore_index=True)
    except FileNotFoundError:
        pass
    result_df.to_csv("reshard_stats.csv", index=False)
    print(result_df)
    return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod_name", type=str, required=True, help="Pod name")
    parser.add_argument("--max_steps", type=int, default=10, help="Max steps")
    args = parser.parse_args()
    get_reshard_data(args)

# python ./src/maxtext/trainers/post_train/rl/extract_time.py --pod_name sanbao-rl-0310-19