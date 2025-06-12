#!/usr/bin/env python3
import os
import json
import argparse
import pandas as pd
import re
from typing import List, Dict, Any, Optional

# --- Configuration ---
DEFAULT_OUTPUT_DIR = "./benchmark_outputs_controlled"

def parse_json_file(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Safely load and parse a single JSON result file.
    Tries to get config from JSON first, then falls back to parsing the filename.
    """
    filename = os.path.basename(filepath)
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        summary = data.get('summary', {})
        status = "Success" if summary.get('completed_requests', 0) > 0 and summary.get('throughput_tokens_per_s') else "Failed"

    except (json.JSONDecodeError, IOError):
        data = {}
        summary = {}
        status = "Failed"

    # --- Data Extraction with Fallback Logic ---
    config = data.get('config', {})
    
    # --- CORRECTED HELPER FUNCTIONS ---
    # Helper for extracting NUMERIC values from filename
    def extract_int_from_filename(pattern: str) -> Optional[int]:
        match = re.search(pattern, filename)
        return int(match.group(1)) if match else None

    # Helper for extracting STRING values from filename
    def extract_str_from_filename(pattern: str) -> Optional[str]:
        match = re.search(pattern, filename)
        return match.group(1) if match else None
    # ------------------------------------

    # 1. Get from JSON, 2. Fallback to filename, 3. Use default
    # Use the correct helper for each type
    attention = config.get('attention') or extract_str_from_filename(r'controlled_([a-z_]+)_bs')
    bs_global = config.get('batch_size') or extract_int_from_filename(r'_bs([0-9]+)')
    prefill = config.get('max_prefill') or extract_int_from_filename(r'_pf([0-9]+)')
    target = config.get('max_target') or extract_int_from_filename(r'_tg([0-9]+)')
    pages = config.get('pagedattn_num_pages') or extract_int_from_filename(r'_np([0-9]+)')
    tpp = config.get('pagedattn_tokens_per_page') or extract_int_from_filename(r'_tpp([0-9]+)')

    flat_data = {
        'status': status,
        'attention': attention or 'N/A',
        'bs_global': bs_global,
        'prefill': prefill,
        'target': target,
        'pages': pages if pages is not None else '-',
        'tpp': tpp if tpp is not None else '-',
        'throughput': summary.get('throughput_tokens_per_s'),
        'decode_throughput': summary.get('decode_throughput_tokens_per_s'),
        'p99_ttft_ms': summary.get('p99_time_to_first_token_ms'),
        'p50_step_ms': summary.get('p50_time_per_decode_step_ms'),
        'filename': filename
    }
    
    return flat_data

def main():
    """Main function to find, parse, and summarize benchmark results."""
    parser = argparse.ArgumentParser(
        description="Summarize MaxText benchmark results from JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dir', type=str, default=DEFAULT_OUTPUT_DIR,
        help="Directory containing the benchmark output JSON files."
    )
    parser.add_argument(
        '--ignore-failed', action='store_true',
        help="If set, only display results from successful runs."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Error: Output directory '{args.dir}' not found.")
        return

    all_results: List[Dict[str, Any]] = []
    print(f"Scanning for results in '{args.dir}'...")

    all_files = os.listdir(args.dir)
    json_files = {f for f in all_files if f.endswith("_detailed.json")}
    log_files = {f for f in all_files if f.endswith(".log")}
    
    processed_runs = set()

    for filename in sorted(json_files):
        filepath = os.path.join(args.dir, filename)
        parsed_data = parse_json_file(filepath)
        if parsed_data:
            all_results.append(parsed_data)
        run_name = filename.replace("_detailed.json", "")
        processed_runs.add(run_name)

    for filename in sorted(log_files):
        run_name = filename.replace(".log", "")
        if run_name not in processed_runs:
            filepath = os.path.join(args.dir, filename)
            parsed_data = parse_json_file(filepath) 
            if parsed_data:
                all_results.append(parsed_data)

    if not all_results:
        print("No valid result files found.")
        return
        
    df = pd.DataFrame(all_results)
    
    if args.ignore_failed:
        df = df[df['status'] == 'Success'].copy()
        if df.empty:
            print("No successful runs found to display.")
            return

    # --- Sorting Logic ---
    df['bs_global_sort'] = pd.to_numeric(df['bs_global'], errors='coerce').fillna(0)
    df['target_sort'] = pd.to_numeric(df['target'], errors='coerce').fillna(0)
    
    df_sorted = df.sort_values(
        by=['attention', 'target_sort', 'bs_global_sort'],
        ascending=[True, True, True],
        na_position='first'
    )

    # --- Formatting for Display ---
    display_columns = {
        'attention': 'Attention', 'bs_global': 'BS', 'prefill': 'Prefill', 'target': 'Target',
        'pages': 'Pages', 'tpp': 'TPP', 'throughput': 'Thrpt (tok/s)', 
        'decode_throughput': 'Decode Thrpt', 'p50_step_ms': 'p50 Step (ms)',
        'p99_ttft_ms': 'p99 TTFT (ms)', 'status': 'Status'
    }

    for col in ['throughput', 'decode_throughput', 'p50_step_ms']:
         df_sorted[col] = df_sorted[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    df_sorted['p99_ttft_ms'] = df_sorted['p99_ttft_ms'].apply(lambda x: f"{x:,.1f}" if pd.notna(x) else "N/A")
    
    for col in ['bs_global', 'prefill', 'target']:
        df_sorted[col] = df_sorted[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else '-')

    df_display = df_sorted[list(display_columns.keys())].rename(columns=display_columns)

    print("\n" + "="*140)
    print(" " * 50 + "SUMMARY OF BENCHMARK RUNS")
    print("="*140)
    print(df_display.to_string(index=False))
    print("="*140)

if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas is not installed. Please run 'pip install pandas'")
        exit(1)
    main()