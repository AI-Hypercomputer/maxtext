"""
Script to read metrics for multiple run metric outputs. For each run output, the script will
calculate the medians of all metrics and output the results to a csv file.

Arguments:
  --RUN_NAMES_FILE: txt file containing run names separated by newline, required (e.g. "run_names.txt")
  --BASE_OUTPUT_DIRECTORY: GCS bucket that contains metrics file output, default is "gs://maxtext-experiments-multipod/"

Example usage:
  python3 read_metrics.py --RUN_NAMES_FILE="run_names.txt"
"""
import argparse
import csv
import json
import os
from statistics import median
import shutil
import glob
import subprocess

# Read in arguments
parser = argparse.ArgumentParser(description='Metrics parsing options')
parser.add_argument('--RUN_NAMES_FILE', type=str, default=None,
                    help='txt file containing run names separated by newline')
parser.add_argument('--BASE_OUTPUT_DIRECTORY', type=str, default='gs://maxtext-experiments-multipod/',
                    help='GCS bucket that contains metrics file output')

args = parser.parse_args()

def main() -> None:
  base_output_dir = args.BASE_OUTPUT_DIRECTORY

  # Check RUN_NAMES_FILE is not empty
  if args.RUN_NAMES_FILE is None or args.RUN_NAMES_FILE == '':
    print('--RUN_NAMES is required.')
    return -1
  
  # Read in run_names from txt file
  with open(args.RUN_NAMES_FILE, 'r') as f:
    run_names = f.read()
    if run_names is None or run_names == '':
      print(f'Error: {args.RUN_NAMES_FILE} is empty.')
      return -1
    run_names = run_names.split('\n')

  final_medians = []
  metrics = ['learning/grad_norm', 'learning/loss', 'learning/param_norm', 'perf/step_time_seconds', 'perf/per_device_tflops', 'perf/per_device_tflops_per_sec', 'learning/current_learning_rate']

  for name in run_names:
    # Generate metrics output directory name to download
    metrics_output_dir = os.path.join(base_output_dir, name, 'metrics')
    print('Metrics output directory:', metrics_output_dir)

    # Download metrics directory from GCS
    print(f'Downloading metrics directory for {name} from GCS...')
    command = ["gsutil", "cp", "-r", metrics_output_dir, os.path.join('.', name, 'metrics')]
    captured_output = subprocess.run(command, capture_output=True)
    if captured_output.returncode != 0:
      print('Error downloading metrics directory from GCS:\n' + captured_output.stderr.decode())
      return -1
    print('Download successful!')

    # Generate metrics file name to open
    metrics_output_file_name = os.path.join(name, 'metrics', 'metrics_step_*_to_step_*.txt')

    # Get metrics file and check it exists
    files = glob.glob(metrics_output_file_name)
    if len(files) == 0:
      print('No metrics file found.')
      # Delete downloaded metrics directory 
      shutil.rmtree(os.path.join('.', name))
      return -1

    # Open metrics file
    print('Reading metrics output file:', files[0])
    with open(files[0], 'r') as f: 
      # Read in data line by line
      lines = f.readlines()
      data = []
      for line in lines: 
        line_dict = json.loads(line)
        data.append(line_dict)

    # Get median of metrics
    result = {m: median([d[m] for d in data]) for m in metrics}
    result['run_name'] = data[0]['run_name']
    final_medians.append(result)

    # Delete downloaded metrics directory 
    shutil.rmtree(os.path.join('.', name))

    f.close()

  # Write final median results to csv
  metrics.insert(0, 'run_name')
  with open('metrics_median.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = metrics)
    writer.writeheader()
    writer.writerows(final_medians)

  print('Finished processing data. Check out metrics_median.csv' )

if __name__ == '__main__':
  main()