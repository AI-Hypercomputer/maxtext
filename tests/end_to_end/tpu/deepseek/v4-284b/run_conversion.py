import subprocess
import sys
import os

def main():
  env = os.environ.copy()
  env["PYTHONPATH"] = os.getcwd() + "/src"
  env["DECOUPLE_GCLOUD"] = "TRUE"


  
  cmd = [
      "/home/snehalv_google_com/maxtext/maxtext_venv/bin/python",
      "src/maxtext/checkpoint_conversion/to_maxtext.py",
      "src/maxtext/configs/base.yml",
      "model_name=deepseek4-tiny",
      "base_output_directory=gs://snehalv-data/deepseek4-conversion-pr/scanned/",
      "scan_layers=true",
      "skip_jax_distributed_system=true",
      "--hf_model_path=tests/end_to_end/tpu/deepseek/v4-284b/hf_tiny_model",
  ]
  
  print("Starting conversion via subprocess...")
  with open("conversion.log", "w") as f:
    result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    
  print(f"Subprocess finished with exit code: {result.returncode}")
  sys.exit(result.returncode)

if __name__ == "__main__":
  main()
