import time
import os
from absl import app
from typing import Sequence
from ray.job_submission import JobSubmissionClient, JobStatus

def main(argv: Sequence[str]) -> None:
    client = JobSubmissionClient("http://127.0.0.1:8265")
    print("Connected to head!", flush=True)
    
    maxtext_cmd_args = " ".join(argv[1:])
    job_id = client.submit_job(
        entrypoint=f"RAY_DEDUP_LOGS=0 python3 src/MaxText/resilient_train.py {maxtext_cmd_args}",
        runtime_env={"working_dir" : "./",
                     "excludes" : ["src/MaxText/test_assets", ".git"]}
    )
    
    print(f"Launched job: {job_id}", flush=True)
    prev_logs = ''
    while True:
        status = client.get_job_status(job_id)
        if status in {JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED}:
            if status in {JobStatus.STOPPED, JobStatus.FAILED}:
                logs = client.get_job_logs(job_id)
                print(logs, flush=True)
            break
        time.sleep(5)
        if status == JobStatus.RUNNING:
            logs = client.get_job_logs(job_id)
            print(logs[len(prev_logs):], flush=True)
            prev_logs = logs
            

    

if __name__ == "__main__":
    app.run(main)
