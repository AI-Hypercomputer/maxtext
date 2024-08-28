from google.cloud import storage
import csv
import uuid
import max_logging

def upload_csv(bucket_name, object_name, data):
  tmp_file_name = f"{str(uuid.uuid4())}.csv"
  with open(tmp_file_name, 'w', encoding="utf-8", newline='') as file:
    csv_writer = csv.writer(file)
    for _, t in enumerate(data):
      csv_writer.writerow(t)
  
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(object_name)
  blob.upload_from_filename(tmp_file_name)
  # Set if_generation_match to 0 for the object insertion to be idempotent
  # such that server-side errors will be auto retried. See
  # https://cloud.google.com/storage/docs/retry-strategy#idempotency-operations and
  # https://cloud.google.com/storage/docs/retry-strategy#tools.
  generation_match_precondition = 0
  try:
    blob.upload_from_filename(tmp_file_name, if_generation_match=generation_match_precondition)
  except Exception as e:
    max_logging.log(f"An error occurred during upload {object_name} to {bucket_name}: {e}")
