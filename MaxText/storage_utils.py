from google.cloud import storage
import csv
import uuid

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