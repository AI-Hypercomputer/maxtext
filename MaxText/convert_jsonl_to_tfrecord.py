import json
import tensorflow as tf
from google.cloud import storage

def jsonl_to_tfrecord(gcs_jsonl_path, gcs_tfrecord_file):
    client = storage.Client()
    input_bucket_name, input_file_path = gcs_jsonl_path.replace('gs://', '').split('/', 1)
    input_bucket = client.get_bucket(input_bucket_name)
    input_blob = input_bucket.blob(input_file_path)
    

    with tf.io.TFRecordWriter(gcs_tfrecord_file) as writer:
        for line in input_blob.download_as_string().decode('utf-8').splitlines():
            try:
                data = json.loads(line)
                # Create features dictionary (adjust based on your data)
                features = {
                                'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['text'].encode('utf-8')]))
                            }

                # Create an Example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=features))

                # Write the serialized example to the TFRecord file
                writer.write(example.SerializeToString())
                print("Success")

            except:
                print(f"Error in parsing {line}")


def jsonl_to_tfrecord_all_files(gcs_jsonl_path, prefix, gcs_tfrecord_path):
    client = storage.Client()
    input_bucket_name = gcs_jsonl_path.replace('gs://', '')
    input_bucket = client.get_bucket(input_bucket_name)

    input_blobs = input_bucket.list_blobs(prefix=prefix)

    for input_blob in input_blobs:
        print(input_blob.name)
    
        gcs_tfrecord_file = gcs_tfrecord_path+input_blob.name.split('/')[-1].replace('jsonl','tfrecords')
        output_blob = input_bucket.blob(gcs_tfrecord_file.replace("gs://"+input_bucket_name+"/",""))

        if not output_blob.exists():

            with tf.io.TFRecordWriter(gcs_tfrecord_file) as writer:
                for line in input_blob.download_as_string().decode('utf-8').splitlines():
                    try:
                        data = json.loads(line)
                        # Create features dictionary (adjust based on your data)
                        features = {
                                        'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['text'].encode('utf-8')]))
                                    }

                        # Create an Example protocol buffer
                        example = tf.train.Example(features=tf.train.Features(feature=features))

                        # Write the serialized example to the TFRecord file
                        writer.write(example.SerializeToString())
                        print("Success")

                    except:
                        print(f"Error in parsing {line}")
        # else:
        #     print(f"Skipping {output_blob} because it already exists")

# Usage (assuming GCS URI format)

gcs_jsonl_bucket = 'mazumdera-test-bucket'
prefix = 'path/jsonl-data'
tfrecord_path = 'gs://'+gcs_jsonl_bucket+'/path/tfrecord-data/' 

jsonl_to_tfrecord_all_files(gcs_jsonl_bucket, prefix, tfrecord_path) 

