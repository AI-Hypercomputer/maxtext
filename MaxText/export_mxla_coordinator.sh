curl -s 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/tpu-env' -H 'Metadata-Flavor: Google' > /tmp/tpu-env # store the metadata
X_MEGASCALE_COORDINATOR_ADDRESS=$(grep '^MEGASCALE_COORDINATOR_ADDRESS' /tmp/tpu-env | cut -d "'" -f 2)
echo $X_MEGASCALE_COORDINATOR_ADDRESS
export X_MEGASCALE_COORDINATOR_ADDRESS