echo "hello"
curl -s 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/tpu-env' -H 'Metadata-Flavor: Google' > /tmp/tpu-env2 # store the metadata
X_MEGASCALE_COORDINATOR_ADDRESS=$(grep '^MEGASCALE_COORDINATOR_ADDRESS' /tmp/tpu-env2 | cut -d "'" -f 2)
echo ${X_MEGASCALE_COORDINATOR_ADDRESS}
export X_MEGASCALE_COORDINATOR_ADDRESS

y=3
export y
bash MaxText/echo_var.sh