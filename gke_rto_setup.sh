echo "Running gke_rto_setup.sh..."

# Stop execution if any command exits with error
set -e

echo "Adjust RTO and apply non cache copy"
first_line_res=$(ip route show | head -n 1)
if [[ "$(echo "$first_line_res" | grep "rto_min lock 5ms" | wc -l)" -eq 0 ]]; then
  ip route change ${first_line_res} rto_min 5ms
fi
dev_name=$(echo "$first_line_res" | awk -F'[[:space:]]' '{ print $5 }')
echo "dev_name=${dev_name}"
ethtool -K "${dev_name}" tx-nocache-copy on

echo "gke_rto_setup.sh finished"