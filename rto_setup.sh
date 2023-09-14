echo "Running rto_setup.sh..."

# Stop execution if any command exits with error
set -e

first_line_res=$(ip route show | head -n 1)
sudo ip route change ${first_line_res} rto_min 5ms
sudo ethtool -K ens9 tx-nocache-copy on
echo "rto_setup finished"