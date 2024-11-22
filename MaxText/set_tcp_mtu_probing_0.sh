echo "Running set_tcp_mtu_probing_0.sh"
echo "Current value of tcp_mtu_probing: "
cat /proc/sys/net/ipv4/tcp_mtu_probing

echo 0 > /proc/sys/net/ipv4/tcp_mtu_probing

echo "New value of tcp_mtu_probing: "
cat /proc/sys/net/ipv4/tcp_mtu_probing