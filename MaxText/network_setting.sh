# Install ip.
apt-get update
yes | apt-get install net-tools
yes | apt-get install iproute2
yes | apt-get install procps
yes | apt-get install ethtool

# Disable slow start after idle
sysctl net.ipv4.tcp_slow_start_after_idle=0

# Disable metrics cache
sysctl net.ipv4.tcp_no_metrics_save=1

# RTO 5ms (new way)

# Address rto_min issue with two default routing entries: screen/7RGgkiXkGXSeYF2
route=$(ip route show | sed -n 1p)
second_route=$(ip route show | sed -n 2p)
if [[ "${second_route}" =~ ^default.* ]]; then
  modified_route=${route//" lock"/}
  ip route delete ${modified_route}
fi
route=$(ip route show | sed -n 1p)
echo "route rto before change: $route"
if [[ "${route}" =~ .*lock.*5ms.* ]]; then
  echo "${route}"
else
  # shellcheck disable=SC2086
  ip route change $route rto_min 5ms
fi
route=$(ip route show | sed -n 1p)
echo "route rto after change: $route"

# Disable Cubic Hystart Ack-Train
echo 2 > /sys/module/tcp_cubic/parameters/hystart_detect

# Improve handling SYN burst
echo 4096 > /proc/sys/net/core/somaxconn
echo 4096 > /proc/sys/net/ipv4/tcp_max_syn_backlog

# TODO(b/309861674): Comment this out until the error is fixed.
# # Increase TCP Buffer limits
# echo "4096 540000 15728640" > /proc/sys/net/ipv4/tcp_rmem
# echo "4096 262144 67108864" > /proc/sys/net/ipv4/tcp_wmem

# Disable MTU Discovery
echo 0 > /proc/sys/net/ipv4/tcp_mtu_probing

# Increase TCP Zerocopy control memory
sysctl -w net.core.optmem_max=131072


yes | apt-get install sysstat

# Printing output of `ip route show`
echo -e "\nPrinting output of 'ip route show':"
ip route show