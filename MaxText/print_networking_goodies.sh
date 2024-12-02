echo "Printing sysctl net.ipv4.tcp_rto_min_us..."
echo "########################################"
sysctl net.ipv4.tcp_rto_min_us

# echo "Printing ss -tmoi"
# echo "########################################"
# ss -tmoi

echo "Printing ip route show"
echo "########################################"
ip route show

echo "Printing ifconfig"
echo "########################################"
ifconfig

echo "Printing cat /proc/sys/net/ipv4/tcp_mtu_probing"
echo "########################################"
cat /proc/sys/net/ipv4/tcp_mtu_probing

echo "Printing nstat"
echo "########################################"
nstat_out=$(nstat)
echo "nstat: ${nstat_out}"

echo "Printing qdisc"
echo "########################################"
qdisc_out=$(tc -s -d qdisc sh dev eth0)
echo "qdisc: ${qdisc_out}"