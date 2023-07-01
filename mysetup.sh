first_line_res=$(ip route show | head -n 1)
sudo ip route change ${first_line_res} rto_min 5ms
sudo ethtool -K ens9 tx-nocache-copy on
pip3 install crcmod
gsutil cp gs://libtpu_internal/rwitten/viperlite/2023-07-01-20:35:06-libtpu.so /home/rwitten//2023-07-01-20:35:06-libtpu.so
