#!/usr/bin/env bash
# Loopback RDMA tests on one host (steve: CX-7 port 0 <-> port 1 cable).
# Each case: server + client on this host (OOB over 127.0.0.1), RDMA WRITE,
# 64 KiB messages, 3 s. Prints BW and the wire counters of both ports, to
# tell cable traffic from NIC-internal loopback.
set -uo pipefail
B=$(dirname "$0")/install/bin
GID=3; SZ=65536; DUR=3; PORT=18515
cnt() { cat /sys/class/infiniband/$1/ports/1/counters/port_xmit_data; }   # 4-byte words
run() {  # name srv_dev srv_mem client_dev client_mem
  local name=$1 sd=$2 sm=$3 cd=$4 cm=$5
  local x0=$(cnt mlx5_0) x1=$(cnt mlx5_1)
  timeout 30 $B/ib_write_bw -d $sd -x $GID -s $SZ -t 32 -D $DUR -p $PORT $sm > /tmp/lb_srv.txt 2>&1 &
  sleep 1
  local out; out=$(timeout 30 $B/ib_write_bw -d $cd -x $GID -s $SZ -t 32 -D $DUR -p $PORT $cm 127.0.0.1 2>&1)
  wait
  local d0=$(( ($(cnt mlx5_0) - x0) * 4 / 1000000 )) d1=$(( ($(cnt mlx5_1) - x1) * 4 / 1000000 ))
  local bw; bw=$(echo "$out" | awk '$1=='$SZ' {print $4" MiB/s, "$5" Mpps"}')
  echo "$name | client $cd ${cm:-host} -> server $sd ${sm:-host} | ${bw:-FAILED} | wire tx MB: mlx5_0=$d0 mlx5_1=$d1"
  [ -z "$bw" ] && echo "$out" | tail -5
}
GPU="--use_cuda=0 --use_cuda_dmabuf"
run "1 same-port, host mem" mlx5_0 "" mlx5_0 ""
run "2 same-port, GPU mem"  mlx5_0 "$GPU" mlx5_0 "$GPU"
run "3 port0->port1, host"  mlx5_1 "" mlx5_0 ""
run "4 port0->port1, GPU->host" mlx5_1 "" mlx5_0 "$GPU"
run "5 port0->port1, GPU->GPU"  mlx5_1 "$GPU" mlx5_0 "$GPU"
