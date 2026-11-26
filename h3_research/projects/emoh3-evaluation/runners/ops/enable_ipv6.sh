#!/bin/bash
echo "=== current disable_ipv6 values ==="
cat /proc/sys/net/ipv6/conf/all/disable_ipv6 2>/dev/null || echo "no /proc/sys/net/ipv6/conf/all/disable_ipv6"
cat /proc/sys/net/ipv6/conf/lo/disable_ipv6 2>/dev/null || echo "no lo file"
echo "=== try enable ==="
sysctl -w net.ipv6.conf.all.disable_ipv6=0 2>&1
sysctl -w net.ipv6.conf.lo.disable_ipv6=0 2>&1
sysctl -w net.ipv6.conf.default.disable_ipv6=0 2>&1
echo "=== after ==="
cat /proc/sys/net/ipv6/conf/lo/disable_ipv6 2>/dev/null || echo "no lo file"
echo "=== AF_INET6 test ==="
/gluster_hdd_cv/user/jinzili/env/h3ft/bin/python -c "import socket; s=socket.socket(socket.AF_INET6, socket.SOCK_STREAM); print('AF_INET6_OK')" 2>&1
