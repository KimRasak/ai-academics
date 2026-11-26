#!/bin/bash
echo "=== ipv6 disable flag ==="
cat /proc/sys/net/ipv6/conf/all/disable_ipv6 2>/dev/null
echo "=== AF_INET6 socket test ==="
/gluster_hdd_cv/user/jinzili/env/h3ft/bin/python -c "import socket; s=socket.socket(socket.AF_INET6, socket.SOCK_STREAM); print('AF_INET6_OK')" 2>&1
echo "=== getaddrinfo 127.0.0.1 ==="
/gluster_hdd_cv/user/jinzili/env/h3ft/bin/python -c "import socket; [print(a) for a in socket.getaddrinfo('127.0.0.1', 29500)]" 2>&1
echo "=== launcher list ==="
curl -s http://osa00-h100-034.osa00.bdjp.io:8659/list
echo
