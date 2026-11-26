#!/bin/bash
# Test TCPStore with USE_LIBUV=0 vs default on 034 (IPv6-disabled host)
for mode in 0 1; do
  echo "=== USE_LIBUV=$mode ==="
  USE_LIBUV=$mode timeout 40 /gluster_hdd_cv/user/jinzili/env/h3ft/bin/python - <<'PY' 2>&1 | tail -3
import os, threading, time
from torch.distributed import TCPStore
s = TCPStore("127.0.0.1", 29999, world_size=2, is_master=True)
def client():
    try:
        c = TCPStore("127.0.0.1", 29999, world_size=2, is_master=False)
        c.set("k", "v")
    except Exception as e:
        print("client err:", type(e).__name__, str(e)[:100])
t = threading.Thread(target=client)
t.start()
try:
    s.get("k"); print("TCPSTORE_OK mode", os.environ.get("USE_LIBUV"))
except Exception as e:
    print("server err:", type(e).__name__, str(e)[:100])
t.join()
PY
done
