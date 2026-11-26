#!/bin/bash
# Test TCPStore with real IPv4 host vs 127.0.0.1
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "host IP: $IP"
for HOST in "$IP" "127.0.0.1"; do
  echo "=== TCPStore host=$HOST ==="
  HOST_TEST=$HOST timeout 25 /gluster_hdd_cv/user/jinzili/env/h3ft/bin/python - <<'PY' 2>&1 | tail -2
import os, threading
from torch.distributed import TCPStore
h = os.environ.get("HOST_TEST", "127.0.0.1")
s = TCPStore(h, 29998, world_size=2, is_master=True)
ok = [False]
def client():
    try:
        c = TCPStore(h, 29998, world_size=2, is_master=False)
        c.set("k", "v"); ok[0] = True
    except Exception as e:
        print("client err:", type(e).__name__, str(e)[:120])
t = threading.Thread(target=client); t.start()
try:
    v = s.get("k"); print("SERVER_GOT", v)
except Exception as e:
    print("server err:", type(e).__name__, str(e)[:120])
t.join()
print("RESULT", "OK" if ok[0] else "FAIL", "host=", h)
PY
done
