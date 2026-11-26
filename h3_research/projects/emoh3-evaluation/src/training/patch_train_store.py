import sys

p = "/gluster_hdd_cv/user/jinzili/tmp/h3ft/train.py"
s = open(p).read()
old = """    if distributed and args.strategy == "ddp":
        torch.distributed.init_process_group("nccl")
    elif distributed and args.strategy == "deepspeed":
        import deepspeed
        deepspeed.init_distributed("nccl")
"""
new = """    if distributed and args.strategy == "ddp":
        torch.distributed.init_process_group("nccl")
    elif distributed and args.strategy == "deepspeed":
        import deepspeed
        # This host has IPv6 compiled out, so torch 2.11's TCPStore (both
        # backends) cannot connect: it resolves to ::ffff:<ip> and creating an
        # AF_INET6 socket fails with EAFNOSUPPORT. Use a FileStore instead --
        # it needs no TCP socket at all.
        store_file = os.environ.get(
            "H3_STORE_FILE",
            str(args.output / "dist_store"),
        )
        os.makedirs(os.path.dirname(store_file), exist_ok=True)
        torch.distributed.init_process_group(
            "nccl",
            init_method="file://" + store_file,
            rank=int(os.environ.get("RANK", "0")),
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
        )
"""
assert old in s, "pattern not found"
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("patched file-store init OK")
