import os

import torch
import inc_collectives

import torch.distributed as dist

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

dist.init_process_group("cpu:inc_backend", rank=0, world_size=1)

# this goes through inc backend
x = torch.ones(6)
dist.all_reduce(x)
print(f"inc allreduce: {x}")