#!/bin/bash

# 64 x 64
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 1 -n 10 64 64 64
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 2 -n 10 64 64 64
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 4 -n 10 64 64 64
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 8 -n 10 64 64 64
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 16 -n 10 64 64 64
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 32 -n 10 64 64 64

# 128 x 128
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 1 -n 10 128 128 128
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 2 -n 10 128 128 128
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 4 -n 10 128 128 128
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 8 -n 10 128 128 128
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 16 -n 10 128 128 128
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 32 -n 10 128 128 128

# 256 x 256
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 1 -n 10 256 256 256
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 2 -n 10 256 256 256
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 4 -n 10 256 256 256
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 8 -n 10 256 256 256
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 16 -n 10 256 256 256
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 32 -n 10 256 256 256

# 512 x 512
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 1 -n 10 512 512 512
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 2 -n 10 512 512 512
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 4 -n 10 512 512 512
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 8 -n 10 512 512 512
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 16 -n 10 512 512 512
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 32 -n 10 512 512 512

# 1024 x 1024
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 1 -n 10 1024 1024 1024
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 2 -n 10 1024 1024 1024
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 4 -n 10 1024 1024 1024
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 8 -n 10 1024 1024 1024
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 16 -n 10 1024 1024 1024
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 32 -n 10 1024 1024 1024

# 2048 x 2048
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 1 -n 10 2048 2048 2048
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 2 -n 10 2048 2048 2048
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 4 -n 10 2048 2048 2048
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 8 -n 10 2048 2048 2048
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 16 -n 10 2048 2048 2048
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 32 -n 10 2048 2048 2048

# 4096 x 4096
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 1 -n 10 4096 4096 4096
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 2 -n 10 4096 4096 4096
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 4 -n 10 4096 4096 4096
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 8 -n 10 4096 4096 4096
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 16 -n 10 4096 4096 4096
srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 32 -n 10 4096 4096 4096
