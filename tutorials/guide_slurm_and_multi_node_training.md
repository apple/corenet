# Guide: Slurm and multi-node training

This guide assumes you have a prior understanding of how `corenet-train` works. If you
are new to the CoreNet, please consider going through [this tutorial](./train_a_new_model_on_a_new_dataset_from_scratch.ipynb) first.

CoreNet supports multi-node distributed training using DDP and FSDP. In the following 
command, we assume a multi-node-multi-gpu or multi-node-single-gpu cluster.

```bash
# Assuming 4-node x 8-GPU Cluster:
export CFG_FILE="path/to/config.yaml"         # Please change this to your desired config file.
export GPU_PER_NODE="8"                       # Please change this to the number of GPUs that are available on each node.
export NUM_NODES="4"                          # Please change this to the number of nodes (i.e. hosts/machines) in your cluster.
export NODE_RANK="0"                          # Please change this to a number in range [0, $NUM_NODES-1]. Each node must use a unique rank.
export MAIN_IP_PORT="tcp://IP_OF_RANK0:PORT"  # Please change "IP_OF_RANK0" to the host name with rank 0 (should be accessible from other nodes) and change "PORT" to a free port number.

export WORLD_SIZE=$((NUM_NODES * GPU_PER_NODE))
export GPU_RANK=$((NODE_RANK * GPU_PER_NODE))

corenet-train --common.config-file $CFG_FILE --common.results-loc results --ddp.rank $GPU_RANK --ddp.world-size $WORLD_SIZE --ddp.dist-url $MAIN_IP_PORT --ddp.backend nccl
```

> **_NOTE:_** As reflected in the above `WORLD_SIZE` and `GPU_RANK` calculations, the
> `--ddp.rank` and `--ddp.world-size` arguments expect gpu indices,
> rather than node indices. However, you should run `corenet-train` once per node.

### Running on a Slurm cluster

Please customize the following template for slurm clusters. Also, **please note that we
have not tested CoreNet on Slurm**. The 
following script assumes a 4-node x 8-GPU cluster.

```bash
#!/bin/bash
#SBATCH --job-name=your-job-name
#SBATCH --nodes=4                # Please change this to the number of nodes (i.e. hosts/machines) in your cluster.
#SBATCH --ntasks-per-node=1      # No need to change this line, even for multi-gpu nodes.
#SBATCH --gpus-per-node=gpu:8    # Please change this to the number of GPUs that are available on each node.

export CFG_FILE="path/to/config.yaml"         # Please change this to your desired config file.

export GPU_PER_NODE="$SLURM_GPUS_PER_NODE"                       
export NUM_NODES="$SLURM_NNODES"
export NODE_RANK="$SLURM_PROCID"
# Inspired by https://discuss.pytorch.org/t/distributed-training-on-slurm-cluster/150417/7
export MAIN_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MAIN_IP="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MAIN_IP_PORT="tcp://$MAIN_IP:$MAIN_PORT"
export WORLD_SIZE=$((NUM_NODES * GPU_PER_NODE))
export GPU_RANK=$((NODE_RANK * GPU_PER_NODE))

corenet-train --common.config-file $CFG_FILE --common.results-loc results --ddp.rank $GPU_RANK --ddp.world-size $WORLD_SIZE --ddp.dist-url $MAIN_IP_PORT --ddp.backend nccl
```
