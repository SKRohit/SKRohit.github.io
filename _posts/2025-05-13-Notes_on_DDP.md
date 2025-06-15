---
title: Notes on PyTorch's Distributed Data Parallel (DDP)
description: A brief summary of PyTorch's implementation of Distributed Data Parallel DDP.
date: 2025-05-13 00:55:00 +0530
author: skrohit
categories: [distributed-training, DDP]
tags: [llms, transformers, deep-learning, distributed data parallel, ddp]
pin: true
---
# Notes on PyTorch Distributed Data Parallel (DDP) 🚀

## 📊 What is DDP
In distributed data parallel, DDP, technique of distributed training, the neural network model is trained across multiple GPUs (across same or different nodes), with each GPU handling a different mini-batch of data. Each GPU run forward operation on a different mini-batch and computes gradients for its own mini-batch. To keep model replicas in sync, gradients are averaged (synchronized) across all processes before the optimizer step. This is done using the `all-reduce` operation. 

🎯 PyTorch's DDP implementation aims to acheive three goals:
- **Mathematical Equivalence**: DDP training should give same result as if all training have been performed locally without model replication.
- **Non Intrusive and interceptive API**: Transition to DDP must be non-intrusive in application code.
- **High Performance**: Overlap communication with computation so that more compute resources convert to higher training throughput.

A sample script to perform DDP training with PyTorch is shown below:

<details>
<summary>💡 DDP Training Code</summary>

```
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

def setup():
    # Reads environment variables set by torchrun
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    # Initialize the process group with NCCL backend
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size,
    )
    # Pin this process to the GPU with id local_rank
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )
    def forward(self, x):
        return self.net(x)

def main():
    rank, world_size, local_rank = setup()
    print(f"[Rank {rank}/{world_size}] using GPU {local_rank}")

    # Create model and move it to the appropriate GPU
    model = SimpleModel().cuda(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])  # gradient sync on WORLD group
    # Dummy dataset
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)

    # Shard the dataset
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, sampler=sampler)

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(5):
        sampler.set_epoch(epoch)
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.cuda(local_rank, non_blocking=True)
            batch_y = batch_y.cuda(local_rank, non_blocking=True)

            optimizer.zero_grad()
            out = ddp_model(batch_x)
            loss = loss_fn(out, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"[Rank {rank}] Epoch {epoch} Loss {total_loss/len(loader):.4f}")

    cleanup()

if __name__ == "__main__":
    main()
```

Please run above code using the following command:

```
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12345 ddp_example.py
```
</details>
 
Let us try to understand the code above as well as the DDP training process in general.

## 🎯 Training Launcher
We will start with the training job launcher. Differnet launchers like [`torchrun`](https://docs.pytorch.org/docs/stable/elastic/run.html), [`mpirun`](https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man1/mpirun.1.html), etc. can be used to launch a distributed training job. Launchers are responsible for setting up the *rendezvous environment variables* (variables that all processes in a distributed job read to discover each other and form the communication group) and launching the training script across multiple processes or ranks. `torchrun`  is a python [console script](https://packaging.python.org/en/latest/specifications/entry-points/#use-for-scripts) to the main module torch.distributed.run. It supports **elastic training**, allowing the number of processes to scale dynamically during execution, and offers **fault tolerance**, enabling automatic recovery from process failures, please refer to the [documentation](https://docs.pytorch.org/docs/stable/distributed.elastic.html) for details.

## 🧠 torch.distributed.init_process_group
Once the training job is launched (and all processes are started), **`torch.distributed`** module and default **process group** needs to be initialized foremost. The **`torch.distributed`** module provides a set of APIs for distributed training, including communication primitives and process group management. The **process group** represents a group of processes on which collectives operate. By default collectives operate on the default process group (also called the world) and require all processes to enter the distributed function call. **`torch.distributed.init_process_group`** performs this task by accepting arguments like `backend` that specifies communication backend to be used, `init_method` that specifies a URL string which indicates where/how to discover peers, `rank` specifies the rank of the current process, and `world_size` specifies the total number of processes in the group etc. For complete list of arguments, please refer to the [documentation](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group).

## 🔧 DDP Details
After successfull distributed backend initilzation a pytorch model can be wrapped in **`torch.nn.parallel.DistributedDataParallel`** (DDP). DDP is an nn.Module class, with its own `forward` implementation that takes care of a few other things apart from forward execution of the model. For backward pass it relies on backward hooks to trigger gradient reduction which will be invoked by autograd engine when executing `backward()` on loss tensor.

Internally, DDP needs to take care of many details and optimizations, 
to achieve the three goals mentioned above. We will look at a few of them:
- **Registering Autograd Hooks**
- **Broadcasting Model Parameters and Buffers**
- **Gradient Bucketing**
- **Overlapping Communication (all-reduce) with backward computation**
- **Finding Unsed Parameters**
- **Gradient Accumulation**


### 🧩 Registering Autograd Hooks
DistributedDataParallel (DDP) registers autograd hooks to efficiently synchronize gradients across processes. Specifically, DDP locates the `AccumulateGrad` node for each model parameter in the autograd graph. The `AccumulateGrad` node is responsible for accumulating gradients for leaf tensors (parameters) during backpropagation.

DDP attaches a post-backward hook to each parameter's `AccumulateGrad` node. When the gradient for a parameter is computed and ready, this hook is triggered. The hook's main job is to signal DDP that the gradient for this parameter is available and can be included in a communication bucket for synchronization.

---
### 🔄 Broadcasting Model Parameters and Buffers
To acheive mathematical equivalence with DDP, it is important that we start with same parameter values, hence DDP constructor broadcasts model states (parameters and buffers) from the rank 0 process to all other processes in the group. Model Buffers are necessary when layers (e.g. `BatchNorm`) needs to keep track of states like the running variance and mean. DDP broadcasts model buffers from rank 0 to all other ranks before each forward pass.

---

### 📦 Gradient Bucketing
Collective communication performs poorly on small tensors due to low bandwidth utilization and large communication overhead. So multiple `all-reduce` operations on small tensors reduces the training throughput. Gradient bucketing reduces this communication overhead by grouping parameters into "buckets" and synchronizing their gradients together rather than individually. Instead of triggering an all-reduce operation for each parameter's gradient separately, once all gradients in a bucket are ready, DDP immediately launches an asynchronous all-reduce operation to average the gradients across all processes. By default, DDP uses a bucket size of 25MB, but this can be adjusted using the `bucket_cap_mb` parameter in the DDP constructor to optimize for specific hardware configurations and model size.

---

### 🏎️ Overlapping Communication (all-reduce) with backward computation
During backpropagation, gradients for the last layers become available before earlier layers. DDP uses this property of neural network models to overlap gradient synchronization with gradient computation in backward pass. DDP registered autograd hooks track when gradients become ready using a reference counting approach. Once all gradients in a bucket are computed and previous bucket has been launched, DDP triggers an asynchronous all-reduce operation for that bucket without waiting for the entire backward pass to finish. The order in which buckets are lauched must be same across all the ranks, because collective operations only validate the type and shape of the tensors. Hence, different order could result in incorrect reduction or program crash/hang. The bucket-to-bucket traversal follows the reverse order of `model.parameters()`. This aligns with the order in which gradients become available during backpropagation (last layers first), maximizing the opportunity for computation-communication overlap.

For very large models, this optimization significantly reduces training time by effectively "hiding" communication costs behind computation. The effectiveness depends on the ratio between computation time and communication bandwidth, with greater benefits for computation-heavy models.

---

### 🔍 Finding Unused Parameters
Sometimes model computation graphs could vary iteration to iteration, meaning that some gradients might be skipped in some iterations. This could create two issues:
- **⏱️ Backward pass hang**: Gradient-to-bucket mapping is determined at the construction time, hence absent gradients would leave some buckets never seeing the final autograd hook and failing to mark the bucket as ready. As a result, the backward pass could hang. To address this problem, DDP traverses the autograd graph from the output tensors of the forward pass to find all participating parameters. The readiness of those participating tensors is a sufficient signal to conclude the completion of the backward pass. Therefore, DDP can avoid waiting for the rest of the parameter gradients by proactively marking them ready at the end of the forward pass.
- **⚠️ Incorrect gradient calculation**: The absence of gradients for certain parameters on one rank is not sufficient to ensure the optimizer step is performed correctly. The same parameter might have participated in the computation graph on other ranks, and accurate gradient synchronization requires this information. For example, optimizer uses gradient absence information to skip updating momentum values. To address this, DDP maintains a bitmap to track which parameters participated locally and performs an additional AllReduce to gather information about globally unused parameters. This bitmap cannot be merged with other gradient AllReduce operations due to possible differences in element types. This extra overhead only occurs when the application explicitly instructs DDP to check for unused parameters, so the cost is incurred only when necessary.

`find_unused_parameters` flag in DDP constructor enables it handle this issue.

---

### 💾 Gradient Accumulation
In some distributed training scenarios, it is beneficial to accumulate gradients over multiple iterations before synchronizing them across processes. This approach is useful when the input batch is too large to fit into device memory, or when reducing the frequency of gradient synchronization can improve performance. Instead of launching an `all-reduce` operation in every iteration, the application can perform several local forward and backward passes on micro-batches, accumulating gradients, and only synchronize (all-reduce) at the boundary of a larger batch. 

PyTorch DDP provides the `no_sync` context manager to support this use case. When entering the `no_sync` context, DDP disables its gradient synchronization hooks, allowing gradients to accumulate locally. First backward pass outside the context will trigger synchronization of all accumulated gradients. Internally, `no_sync` simply toggles a flag that is checked in DDP's forward function. While in `no_sync` mode, DDP hooks are disabled, and information about globally unused parameters is also accumulated in bitmap. When synchronization resumes, all relevant gradients are reduced together, and the accumulated state is cleared.

---

## 🚀 DDP with torch.compile
DDP’s performance advantage comes from overlapping allreduce collectives with computations during backwards. AotAutograd prevents this overlap when used with TorchDynamo for compiling a whole forward and whole backward graph, because allreduce ops are launched by autograd hooks _after_ the whole optimized backwards computation finishes.

TorchDynamo’s DDPOptimizer helps by breaking the forward graph at the logical boundaries of DDP’s allreduce buckets during backwards. Note: the goal is to break the graph during backwards, and the simplest implementation is to break the forward graphs and then call AotAutograd and compilation on each section. This allows DDP’s allreduce hooks to fire in-between sections of backwards, and schedule communications to overlap with compute.

---

## References:
- [PyTorch DDP Paper](https://arxiv.org/pdf/2006.15704)
- [PyTorch DDP Internal Blog](https://yi-wang-2005.medium.com/pytorch-distributeddataparallel-internals-c01c30a41192)
- [PyTorch DDP Internal Design](https://docs.pytorch.org/docs/stable/notes/ddp.html)
- [torch.distributed PyTorch doc](https://docs.pytorch.org/docs/stable/distributed.html)
- [torch.nn.parallel.DistributedDataParallel PyTorch doc](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
---
*"The art of distributed training is not just dividing work, but orchestrating it."*
