---
title: 'Distributed training technologies for Transformers: Overview'
description: A brief summary of different distributed training strategies used to train LLMs.
date: 2024-08-30 00:55:00 +0530
author: skrohit
categories: [distributed-training]
tags: [llms, tensor parallelism, pipeline parallelism, ZeRO, fsdp, transformers, deep-learning]
pin: true
---

## Why do distributed training?
Current transformers-based large language models (LLMs) have tens or even hundreds of billions of parameters. Training such models on a single machine is often infeasible due to memory and computational limitations. For instance, the Nvidia A100 GPU, one of the most advanced options available, offers 80 GB of high-bandwidth memory, while the Llama3 70B model, one of the top open-source LLMs, requires around 140 GB just to store its parameters.

These models are also trained on vast datasets containing trillions of tokens, necessitating a significant amount of compute power, which can lead to prohibitively long training times. By distributing the training of such models across multiple GPUs, the workload can be parallelized, significantly reducing training time and making the training process feasible. There are multiple strategies to perform such parallel training each with its own advantages and disadvantages. Let us understand them one by one.

## Different distributed training strategies
- Distributed Data Parallel
- ZeRO Optimizer
- Model/Tensor Parallelism
- Pipeline Parallelism
- Pytorch's Fully Sharded Data Parallel

### Distributed Data Parallel
Distributed Data Parallel is the most common strategy for parallel training. In DDP, the model is replicated across multiple GPUs (in seperate processes), with each GPU handling a different mini-batch of data. The gradients are calculated independently on each device. And as soon as gradients are ready it triggers the hook on model parameters to synchronize gradients using the `AllReduce`. This approach efficiently utilizes multiple GPUs and reduces idle time by overlapping gradient synchronization with backward calculation. After gradient's synchronization optimizer updates the model weight's locally. This [design note](https://pytorch.org/docs/master/notes/ddp.html) on DDP is very helpful in understanding it better.

### ZeRO Optimizer
The ZeRO (Zero Redundancy Optimizer) is a specialized optimizer designed to address memory bottlenecks in large-scale model training. Before going into the details of ZeRO lets us get an overview of the [Adam Optimizer](https://arxiv.org/abs/1412.6980). During training Adam maintains moving average of momentum and variance of each trainable parameter (in FP32 precision) and during mixed precision training, additionally, a copy of parameters are kept in FP32 precisions. These are called optimizer states. DDP replicates such optimizer states along with model parameters across GPUs causing memory redundancy. ZeRO removes such redundancies by partitioning the optimizer states, gradients, and model parameters across multiple devices, enabling the training of models with more parameters. Depending on the kind of parameter partition ZeRO is divided into three stages, each stage offering a progressively more aggressive memory reduction strategy.

In the first stage only optimizer states and in second stage both optimizer states and parameter graidents are partitioned across GPUs respectively. It is always preferred to use ZeRO from second stage of ZeRO. Because when we partition optimizer states, model parameters and optimizer states should be updated only on their respective partitions. For these updates each partition needs the reduced gradients for the corresponding parameters. And gradients are reduced only on the data parallel process (GPUs) responsible for updating the corresponsing parameters. After the reduction, gradients are not required so keeping them in memory in each partitions is waste of HBM. Instead partitioning gradients corresponding to the optimizer states and only keeping those in memory makes sense. At the end all-gather is performed to collect all the updated parameters from all the GPUs. During ZeRO second stage the total communication volume is same as DDP.

> Note: Forward and backward operations doesn't change between DDP and ZeRO in the sense that each GPU performs its own forward and backward in parallel. Main difference lies in the way tensors are stored and communicated to perform parameter updates.
{: .prompt-tip }

In the third stage model parameters are also partitioned among GPUs. Each partition only keeps parameters that it updates. Therefore, during forward propagation it needs to receive the paramters for all other partitions. However, this can be pipelined to avoid memory overhead. Before computing the forward propagation on the part of the model corresponding to a particular partition, the data parallel process responsible for that partition can broadcast the weights to all the data parallel processes. Once the forward propagation for that partition is done, the parameters can be discarded. In other words, we reschedule the parameter all-gather by spreading it across the entire forward propagation, and discarding the parameters once they have been used.  When parameters are required for forward and backward propagation, they are received from appropriate data parallel process (GPU) through broadcast.

### Model/Tensor Parallelism
Model parallelism was first introduced in the Megatron paper by Nvidia. Unlike data parallelism, where the model is replicated across devices, model parallelism splits the model itself across multiple GPUs. This is particularly useful when the model is too large to fit into the memory of a single device. In model parallelism, different layers (or even parts of layers) of the model are assigned to different GPUs, which compute their respective parts of the forward and backward passes. This approach is essential for training extremely large models, such as those with hundreds of billions of parameters.

### Pipeline Parallelism
Pipeline parallelism is a strategy that performs parallelism by dissecting models horizontally across the layers. In this approach, the model is divided into stages, with each stage assigned to a different device. As data passes through the pipeline, each device processes its stage of the model before passing the output to the next device. This allows multiple micro-batches to be processed concurrently, improving overall throughput. Pipeline parallelism can be combined with model and data parallelism for even greater scalability, though it requires careful management of inter-stage communication to minimize idle times and maximize efficiency.

### PyTorch's FSDP
PyTorch's Fully Sharded Data Parallel (FSDP) is a newer distributed training strategy introduced in the PyTorch framework. FSDP shards both the model parameters and optimizer states across all participating devices, reducing memory usage significantly. This method allows training models that would otherwise be too large to fit in memory, even with advanced GPUs. FSDP also supports mixed precision training and can be used in conjunction with other parallelism strategies, making it a versatile option for large-scale distributed training.