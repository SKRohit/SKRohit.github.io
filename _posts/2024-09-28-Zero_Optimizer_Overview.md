---
title: 'Zero Redunduncy Optimizer (ZeRO): Paper Summary'
description: A brief note on ZeRO's workings.
date: 2024-09-28 00:55:00 +0530
author: skrohit
categories: [ZeRO Optimizer]
tags: [llms, ZeRO, transformers]
pin: false
---

> Note: Will not find anything new if you have already read [ZeRO paper](https://arxiv.org/pdf/1910.02054).
{: .prompt-tip }

## What is ZeRO?
ZeRO is a distributed optimizer designed to efficiently train large deep learning models by reducing memory redundancy across multiple devices.

### Other parallelism techniques:
- **Data Parallelism**: Does not reduce the memory consumption per device, as each device still holds a complete copy of the model.
- With pp, mp and cpu offloading etc, there is trade-off between functionality, usability, as well as memory and compute/communication efficiency.
- **Model Parallelism**: Splits the model vertically , partitioning the computation and parameters in each layer across devices, requiring signification communication between each layer. So they are usually applied within the node, where inter gpu communication bandwidth is high.
- **Pipeline Parallelism**: Divides the model into stages, but does not address memory redundancy.

### Memory consumption spectrum in LLMs:
- **Model States (ZeRO DP)**: Parameters, gradients, and optimizer states (e.g. momentum and variances in Adam). DP (Data Parallel) replicates the models across all devices, leading to redundant memory usage. While MP partitions these states, to obtain high memory efficiency,  but often result in too fine-grained computation and expensive communication that is less **scaling efficient**. Because all the model states are not required all the time, ZeRO DP partitions the model states across devices. It retains computational/communication efficiency, by retaining the computational granularity and communication volume of DP, using a dynamic communication schedule. Zero DP has three stages:
  - **ZeRO Stage 1**: Partitions optimizer states across devices. (4 times memory reduction with same communication volume as DP)
  - **ZeRO Stage 2**: Partitions optimizer states + gradients across devices. (8 times memory reduction with same communication volume as DP)
  - **ZeRO Stage 3**: Partitions optimizer states + gradients + model parameters across devices. (Memory reduction depends on the number of devices, with same 1.5 Times communication volume as DP)

- **Residual State Memory (ZeRO R)**: Memory consumed by activation, temporary buffers, and unusable fragmented memory.
    - **Activation Memory**: Memory consumed by activations during the forward pass, which is required for the backward pass can be significant for large language models. Activation checkpointing (or activation recomputation) is a common approach to reduce the activation memory by approximately the square root of the total activations at the expense of 33% re-computation overhead. The activation memory of a transformer-based model is proportional to the number of transformer layers × hidden dimensions × sequence length × batch size. For a GPT-2 like architecture the total activations is about 12 × hidden dim × batch × seq length.
    - **Temporary Buffers**: Memory consumed by temporary buffers during computation, such as  gradient all-reduce, or gradient norm computation tend to fuse all the gradients into a single flattened buffer before applying the operation in an effort to improve throughput. or example, the bandwidth of all-reduce across devices improves with large message sizes. While the gradient themselves are usually stored as fp16 tensors, the fused buffer can be an fp32 tensor depending on the operation. When the size of the model is large, these temporary buffer sizes are non-trivial.
    - **Fragmented Memory**: Additionally, it is possible to run out of usable memory even when there is plenty of available memory. This can happen with memory fragmentation. A request for a memory will fail if there isn’t enough contiguous memory to satisfy it, even if the total available memory is larger than requested. We observe significant memory fragmentation when training very large models, resulting in out of memory issue with over 30% of memory still available in some extreme cases

### Understanding Memory Reduction with ZeRO DP Stages:
Let us assume a model has X parameters, then in order to store its weights, and gradients in bf16/fp16 format, we need 2*(X + X ) memory. And if we are doing mixed precision training and want to store Adam optimizer states, then we need 4X (weights) + 4X (momentum) + 4X (variance) memory. Lets denote *K* as the memory multiplier of the optimizer states i.e., the additional memory required to store them is *K*X bytes. Mixed-precision Adam has K = 12. This results in 2X + 2X + *K*X = 16X bytes of memory requirement.

- **ZeRO Stage 1**: Partitions optimizer states, reducing optimizer state memory from *K*X bytes to $\frac{KX}{Num of Devices}$ bytes. When Number of devices is large, this can lead to significant memory savings. So, memory consumption reduced from 16X to ~ 4X bytes as $\frac{KX}{Num of Devices}$ is very small for large number of devices i.e. 4 times reduction.

- **ZeRO Stage 2**: Partitions optimizer states and gradients, reducing memory further to $\frac{(K+2)X}{Num of Devices}$ bytes.Using above logic , memory consumption reduced from 16X to ~ 2X bytes as $\frac{(K+2)X}{Num of Devices}$ is very small for large number of devices i.e. 8 times reduction.

- **ZeRO Stage 3**: Partitions optimizer states, gradients, and model parameters, reducing memory to $\frac{(K+4)X}{Num of Devices}$ bytes. This is the most memory-efficient stage. ZeRO stage 3 allows one to fit models with arbitrary size as long as there are sufficient number of devices to share the model states

### Understanding Communication Overhead in ZeRO DP:
- **ZeRO Stage 1**: Communication volume is similar to Data Parallelism (DP), as it only partitions optimizer states.
- **ZeRO Stage 2**: Communication volume increases slightly due to partitioning gradients, but remains manageable.
- **ZeRO Stage 3**: Communication volume increases further, but is still lower than full model replication in DP. It uses a dynamic communication schedule to optimize communication efficiency.



## Major pain points of distributed training:
- **Memory Redundancy**: Each device holds a complete copy of the model parameters, gradients, and optimizer states, leading to significant memory overhead.
- **Communication Overhead**: Synchronizing model parameters and gradients across devices can be slow and inefficient, especially with large models.
- **Scalability**: As model size increases, the memory requirements grow, making it challenging to scale training across multiple devices.



## References:
- [All Reduce Operations Analysis](https://oneflow2020.medium.com/how-to-derive-ring-all-reduces-mathematical-property-step-by-step-9951500db96)