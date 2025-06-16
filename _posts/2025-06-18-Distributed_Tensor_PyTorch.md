---
title: 'Distributed Tensor (DTensor) in PyTorch: Overview'
description: A brief note on DTensor's workings.
date: 2024-09-28 00:55:00 +0530
author: skrohit
categories: [DTensor, pytorch]
tags: [llms, DTensor, transformers, pytorch]
pin: false
---

> Note: Will not find anything new if you have already read [DTensor RFC](https://dev-discuss.pytorch.org/t/rfc-pytorch-distributedtensor/740) and [DTensor Status](https://dev-discuss.pytorch.org/t/dtensor-status-design-and-looking-forward/2749).
{: .prompt-tip }

## What is DTensor?
DTensor is a distributed tensor representation designed to efficiently train large deep learning models by reducing memory redundancy across multiple devices.

