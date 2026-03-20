---
title: 'PyTorch Memory Deep Dive: view, reshape, transpose, permute and the Contiguity Puzzle'
description: A practical deep dive into how PyTorch tensors use storage, stride, views and contiguity.
date: 2026-03-20 10:30:00 +0530
author: skrohit
categories: [pytorch, tensors]
tags: [pytorch, tensors, memory-layout, stride, contiguity, deep-learning]
pin: true
---

# PyTorch Memory Deep Dive: `view`, `reshape`, `transpose`, `permute` and the Contiguity Puzzle 🧠

If you use PyTorch long enough, you will eventually hit one of these:

- `RuntimeError: view size is not compatible with input tensor's size and stride`
- A silent copy created by `reshape()`
- A kernel running slower after `permute()` or `transpose()`

At first sight, these APIs look very similar because all of them seem to "rearrange" a tensor. But internally they are doing very different things. The key to understanding them is to understand three ideas:

- **Storage**: the underlying 1D memory buffer.
- **Size**: the logical shape of the tensor.
- **Stride**: how PyTorch walks memory when you move along each dimension.

Once these are clear, the contiguity puzzle becomes much easier.

## 📦 Tensor Memory Model
Let us start with a simple tensor:

```python
import torch

x = torch.arange(12)
x = x.view(3, 4)

print(x)
print("size:", x.size())
print("stride:", x.stride())
print("storage_offset:", x.storage_offset())
print("is_contiguous:", x.is_contiguous())
```

Output:

```python
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
size: torch.Size([3, 4])
stride: (4, 1)
storage_offset: 0
is_contiguous: True
```

The stride `(4, 1)` means:

- Move by `1` element in memory when advancing along columns.
- Move by `4` elements in memory when advancing along rows.

This is a standard row-major contiguous layout.

For a contiguous 2D tensor with shape `(3, 4)`, the element `x[i, j]` lives at:

```python
address = storage_offset + i * 4 + j * 1
```

So a tensor is not just its values. It is really:

- a storage buffer
- a shape
- a stride
- an optional storage offset

Most memory-layout confusion comes from changing one of these without changing the others.

## 👀 What is a View?
A **view** is another tensor object that looks at the same underlying storage.

```python
x = torch.arange(12).view(3, 4)
y = x.view(2, 6)

print(x.data_ptr() == y.data_ptr())
```

This prints `True`, meaning both tensors share the same storage.

If you modify one, the other also reflects the change:

```python
y[0, 0] = 100
print(x)
```

No data was copied here. Only the metadata changed.

This is why views are cheap and memory efficient.

## 🔧 `view()`: Fast but Strict
`view()` only works when PyTorch can reinterpret the same storage with a new shape **without changing the physical memory order**.

```python
x = torch.arange(12).view(3, 4)
y = x.view(2, 6)   # works
```

This works because `x` is contiguous and the new shape is compatible with its existing stride pattern.

Now look at this:

```python
x = torch.arange(12).view(3, 4)
t = x.transpose(0, 1)
print(t.size())      # torch.Size([4, 3])
print(t.stride())    # (1, 4)
print(t.is_contiguous())  # False

t.view(2, 6)         # RuntimeError
```

Why does it fail? Because `transpose()` changed how indices map to memory. The tensor `t` is still a view, but it is no longer contiguous in the standard row-major sense. `view()` is strict and refuses to pretend the memory is laid out differently than it really is.

In short:

- `view()` never rearranges data.
- `view()` only changes metadata.
- `view()` requires shape and stride compatibility.

## 🔄 `reshape()`: Flexible but Not Predictable
`reshape()` is usually what you want in user code when you are not sure whether the tensor is contiguous.

```python
x = torch.arange(12).view(3, 4)
t = x.transpose(0, 1)
r = t.reshape(2, 6)

print(r.shape)
print(r.is_contiguous())
```

`reshape()` tries to return a view when possible. If that is not possible, it makes a copy.

That is why `reshape()` is convenient, but there is an important caveat:

- Do not write code that depends on whether `reshape()` returned a view or a copy.

This matters for both memory usage and performance. Two calls that look identical at the Python level may behave differently depending on the tensor layout they receive.

Good mental model:

- **`view()`** = "I require a view"
- **`reshape()`** = "Give me this shape, use a view if you can"

## ↕️ `transpose()`: Swap Two Dimensions
`transpose(dim0, dim1)` swaps exactly two dimensions.

```python
x = torch.arange(24).view(2, 3, 4)
t = x.transpose(1, 2)

print("x.size:", x.size(), "x.stride:", x.stride())
print("t.size:", t.size(), "t.stride:", t.stride())
```

Typical output:

```python
x.size: torch.Size([2, 3, 4]) x.stride: (12, 4, 1)
t.size: torch.Size([2, 4, 3]) t.stride: (12, 1, 4)
```

Notice what changed:

- Shape changed from `(2, 3, 4)` to `(2, 4, 3)`
- Stride changed from `(12, 4, 1)` to `(12, 1, 4)`
- Storage did **not** change

So `transpose()` is usually cheap because it just returns a view with different metadata.

But that also means the output is often non-contiguous.

## 🔀 `permute()`: Generalized Dimension Reordering
`permute()` is a general version of `transpose()`. Instead of swapping two dimensions, it reorders all dimensions.

```python
x = torch.randn(2, 3, 4, 5)
y = x.permute(0, 2, 3, 1)

print("x.shape:", x.shape, "x.stride:", x.stride())
print("y.shape:", y.shape, "y.stride:", y.stride())
```

This is very common in deep learning code:

- `NCHW -> NHWC`
- `B, S, H, D -> B, H, S, D`
- sequence-first to batch-first conversions

Like `transpose()`, `permute()` usually returns a view. It does not physically reorder the underlying storage. It only changes how PyTorch interprets the same storage with a different stride pattern.

That is why a `permute()` is cheap by itself, but the next operation may become expensive if it expects contiguous memory.

## 🧩 What Exactly is Contiguity?
A tensor is **contiguous** when its stride pattern matches the expected row-major memory layout for its shape.

For shape `(2, 3, 4)`, the standard contiguous stride is:

```python
(12, 4, 1)
```

because:

- last dimension moves by `1`
- previous dimension moves by `4`
- first dimension moves by `3 * 4 = 12`

Let us see what `transpose()` does:

```python
x = torch.arange(12).view(3, 4)
t = x.transpose(0, 1)

print(x.stride())  # (4, 1)
print(t.stride())  # (1, 4)
```

If `t` were physically laid out as a contiguous `(4, 3)` tensor, its stride would have been `(3, 1)`. But its actual stride is `(1, 4)`. Therefore, `t` is non-contiguous.

This is the core idea:

- **Contiguous** means the current logical order matches the current physical layout.
- `transpose()` and `permute()` often break that match.

## 💡 Why Non-Contiguous Tensors Matter
Many PyTorch ops can handle non-contiguous tensors correctly. So non-contiguous does **not** automatically mean wrong.

But it can still matter for two reasons:

- Some APIs like `view()` require compatible stride layout and will fail.
- Some kernels run faster on contiguous inputs because memory access is simpler and more cache friendly.

A common pattern is:

```python
y = x.permute(0, 2, 3, 1)
y = y.contiguous()
```

This forces a real memory reordering so that `y` is now physically stored in its new logical order.

## 🛠️ `contiguous()`: When Metadata Change is Not Enough
`contiguous()` returns the same tensor if it is already contiguous. Otherwise, it allocates new memory and copies data into a contiguous layout.

```python
x = torch.arange(12).view(3, 4)
t = x.transpose(0, 1)
c = t.contiguous()

print(t.is_contiguous())  # False
print(c.is_contiguous())  # True
print(t.data_ptr() == c.data_ptr())  # False
```

After `contiguous()`, you can safely call `view()` again:

```python
z = t.contiguous().view(2, 6)
```

This works because `contiguous()` materialized the transposed layout into fresh memory.

## 🧪 A Simple Rule to Predict Behavior
Here is a practical cheat-sheet:

- If an op only changes shape and preserves element order, a view may be possible.
- If an op reorders dimensions like `transpose()` or `permute()`, the result is often non-contiguous.
- If you need a guaranteed metadata-only reshape, use `view()`.
- If you need shape conversion and do not care whether PyTorch copies, use `reshape()`.
- If a later op needs standard layout, call `contiguous()`.

## ⚠️ A Very Common Bug Pattern
This pattern appears often in model code:

```python
x = x.permute(0, 2, 3, 1)
x = x.view(batch_size, -1)   # may fail
```

The fix is usually one of these:

```python
x = x.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
```

or simply:

```python
x = x.permute(0, 2, 3, 1).reshape(batch_size, -1)
```

The first version makes the copy explicit. The second version lets PyTorch decide.

In performance-sensitive code, the explicit version is often better because it makes data movement visible.

## 🧠 `view()` vs `reshape()` vs `transpose()` vs `permute()`
The differences are easier to remember in one table:

| API | What it does | Usually shares storage? | Can make copy? | Often non-contiguous? |
| --- | --- | --- | --- | --- |
| `view()` | Reinterpret shape | Yes | No | No, if it succeeds |
| `reshape()` | Change shape | Maybe | Yes | Result may be either |
| `transpose()` | Swap two dims | Yes | No for strided dense tensors | Yes |
| `permute()` | Reorder dims | Yes | No for strided dense tensors | Yes |
| `contiguous()` | Materialize standard layout | No if copy needed | Yes | No |

## 🎯 Practical Advice
In real model code, these rules are usually enough:

- Prefer `reshape()` in high-level code if you only care about final shape.
- Prefer `view()` when you explicitly require no copy.
- Expect `transpose()` and `permute()` to return non-contiguous views.
- Use `contiguous()` intentionally, not mechanically, because it may allocate and copy.
- When debugging layout issues, always print `shape`, `stride`, `storage_offset`, and `is_contiguous()`.

That last point saves a lot of time.

## 📚 References
- [PyTorch Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)
- [torch.Tensor.view](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html)
- [torch.reshape](https://docs.pytorch.org/docs/stable/generated/torch.reshape.html)
- [torch.transpose](https://docs.pytorch.org/docs/stable/generated/torch.transpose.html)
- [torch.Tensor.contiguous](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html)

---
*"In PyTorch, shape tells you how a tensor looks. Stride tells you how it walks through memory."*
