---
title: "Enabling Encoder–Decoder Models on Intel XPU in SGLang: A KV-Cache Story"
date: 2026-08-27
tags: [sglang, xpu, kv-cache, flash-attention, whisper, encoder-decoder]
---

*This post walks through a KV Cache misalignment issue for encoder–decoder models (Whisper, Llama-3.2-Vision/Mllama) on Intel XPU in SGLang. We will start our discussion with how SGLang manages its KV cache and then discuss the issue with possible fixes*

## Introduction

Before we get to SGLang or XPU, it's worth grounding two ideas that the rest of this post leans on: the **KV cache** and **paged attention**. Almost everything about serving-time memory in a modern inference engine is built on these two.

Let us start with the KV cache. A transformer generates text autoregressively — one token at a time — and each new token attends to the keys and values of *every* token before it. Those keys and values only depend on tokens already produced, so recomputing them at every step would be pure waste. Instead we compute each token's key/value once and **cache** them; the next step just appends its own K/V and attends over the whole cache. This is the KV cache, and during serving it grows by one entry per generated token and quickly becomes the dominant consumer of GPU memory.

That growth is exactly what makes the cache hard to store. The naive layout gives each sequence one contiguous block of memory sized to its maximum length — but sequences arrive and finish at different times and reach very different lengths, so the pool fragments badly and a lot of reserved space is never used. **Paged attention** (introduced by [vLLM's PagedAttention](https://arxiv.org/abs/2309.06180)) borrows the operating system's virtual-memory trick to fix this: chop the KV cache into fixed-size **blocks**, or *pages*, drawn from one shared pool, and give each sequence a **page table** — the list of block indices its tokens occupy. The attention kernel reads K/V *through* that page table, so a single sequence's cache can live in physically non-contiguous pages. Fragmentation disappears (any freed page is reusable by anyone), sequences can grow by grabbing one page at a time, and identical prefixes can even *share* pages. The size of each page — the **page size** — is the central knob of the whole scheme.

SGLang implements precisely this design, and the page size turns out to interact in a surprisingly subtle way with **encoder–decoder** models such as Whisper and Llama-3.2-Vision (Mllama), especially on Intel XPU. The rest of this post builds up SGLang's KV-cache machinery around the page size, shows how encoder–decoder models fit into it, and then compares two ways to make them work on XPU: **page-aligning the encoder** and **gathering into a ragged buffer for variable-length FlashAttention**. I'll assume familiarity with transformer attention; everything else we'll build up from here.

## How SGLang stores the KV cache: two pools and a page size

The first thing to internalise is that SGLang splits KV-cache bookkeeping into **two** separate pools, both defined in `python/sglang/srt/mem_cache/memory_pool.py`.

The first is the **`ReqToTokenPool`**, wired up in `python/sglang/srt/mem_cache/kv_cache_configurator.py`. Its heart is a single 2-D integer tensor, allocated once in the constructor:

```python
# ReqToTokenPool.__init__  (memory_pool.py)
self.req_to_token = torch.zeros(
    (size + 1, max_context_len), dtype=torch.int32, device=device
)
self.free_slots = list(range(1, size + 1))   # a free-list of ROW indices (req_pool_idx)
```

`req_to_token[req, position]` is the **slot index** where token `position` of request `req` stores its key and value. The table is *token-granular*: one column per token position, never per page — it doesn't know pages exist. Each **row** belongs to one in-flight request: `alloc()` pops a free row off `free_slots` and stamps it onto the request as its `req_pool_idx`, and `free()` returns the row when the request finishes. (Row 0 is a padding row — CUDA-graph-padded batches default their request index to 0, so dummy reads/writes land there harmlessly, which is why the tensor is `size + 1` rows.)

So the **shape** of `req_to_token` is `[max_num_reqs + 1, max_context_len]`, and it is fixed at startup. Two command-line arguments decide those two dimensions:

| Dimension | Value | Comes from |
|---|---|---|
| **rows** = `max_num_reqs` (+1 padding) | how many requests can be tracked at once | `--max-running-requests` (read as `get_schedule().max_running_requests`) |
| **columns** = `max_context_len` = `model_config.context_len + extra` | how long any one request may grow, with a small `extra = 4 (+ speculative_num_draft_tokens)` slack | the model's context length / `--context-length` (plus the speculative-decoding flags) |

That per-request row bookkeeping is deliberately simple: `free_slots` is just a Python list used as a stack of available row indices, and two small methods manage it:

```python
def alloc(self, reqs):
    reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]  # chunked prefill keeps its row
    need_size = len(reqs) - len(reusing)
    if need_size > len(self.free_slots):
        return None                          # not enough rows -> scheduler can't admit this batch
    select = self.free_slots[-need_size:]    # pop from the TAIL (O(need_size))
    del self.free_slots[-need_size:]
    # assign each new req its req_pool_idx from `select`; bump req_generation[idx]
    ...

def free(self, req):
    self.free_slots.append(req.req_pool_idx)  # row handed back for reuse (LIFO)
    req.req_pool_idx = None
```

In short, these two methods just **allocate and release the row a request lives in**: `alloc` claims a free row for each new request, and `free` hands it back when the request finishes. (Only the *rows* recycle here; the KV *slots* those rows point at are recycled by the second pool.)

The second is the **`MHATokenToKVPool`** (and `MLATokenToKVPool` for latent attention). This is where the actual tensors live: one big contiguous buffer per layer, shaped roughly `[max_total_tokens, num_kv_heads, head_dim]`, allocated **once at server start**. When you see a log line like `KV Cache is allocated. dtype: torch.bfloat16, #tokens: 65536`, that is this pool being sized.

So the mental model is a classic indirection: `req_to_token` maps *(request, position) → slot*, and the KV pool is a flat array of slots. Nothing here is fragmented, because the pool is one pre-allocated block.

The interesting part is *how those slot numbers are handed out and reclaimed*, and that is where the **page size** enters. Both allocators live under `python/sglang/srt/mem_cache/allocator/` and share one idea — a **free-list** of what's available — but differ in the unit they track.

**`TokenToKVPoolAllocator` (`token.py`, `page_size == 1`).** Its free-list is a 1-D tensor of individual **slot indices**, initialised in `clear()` to `torch.arange(1, size + 1)` (slot 0 is reserved as padding, mirroring `req_to_token`'s row 0). Allocation and freeing are just stack operations on that tensor:

- `alloc(need_size)` pops the first `need_size` slots — `sel = free_pages[:need_size]; free_pages = free_pages[need_size:]` — and returns them; those *are* the `out_cache_loc`. If fewer than `need_size` remain it returns `None` (the same back-pressure signal as the row pool).
- `free(indices)` concatenates the freed slots back onto the free-list (or, if a `free_group` batch is open, defers them into `release_pages` and flushes with one `cat` at `free_group_end`).

Slots are handed out individually, so a sequence's tokens can end up on arbitrary, non-contiguous slots — which is fine at `page_size == 1`, because the page table simply *is* the token-slot table.

**`PagedTokenToKVPoolAllocator` (`paged.py`, `page_size > 1`).** Same interface, but the free-list tracks **page indices** (`num_pages = size // page_size`), and it hands out whole pages expanded to their slots:

- `alloc(need_size)` (page-aligned) pops `need_size // page_size` free pages and expands each page `p` to its slots `p * page_size + arange(page_size)`, returning the flat slot indices.
- `alloc_extend(prefix_lens, seq_lens, last_loc, ...)` / `alloc_decode(...)` are the growth paths used during prefill/decode: they first **fill the partially-used last page** of the sequence (starting at `last_loc`), then grab new pages for the overflow. This is what keeps a sequence's tokens **contiguous within each page** — the invariant the backend relies on when it recovers page indices with `req_to_token[:, ::page_size] // page_size`.
- `free(indices)` maps freed slots back to their pages (`torch.unique(indices // page_size)`) and returns those pages to the free-list.

Because both allocators draw fixed-size units from one shared pool, large-batch serving doesn't fragment: a freed page (or slot) is immediately reusable by any other sequence, whatever its length.

It's worth being concrete about *what* gets written into a row and *where those numbers come from*. When a batch is scheduled, `alloc_for_extend` (`python/sglang/srt/mem_cache/allocation.py`) asks the allocator for `extend_num_tokens` fresh slots — `alloc_token_slots` at `page_size == 1`, or a page-based allocation otherwise — and gets back `out_cache_loc`, a 1-D tensor of **physical slot numbers** from the pool's free list. It then calls `write_cache_indices` to lay those (and any reused) slots into the request's row.

`write_cache_indices` is the piece that actually fills a row, and it does so in **two parts**, because not all of a request's tokens are new: some may already have KV in the pool — a shared prefix served from the radix cache, or the earlier chunks of a chunked prefill — while the rest are being computed this step. So for each request *i* it writes:

```python
# write_cache_indices (allocation.py) — Python fallback; a Triton kernel does the same
req_to_token_pool.write((req_idx, slice(0, prefix_len)), prefix_tensors[i])                          # reused prefix slots
req_to_token_pool.write((req_idx, slice(prefix_len, seq_len)), out_cache_loc[pt : pt + extend_len])  # freshly allocated slots
```

The first write copies the **already-cached** slot numbers for the prefix (`prefix_tensors[i]`) into columns `[0:prefix_len]`; the second scatters the **freshly allocated** `out_cache_loc` slots into columns `[prefix_len:seq_len]`. Afterwards the row is a complete token→slot map for the whole sequence, mixing reused and new slots — which is exactly why a prefix-cache hit, or the continuation of a chunked prefill, costs no new allocation for its prefix portion.

Two implementations do this, chosen by whether the prefill attention backend supports Triton: a **Triton kernel** (`write_req_to_token_pool_triton`) that launches one program per request and fills all rows in parallel with no per-request host syncs — it receives an array of `prefix_pointers` (the data pointers of each `prefix_tensors[i]`, since every request's prefix buffer is a separate tensor) — and the **Python fallback** shown above, a per-request loop that pays a few `.item()` syncs. Both produce the identical row layout.

So the **values** living in `req_to_token` are indices into the KV pool, ranging over `[0, max_total_num_tokens)`. And `max_total_num_tokens` — the number of KV slots — is itself derived at startup from how much memory is left for the cache, i.e. from `--mem-fraction-static` and `--max-total-tokens`. Those two flags don't change the *shape* of `req_to_token`, but they bound the *range* of the numbers inside it (and can force `max_running_requests`, hence the row count, down when memory is tight). `--page-size` is different again: it changes only *how* those slots are allocated — a whole page at a time versus one token at a time — and never the token-granular shape of `req_to_token`.

The page size itself is resolved from server args. On CUDA the default is `page_size = 1` (see `_page_size_default` in `python/sglang/srt/arg_groups/overrides.py`). On Intel XPU it is **forced** to 64 or 128 by `_intel_xpu_page_constraint` in the same file — the XPU FlashAttention kernels are only compiled for those page sizes. Hold onto that fact; it is the whole story.

### How the pieces link up

Stepping back, one batch step wires all of these together. `ReqToTokenPool` owns the *rows* (which request lives where), the allocator owns the *slots/pages* (which KV-cache cells are free), `write_cache_indices` stitches them into `req_to_token`, and the attention backend reads the `MHATokenToKVPool` tensors through that map:

```text
per batch step  (managers/schedule_batch.py -> mem_cache/allocation.py)

   Scheduler admits requests
        |                                        |
        v                                        v
   ReqToTokenPool.alloc(reqs)            allocator.alloc / alloc_extend(need_size)
     -> req_pool_idx  (a ROW)              -> out_cache_loc  (KV-slot indices)
        |                                        |
        +--------------------+-------------------+
                             v
        write_cache_indices(req_pool_idx, out_cache_loc, prefix_tensors)
                             |   fills the request's row:
                             v
   req_to_token[req_pool_idx] = [ reused prefix slots | new out_cache_loc slots ]
                             |   (each value indexes a slot in the pool below)
                             v
   MHATokenToKVPool : per-layer K/V buffer  [ max_total_num_tokens, num_kv_heads, head_dim ]
                             ^
                             |   init_forward_metadata slices req_to_token[row] -> page_table
                             |   (paged: // page_size) ; the kernel reads/writes KV at those slots
                   XPUAttentionBackend.forward_extend / forward_decode

on finish / eviction:
   ReqToTokenPool.free(req)        -> ROW returned to its free-list
   allocator.free(out_cache_loc)   -> SLOTS/PAGES returned to the allocator free-list
```

Two independent free-lists recycle in lockstep: the **row** free-list in `ReqToTokenPool` and the **slot/page** free-list in the allocator. `req_to_token` is the join between them, and the page size only changes how the allocator's slots are grouped and how the backend turns that row into a page table — everything else is identical.

## From slots to a page table: how attention reads the cache

The attention kernel doesn't consume `req_to_token` directly. The paged FlashAttention kernel wants a **page table** — per request, the list of *block* indices its tokens occupy — because internally it resolves a key at logical position `t` as:

```
slot = page_table[req, t // page_size] * page_size + (t % page_size)
```

SGLang derives that page table from `req_to_token` with a stride-and-divide, which you can see in the XPU backend's `init_forward_metadata` (`python/sglang/srt/layers/attention/xpu_backend.py`):

```python
# Convert the token-granular slot table into a page-granular block table
if self.page_size > 1:
    strided = torch.arange(0, page_table.shape[1], self.page_size, device=self.device)
    metadata.page_table = req_to_token[:, strided] // self.page_size
```

This works only because paged allocation puts a request's tokens in contiguous slots within a page: taking every `page_size`-th column and dividing by `page_size` recovers the block index. At `page_size == 1` the transform is the identity (`x // 1 == x`), which is why CUDA never thinks about any of this — the token-granular table *is* the page table.

That single `// page_size` is the seam everything tears along.

## Encoder–decoder models: two KV regions in one row

Decoder-only models have one kind of attention. Encoder–decoder models like Whisper and Mllama have two, and they share a single `req_to_token` row.

During prefill, SGLang prepends the encoder's output as "tokens" to the request. `prepare_encoder_info_extend` in `python/sglang/srt/managers/schedule_batch.py` then splits the allocated `out_cache_loc` into two contiguous regions:

- `encoder_out_cache_loc` — the first `encoder_len` slots, holding the **encoder's** key/value (consumed by *cross-attention*).
- `decoder_out_cache_loc` — the remainder, holding the decoder's own key/value (consumed by *self-attention*).

So one row of `req_to_token` looks like this:

```
req_to_token[req] = [ encoder KV slots (0 : encoder_len) | decoder KV slots (encoder_len : seq_len) ]
```

A layer knows which region to read from a flag on `RadixAttention`: `layer.is_cross_attention`. Cross-attention points at the encoder region (`encoder_page_table`, `encoder_lens`); self-attention points at the decoder region (offset by `encoder_len`). For Whisper-large-v3 the encoder always emits exactly **1500** tokens, so `encoder_len = 1500` for every request.

## Why XPU breaks: page size 64/128 meets a token-granular cache

Now put the two facts together. XPU forces `page_size = 128`. The encoder occupies `req_to_token` columns `[0:1500]`, and `1500 % 128 = 92`. Two things go wrong.

**Cross-attention mis-indexes.** Here is the crucial asymmetry. The `// page_size` conversion from the previous section is applied only to the *self-attention* table (`metadata.page_table`, and the SWA table) — the **`encoder_page_table` is never run through it.** It is built as `req_to_token[:, :1500]` and handed to the paged kernel **raw**, still holding **token-slot indices** (page_size=1 semantics). The kernel then does what it always does: it treats each entry as a *block* index and multiplies by `page_size`. So the first encoder slot — say `128` — is read as block `128` → physical slot `128 * 128`, and cross-attention reads entirely the wrong rows. (You might ask why SGLang doesn't just divide `encoder_page_table` by `page_size` too. It can't meaningfully: the encoder occupies token-granular slots `[0:1500]`, which is genuinely page_size=1 data — turning it page-granular requires either the page-alignment reshuffle or the gather described in the two approaches below. On CUDA the missing conversion is harmless only because `page_size == 1` makes it an identity.)

**Decoder self-attention is also misaligned.** The decoder region starts at slot `base + 1500`, and `1500` is not a multiple of 128, so the strided `// page_size` page table for self-attention no longer lines up with page boundaries; it reads encoder slots as if they were decoder KV.

And the degenerate case is worse than "wrong output." During text-only warmup Whisper runs with `encoder_lens = 0` — a cross-attention over an empty key set. I verified directly on the hardware that feeding an empty KV to the XPU kernel doesn't return zeros or NaNs; it faults the device with `UR_RESULT_ERROR_DEVICE_LOST`. That was my warmup hang.

On CUDA none of this happens, purely because `page_size == 1` there: the token-granular table is already correct, and the CUDA kernel has a `page_size == 1` code path. XPU has neither.

So the problem statement is precise: *on XPU the encoder KV is stored token-granularly, but the paged kernel demands page-granular block indices and only supports page sizes 64/128.* There are two honest ways out.

## Approach 1: page-align the encoder (the `encoder_offset` method)

The first approach keeps using the paged kernel and makes the layout satisfy it. The idea is to **reserve a page-aligned amount of space for the encoder** so that the decoder region always begins on a page boundary.

Concretely, instead of reserving exactly `encoder_len` slots, reserve `ceil_align(encoder_len, page_size)`:

```python
encoder_reserve = ceil_align(encoder_len, page_size)   # e.g. ceil_align(1500, 128) = 1536
```

That change ripples outward. In `prepare_encoder_info_extend` you split `out_cache_loc` on the padded reserve instead of the raw length, leaving a gap of up to `page_size - 1` slots. In the attention backends you offset the decoder's page table by the aligned amount — for the FlashInfer backend this is the `kv_start_idx` in `python/sglang/srt/layers/attention/flashinfer_backend.py`, which becomes `ceil_align(encoder_lens, page_size)` instead of `encoder_lens`; for the XPU backend it is an `encoder_offset` plus the usual `// page_size` stride. Because `ceil_align(x, 1) == x`, every one of these changes is a no-op on CUDA, which is what keeps the CUDA path byte-identical.

I find this approach conceptually clean — "just make everything page-aligned" — but it has real downsides. It touches the scheduler *and* two attention backends *and* the graph-capture metadata, so the change is spread across many files and easy to get subtly wrong. It wastes up to `page_size - 1` slots of internal fragmentation per request. And crucially it still hands the empty encoder case to the paged kernel, so the `encoder_lens == 0` device fault is not solved by alignment alone — you also need a `skip_cross_attention` guard threaded through the model. It makes the paged kernel *work*, but it doesn't make the mismatch *go away*.

## Approach 2: gather and go ragged (variable-length FlashAttention)

The second approach is the one I shipped, and it comes from a neat observation about how FlashAttention is actually exposed on XPU. There are **two** entry points, and they are the same kernel underneath (`torch.ops.sgl_kernel.fwd` → `mha_fwd` in `sgl-kernel-xpu`), differing only in how they address K/V:

- `flash_attn_with_kvcache` — the **paged** entry. You pass `k_cache`/`v_cache` (the block pool), a `page_table`, and `cache_seqlens`; it reads scattered blocks in place. This is the one that needs page size 64/128.
- `flash_attn_varlen_func` — the **ragged** entry. You pass `k`/`v` as *contiguous* tensors plus `cu_seqlens_q`/`cu_seqlens_k` (cumulative-length boundaries); it walks them by simple offsets. No page table, no page-size constraint.

The encoder KV is token-granular — that is precisely "page_size = 1 semantics." So instead of forcing it into the paged kernel, I gather it and feed the ragged kernel. In `_varlen_gather_attn` (`python/sglang/srt/layers/attention/xpu_backend.py`) the core is:

```python
# page_table holds token-slot indices; cache_seqlens caps each row.
valid      = torch.arange(m)[None, :] < seqlens[:, None]   # which columns are real keys
flat_slots = page_table[valid]                              # packed per request, cu_seqlens order
k_ragged   = k_flat.index_select(0, flat_slots).contiguous()  # gather scattered slots -> dense
v_ragged   = v_flat.index_select(0, flat_slots).contiguous()
out = flash_attn_varlen_func(q, k_ragged, v_ragged,
                             cu_seqlens_q, cu_seqlens_k,
                             max_seqlen_q, max_seqlen_k, causal=causal, softcap=softcap)
```

The `index_select` is a **gather copy**: it collects the request's scattered token-slots into one contiguous buffer that the ragged kernel can read by offset. I use this for both cross-attention (encoder keys, `causal=False`) and decoder self-attention (decoder keys, `causal=True`), so the entire encoder–decoder attention path bypasses the paged kernel and its page-size limitation. There is no `ceil_align`, no `encoder_offset`, no stride-and-divide, and no change to the scheduler — the fix lives almost entirely in one backend file.

Two details matter. First, I skip the `// page_size` transform for encoder–decoder metadata, because the varlen path *wants* the raw token-slot indices. Second, the empty case gets an explicit guard:

```python
if max_seqlen_k == 0:
    # Empty-KV faults the XPU device (UR_RESULT_ERROR_DEVICE_LOST from mha_fwd,
    # which has no zero-length guard). Return a zero context instead of launching.
    return q.new_zeros((q.shape[0], q.shape[1], v_flat.shape[-1]))
```

That one branch is what makes the text-only warmup safe, and it removes the need for any model-side `skip_cross_attention` plumbing.

The trade-off is honest: the ragged path pays a gather copy of `sum(seqlens)` rows per forward, whereas the paged kernel reads in place. It also can't be captured into a fixed-shape graph (the ragged buffer size is data-dependent), so it is eager-only — which on XPU costs nothing today, since XPU disables CUDA-graph capture by default. In exchange it is page-size-agnostic, it handles the empty case, and it is contained to a single file.

One more thing I want to stress, because it confused me at first: **the anti-fragmentation property is not sacrificed by going ragged.** Fragmentation is prevented by the *paged allocator*, which still owns storage; the ragged kernel only changes how KV is *read* (a transient gather), not how it is *allocated*. Storage stays paged; the copy is temporary.

## Results

I verified the varlen approach end-to-end on an Intel XPU. Whisper-large-v3 transcribed real audio correctly — the classic "*ask not what your country can do for you*" clip came back verbatim, with `page_size` forced to 128 and `cuda graph: False`, and the warmup no longer hangs.

To prove it generalises beyond Whisper, I ran Llama-3.2-11B-Vision (Mllama, also encoder–decoder) on `docvqa_val`:

| Config | Concurrency observed | DocVQA ANLS (2000 samples) |
|---|---|---|
| `--max-running-requests 1` | serialized | 0.8296 |
| `--max-running-requests 4` | up to 4 batched | 0.8284 |

The two numbers are within one standard error of each other, which tells me the per-request gather is correct even when several encoder–decoder requests batch together — the case that the old `encoder_lens.numel() == 1` restriction used to forbid.

## Caveats and takeaways

A few honest caveats. The ragged path is eager-only; if XPU graph capture ever lands, the encoder–decoder metadata builders would need real work, and I left a loud assertion in the backend constructor so that a future graph run fails clearly instead of silently mis-indexing. GQA/MQA head ordering is handled inside the kernel, so I kept the on-device correctness test to the MHA case that Whisper actually uses. And the gather copy is a genuine cost — for very long decoder sequences the in-place paged read would win, but transcription and VQA decode short enough that it doesn't matter.

If I had to compress the whole journey into one sentence: *the bug was never in the model — it was a page size of 128 meeting a cache that was really page size 1, and the cleanest fix was to stop pretending otherwise and read that cache with the variable-length kernel that was built for it.*
