---
title: Notes on torch.compile, AOTAutograd and Backend Integration of Intel Gaudi
description: A practical note on how torch.compile works internally, with focus on TorchDynamo, AOTAutograd, autograd integration and backend contracts.
date: 2026-03-20 22:15:00 +0530
author: skrohit
categories: [pytorch, torch-compile]
tags: [pytorch, torch.compile, torchdynamo, aotautograd, autograd, fx, gaudi, backend]
pin: false
---
# Notes on `torch.compile`, AOTAutograd and Backend Integration 🚀

## 📊 What is `torch.compile`
`torch.compile` is PyTorch's compiler entry point for taking normal eager PyTorch code and running it through a compilation pipeline. The goal is simple: keep the user facing programming model mostly unchanged, but make execution faster by tracing the program into graphs, transforming those graphs, and letting a backend compile them.

At a high level, the pipeline looks like this:

```text
Python model/function
  -> TorchDynamo captures a graph
  -> AOTAutograd prepares forward/backward or inference graph
  -> backend compiles graph(s)
  -> wrapped runtime callable is returned
```

The important thing to remember is that `torch.compile` is not one compiler. It is a pipeline and a coordination layer. Multiple components participate:
- **TorchDynamo**: captures Python into FX graphs.
- **AOTAutograd**: prepares functional forward/backward graphs and runtime wrappers.
- **Backend**: compiles FX graphs into executable callables.

---
## 🧠 What does `torch.compile` return
`torch.compile` returns a Python callable or module wrapper that behaves like the original eager function/module, but internally executes compiled artifacts when possible.

That returned object is not just "compiled code". It usually contains:
- graph capture decisions,
- guard checks,
- compiled forward callable,
- sometimes compiled backward callable,
- runtime wrappers to preserve eager semantics.

This last point is important. A compiled graph alone is not enough. PyTorch eager semantics include:
- in-place mutations,
- aliasing and view relationships,
- autograd behavior,
- grad mode behavior,
- random number generator behavior.

The compile pipeline must preserve all of those.

---
## 🎯 TorchDynamo's role
TorchDynamo is the front-end of `torch.compile`. It watches Python bytecode execution and extracts a graph region that can be compiled. The output of this stage is generally an **FX graph**, represented by `torch.fx.GraphModule`.

When `torch.compile(..., backend="hpu_backend")` is used, Dynamo resolves the backend name, captures a graph, and hands the captured `GraphModule` plus example inputs to the backend.

In the backend registry path, this resolution happens through PyTorch's backend lookup machinery. On the Gaudi side, the backend entrypoint is:

[`backends.py`](https://github.com/HabanaAI/gaudi-pytorch-bridge/tree/1.23.0/python_packages/habana_frameworks/torch/dynamo/compile_backend/backends.py)

The key function is:

```python
@register_backend
def hpu_backend(graph_module, example_inputs, **kwargs):
    ...
    return aot_autograd(...)(graph_module, example_inputs)
```

This means the Gaudi backend does not directly compile the Dynamo graph. It first routes it through **AOTAutograd**.

---
## 🧩 Why AOTAutograd exists
TorchDynamo gives you a forward graph. But real training needs more than that. Training needs:
- forward graph,
- backward graph,
- autograd integration,
- correct handling of mutations and aliases.

That is where AOTAutograd comes in.

The main purpose of AOTAutograd is:
- trace forward and backward ahead of time,
- make the graph more functional and backend-friendly,
- partition joint graph into forward and backward graphs,
- ask backend to compile those graphs,
- return a runtime callable that still behaves like eager PyTorch.

The main entry point is:

[`aot_autograd.py`](https://github.com/pytorch/pytorch/tree/main/torch/_functorch/aot_autograd.py)

Important functions:
- `aot_function`
- `aot_module`
- `aot_module_simplified`
- `_create_aot_dispatcher_function`
- `aot_export_module`
- `_aot_export_function`

For `torch.compile` with Dynamo, the common path is `aot_module_simplified`, not `aot_function`.

---
## 🔧 `aot_module_simplified` in the `torch.compile` path
For Dynamo-produced graphs, AOTAutograd uses:

[`aot_module_simplified(...)`](https://github.com/pytorch/pytorch/tree/main/torch/_functorch/aot_autograd.py)

Why this "simplified" path exists:
- Dynamo already flattened the problem into a static graph-like form.
- It avoids some generic pytree overhead.
- It avoids repeatedly re-reading params and buffers.
- It is cheaper than the more general `aot_module`.

Roughly, `aot_module_simplified` does this:

```text
collect params and buffers
  -> prepend them to runtime args
  -> build AOTConfig
  -> create fake inputs
  -> call create_aot_dispatcher_function(...)
  -> get back compiled callable
  -> wrap it as returned forward function
```

This is the place where AOTAutograd starts taking ownership of the backend flow.

---
## 🧪 Fake tensors and symbolic shapes
Before tracing, AOTAutograd creates or detects **fake tensors**. Fake tensors carry metadata like:
- shape,
- dtype,
- device,
- sometimes symbolic dimensions,

without requiring real device execution.

This is useful because PyTorch wants to analyze the computation and build graphs without actually running real kernels on real data.

Related logic lives around:

[`aot_autograd.py`](https://github.com/pytorch/pytorch/tree/main/torch/_functorch/aot_autograd.py)

Key idea:
- use real program structure,
- but use fake values for tracing and metadata analysis.

This keeps compile-time analysis cheaper and safer.

---
## 🔍 Where tracing actually happens
This is one of the most important internals.

Tracing does **not** happen directly inside `jit_compile_runtime_wrappers.py`. That file is mostly orchestration and runtime wrapping.

The actual graph tracing is triggered from:

[`dispatch_and_compile_graph.py`](https://github.com/pytorch/pytorch/tree/main/torch/_functorch/_aot_autograd/dispatch_and_compile_graph.py)

The key helper is:

```python
def _create_graph(f, args, *, aot_config):
    with enable_python_dispatcher(), FunctionalTensorMode(...):
        fx_g = make_fx(...)(*args)
```

This is where dispatch-based tracing happens.

### What does "dispatch-based tracing" mean
PyTorch tensor operations go through a dispatcher. When tracing is enabled through mechanisms like `make_fx`, `FunctionalTensorMode`, and python dispatcher modes, tensor ops are intercepted and turned into FX nodes instead of just executing normally.

In simple words:
- your Python function runs,
- tensor ops are intercepted,
- those ops are recorded into an FX graph.

For inference-only path, AOTAutograd calls:
- `aot_dispatch_base_graph(...)`

For training path, AOTAutograd calls:
- `aot_dispatch_autograd_graph(...)`

Both eventually call `_create_graph(...)`.

---
## 🛠️ Functionalization: why mutations are rewritten
Backends usually prefer **functional** graphs. That means:
- no in-place updates,
- no side-effect-heavy ops in the graph body,
- inputs and outputs behave like pure function inputs and outputs.

But user PyTorch code is often not purely functional. It may do:

```python
x.mul_(2)
```

or

```python
x.t_()
```

AOTAutograd handles this by rewriting mutation-heavy code into a functional form. For example:

```python
def f(x):
    x.mul_(2)
    return x.mul(3)
```

becomes conceptually:

```python
def compiled_forward_graph(x):
    x_updated = x.mul(2)
    out = x_updated.mul(3)
    return x_updated, out
```

Then after graph execution, AOTAutograd performs an **epilogue** step that applies the update back to the original tensor:

```python
x.copy_(x_updated)
```

This lets the graph be compiler-friendly while preserving eager behavior.

---
## 🧾 What is the epilogue
The **epilogue** is runtime work that happens after a compiled graph returns. It is not part of the backend compiled graph itself. It is AOTAutograd's way of restoring user-visible eager semantics.

Typical epilogue work:
- copy updated inputs back to original inputs,
- replay metadata changes like shape/stride updates,
- rebuild aliased outputs or views in a safe way,
- restore some runtime state.

This logic is implemented in:

[`runtime_wrappers.py`](https://github.com/pytorch/pytorch/tree/main/torch/_functorch/_aot_autograd/runtime_wrappers.py)

especially in `_create_runtime_wrapper(...)`.

This is why AOTAutograd is not "just tracing and compiling". It also owns part of runtime semantics.

---
## 🧵 Inference path vs training path
The central decision point is:

[`_create_aot_dispatcher_function(...)`](https://github.com/pytorch/pytorch/tree/main/torch/_functorch/aot_autograd.py)

This function:
- collects metadata,
- checks if autograd is needed,
- handles decompositions,
- chooses one of the dispatchers:
  - `aot_dispatch_base` for inference,
  - `aot_dispatch_autograd` for training,
  - `aot_dispatch_export` for export.

### Inference path
If no backward support is needed, AOTAutograd:
- builds forward-only graph,
- compiles it with `inference_compiler` if provided, otherwise `fw_compiler`,
- wraps the compiled callable in runtime epilogue logic.

This path is simpler because there is no custom autograd node for backward.

### Training path
If backward is needed, AOTAutograd:
- builds a **joint graph** containing forward and backward related logic,
- partitions that graph into forward and backward graphs,
- compiles forward and backward separately,
- builds a custom `torch.autograd.Function`,
- wraps that in runtime epilogue logic.

This training path is the most important part of AOTAutograd.

---
## 🔄 What is a joint graph
A **joint graph** is a combined graph representation that contains enough information to derive:
- forward computation,
- backward computation,
- tensors that need to be saved for backward.

Why build a joint graph first instead of tracing fw and bw independently:
- backward depends on exactly what happened in forward,
- saved tensors and tangent structure must match,
- partitioning can optimize what gets saved or recomputed.

In AOTAutograd training flow:

```text
original function
  -> prepared for autograd
  -> joint graph traced
  -> partition_fn splits it into fw_module and bw_module
```

This happens in:

[`dispatch_and_compile_graph.py`](https://github.com/pytorch/pytorch/tree/main/torch/_functorch/_aot_autograd/dispatch_and_compile_graph.py)

and then:

[`jit_compile_runtime_wrappers.py`](https://github.com/pytorch/pytorch/tree/main/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py)

---
## ✂️ What partitioning means
Partitioning means splitting the joint graph into:
- a forward graph,
- a backward graph.

PyTorch AOTAutograd lets backends supply their own partition function. That matters because different backends may want:
- different rematerialization tradeoffs,
- different saved tensor structure,
- different graph layouts.

The Gaudi backend provides:

[`partition_fn.py`](https://github.com/HabanaAI/gaudi-pytorch-bridge/tree/1.23.0/python_packages/habana_frameworks/torch/dynamo/compile_backend/partition_fn.py)

Its `hpu_partition(...)`:
- can remove unnecessary clones,
- can constant-fold joint graph,
- can run SDPA fusion,
- uses `default_partition`,
- reorders backward to mimic autograd engine execution order,
- optionally marks reusable backward inputs.

So in Gaudi, partitioning is not just a split. It is also an optimization point.

---
## 🧠 Why AOTAutograd creates a custom `torch.autograd.Function`
This is the key training-time idea.

During eager execution, if you run:

```python
y = x.sin() * 2
```

PyTorch creates a chain of autograd nodes internally. Each differentiable op contributes to the autograd graph, and `y.grad_fn` points into that graph.

But after compilation, the forward region becomes an opaque callable like:

```python
y = compiled_fw(x)
```

From autograd's point of view, it no longer sees the original internal ops individually. It sees one Python callable. That is not enough for autograd to know:
- what tensors to save,
- which backward function to run,
- how to connect gradient flow to prior tensors.

So AOTAutograd explicitly creates a custom autograd node using:

```python
class CompiledFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ...):
        ...
    @staticmethod
    def backward(ctx, ...):
        ...
```

This is built in:

[`runtime_wrappers.py`](https://github.com/pytorch/pytorch/tree/main/torch/_functorch/_aot_autograd/runtime_wrappers.py)

inside `AOTDispatchAutograd.post_compile(...)`.

### Why this is required
Tracing a backward graph ahead of time is a **compile-time** activity.  
Actually running backward later is a **runtime autograd engine** activity.

Those are different things.

The custom `autograd.Function` is the bridge between:
- compiled fw/bw artifacts,
- PyTorch autograd runtime.

Without this custom node, autograd engine would not know how to enter the compiled backward when the user later calls `loss.backward()`.

---
## ⚙️ What does "integrating with autograd engine" mean
PyTorch autograd engine is the runtime component that coordinates backward execution.

In plain terms, when you call:

```python
loss.backward()
```

the autograd engine:
- starts from `loss`,
- walks backward through the graph,
- figures out which backward nodes must run,
- schedules them when their inputs are ready,
- accumulates gradients into leaf parameters.

For eager ops, this works automatically because PyTorch created autograd nodes as the ops ran.

For compiled regions, AOTAutograd must supply one explicit node that says:
- forward is handled here,
- backward is handled here,
- saved tensors are stored in `ctx`.

That is exactly what `CompiledFunction.apply(...)` gives to the engine.

---
## 🧱 When control reaches `CompiledFunction`
This flow is worth making very explicit.

### Training path flow

```text
torch.compile(...)
  -> backend="hpu_backend"
  -> hpu_backend(graph_module, example_inputs)
  -> aot_autograd(...)(graph_module, example_inputs)
  -> aot_module_simplified(...)
  -> create_aot_dispatcher_function(...)
  -> _create_aot_dispatcher_function(...)
  -> choose_dispatcher(...) -> aot_dispatch_autograd(...)
  -> trace joint graph
  -> partition into fw_module and bw_module
  -> fw_compiler(fw_module, fw_inputs)
  -> bw_compiler(bw_module, bw_inputs)
  -> AOTDispatchAutograd.post_compile(...)
  -> define CompiledFunction(torch.autograd.Function)
  -> return RuntimeWrapper(CompiledFunction.apply)
```

So `CompiledFunction` is not created during initial Dynamo capture. It is created after:
- AOTAutograd has traced,
- partitioned,
- compiled fw and bw.

Then the final callable returned to the user eventually calls `CompiledFunction.apply(...)`.

---
## 📦 What are boxed functions and why are they used
A **boxed function** uses a calling convention like:

```python
fn(args: list[Any]) -> Any
```

instead of:

```python
fn(arg0, arg1, arg2, ...)
```

Why AOTAutograd likes boxed functions:
- it gives a uniform runtime convention,
- wrappers can intercept, reorder, or clear args easily,
- it is useful when inputs need to be freed early,
- some backends naturally work better with one-list calling convention.

If a backend compiler returns a normal callable, AOTAutograd may wrap it using `make_boxed_func`.

Gaudi backend itself often returns boxed wrappers:

[`compilers.py`](https://github.com/HabanaAI/gaudi-pytorch-bridge/tree/1.23.0/python_packages/habana_frameworks/torch/dynamo/compile_backend/compilers.py)

For example, `hpu_compiler_inner(...)` returns:
- either `functorch.compile.make_boxed_func(graph_module.forward)`,
- or a custom wrapper with `wrapper._boxed_call = True`.

This is done because AOTAutograd's runtime wrappers expect and prefer boxed execution.

---
## 📝 Backend compiler contract
When AOTAutograd calls backend compiler hooks, the contract is roughly:

### Forward compiler
Input:
- `graph_module: torch.fx.GraphModule`
- `example_inputs: list[Any]`

Output:
- callable that executes semantics of the forward graph

### Backward compiler
Input:
- `graph_module: torch.fx.GraphModule`
- `example_inputs: list[Any]`

Output:
- callable that executes semantics of the backward graph

### Inference compiler
Input:
- `graph_module: torch.fx.GraphModule`
- `example_inputs: list[Any]`

Output:
- callable that executes semantics of inference graph

The output is **not** a `torch.autograd.Function`. It is just an executable callable. AOTAutograd later wraps those callables into:
- runtime wrapper for inference,
- custom autograd node plus runtime wrapper for training.

In the Gaudi backend these hooks are:
- `hpu_training_compiler_fw`
- `hpu_training_compiler_bw`
- `hpu_inference_compiler`

from:

[`compilers.py`](https://github.com/HabanaAI/gaudi-pytorch-bridge/tree/1.23.0/python_packages/habana_frameworks/torch/dynamo/compile_backend/compilers.py)

---
## 🏗️ What the Gaudi backend actually does
The Gaudi backend is not a trivial passthrough. It does several things around the FX graphs it receives.

At the top level:

[`backends.py`](https://github.com/HabanaAI/gaudi-pytorch-bridge/tree/1.23.0/python_packages/habana_frameworks/torch/dynamo/compile_backend/backends.py)

it builds:
- AOTAutograd frontend,
- HPU forward compiler,
- HPU backward compiler,
- HPU inference compiler,
- HPU partition function,
- HPU decomposition table.

Then in:

[`compilers.py`](https://github.com/HabanaAI/gaudi-pytorch-bridge/tree/1.23.0/python_packages/habana_frameworks/torch/dynamo/compile_backend/compilers.py)

the backend:
- may deep-copy graph for backward safety,
- may fuse attention,
- may run flex attention passes,
- runs pre-placement optimization passes,
- runs pre-partitioner passes,
- runs partitioner passes,
- runs post-partitioner passes,
- returns a boxed executable wrapper.

This means the backend sees already-traced FX graphs and transforms them further before execution.

---
## 🧭 How Gaudi decides eager vs HPU execution
One Gaudi-specific step that is easy to miss is that the backend does **not** assume the whole FX graph can run on HPU.

Instead, during backend passes it classifies nodes into:
- `eager`
- `hpu_cluster`

This happens in:

[`passes.py`](https://github.com/HabanaAI/gaudi-pytorch-bridge/tree/1.23.0/python_packages/habana_frameworks/torch/dynamo/compile_backend/passes.py)

especially in `pass_mark_placement(...)`.

That placement decision is driven by:
- explicit Python-side support and fallback lists,
- dynamic shape restrictions,
- conditional support rules,
- the **shared layer** validation.

The shared layer lives in:

[`shared_layer.py`](https://github.com/HabanaAI/gaudi-pytorch-bridge/tree/1.23.0/python_packages/habana_frameworks/torch/dynamo/compile_backend/shared_layer.py)

Its job is not to compile. Its job is to answer:

```text
Can this FX node safely stay inside a Gaudi compiled graph,
or must it fall back to eager?
```

Conceptually:

```text
FX node
  -> default fallback checks
  -> conditional support checks
  -> shared_layer_validation(...)
  -> placement = eager or hpu_cluster
```

So the shared layer is the support/capability boundary between Python FX lowering and lower-level Gaudi runtime support.

---
## 🔀 What a "mixed compiled callable" means
This is the detail that causes most confusion.

The Gaudi compiler hooks return boxed callables around a transformed `graph_module.forward`. But after placement and clustering, that outer forward graph can still contain both:
- eager ops,
- compiled HPU cluster calls.

So the result is often a **mixed compiled callable**, not a single pure HPU binary.

Conceptually, after Gaudi passes the graph can look like:

```python
def forward(self, x, y):
    a = torch.ops.aten.add.Tensor(x, y)        # eager op or eager fallback
    b = self.hpu_cluster_0(a)                  # compiled HPU cluster
    c = torch.ops.aten.nonzero.default(b)      # eager fallback
    d = self.hpu_cluster_1(b)                  # compiled HPU cluster
    return c, d
```

Those `self.hpu_cluster_*` submodules are produced during:
- partitioning,
- cluster compilation,
- replacement of HPU submodules with callable recipe modules.

That replacement happens in:

[`passes.py`](https://github.com/HabanaAI/gaudi-pytorch-bridge/tree/1.23.0/python_packages/habana_frameworks/torch/dynamo/compile_backend/passes.py)

inside `pass_compile_clusters(...)`, where the backend deletes the original submodule and adds back a callable recipe module under the same target name.

If compiled recipes are enabled, that callable recipe module is typically:

[`recipe_compiler.py`](https://github.com/HabanaAI/gaudi-pytorch-bridge/tree/1.23.0/python_packages/habana_frameworks/torch/dynamo/compile_backend/recipe_compiler.py)

`HabanaGraphModule`, whose runtime behavior is roughly:

```text
first call:
  graph_compile(...) -> recipe_id

every call:
  graph_launch(recipe_id, ...)
```

So one compiled forward callable may execute:
- normal eager ops directly,
- HPU regions by calling `HabanaGraphModule.__call__`.

---
## 🧾 What Gaudi actually returns in inference vs training
This distinction is important.

### Inference path
Gaudi `inference_compiler` returns:
- a boxed callable around the transformed forward FX graph.

Then AOTAutograd wraps that callable with runtime wrappers.

So the final object returned from the AOT inference path is:

```text
RuntimeWrapper(compiled_fw)
```

where `compiled_fw` itself may be mixed:
- eager ops in the outer graph,
- compiled HPU cluster submodules inside it.

There is **no custom `torch.autograd.Function`** in this path.

### Training path
Gaudi:
- `fw_compiler` returns boxed callable around transformed forward graph,
- `bw_compiler` returns boxed callable around transformed backward graph.

Then AOTAutograd builds:

```python
class CompiledFunction(torch.autograd.Function):
    forward(ctx, ...)  -> run compiled_fw
    backward(ctx, ...) -> run compiled_bw
```

and returns:

```text
RuntimeWrapper(CompiledFunction.apply)
```

The important nuance is that `compiled_fw` and `compiled_bw` can both be mixed callables. So in training, the custom autograd node does **not** wrap one monolithic HPU artifact. It wraps:
- a forward callable that may run eager fallback ops plus HPU clusters,
- a backward callable that may run eager backward ops plus HPU clusters.

In other words:

```text
training final callable
  -> CompiledFunction.apply
      -> forward(ctx): call mixed compiled_fw
      -> backward(ctx): call mixed compiled_bw
```

and:

```text
inference final callable
  -> RuntimeWrapper
      -> call mixed compiled_fw directly
```

---
## 🪜 End-to-end Gaudi flow
This is the most concrete version of the control flow:

```text
User model call
  -> Dynamo captures FX graph
  -> hpu_backend(graph_module, example_inputs)
  -> aot_autograd(...)(graph_module, example_inputs)

If inference:
  -> aot_dispatch_base
  -> aot_dispatch_base_graph
  -> hpu_inference_compiler(fw_graph, example_inputs)
  -> hpu_compiler_inner(...)
  -> placement: eager vs hpu_cluster
  -> cluster compile: replace HPU submodules with HabanaGraphModule
  -> return boxed compiled_fw
  -> RuntimeWrapper(compiled_fw)

If training:
  -> aot_dispatch_autograd
  -> aot_dispatch_autograd_graph
  -> hpu_partition(joint_graph) -> fw_graph + bw_graph
  -> hpu_training_compiler_fw(fw_graph, fw_inputs)
  -> hpu_training_compiler_bw(bw_graph, bw_inputs)
  -> both compilers run Gaudi passes, placement and cluster compilation
  -> both return boxed callables
  -> AOTDispatchAutograd.post_compile(...)
  -> build CompiledFunction(torch.autograd.Function)
  -> RuntimeWrapper(CompiledFunction.apply)
```

This is the clean mental model:
- Dynamo captures FX.
- AOTAutograd decides inference or training.
- Gaudi lowers FX into a mixed outer graph plus compiled HPU cluster modules.
- In inference, that callable is run directly.
- In training, that callable is plugged into a custom `torch.autograd.Function`.

---
## 🔬 Why the backward graph is traced ahead of time but still needs runtime insertion
This is the point that causes the most confusion.

### Compile time
AOTAutograd traces the backward graph ahead of time so it can:
- optimize it,
- partition it,
- compile it,
- know its saved tensor needs,
- make guard decisions.

### Runtime
Later, when the user calls `loss.backward()`, autograd engine needs a live node in the current autograd graph to invoke the compiled backward.

So:
- compile-time tracing gives us the backward program,
- runtime autograd node gives PyTorch a place to call that backward program from.

Both are required.

If you only had the compiled backward graph but no custom autograd node, PyTorch autograd would not know that the output tensor of the compiled forward should route its gradients into that compiled backward callable.

---
## 🧰 What runtime jobs AOTAutograd handles
AOTAutograd also owns a lot of runtime behavior after compilation.

Main runtime jobs:
- run compiled fw callable,
- save tensors and symbolic values for backward,
- wrap outputs that alias inputs,
- apply mutation epilogue,
- run compiled backward later,
- optionally lazily compile backward in some cases,
- manage some RNG runtime details.

This runtime handling mostly lives in:

[`runtime_wrappers.py`](https://github.com/pytorch/pytorch/tree/main/torch/_functorch/_aot_autograd/runtime_wrappers.py)

The most important classes/functions here are:
- `RuntimeWrapper`
- `_create_runtime_wrapper`
- `AOTDispatchAutograd.post_compile`
- `CompiledFunction(torch.autograd.Function)`

So AOTAutograd is best thought of as:
- a compile-time graph preparation system,
- plus a runtime compatibility layer.

---
## 🧠 A concrete mental model
If you are new to PyTorch internals, this model is sufficient:

1. `torch.compile` asks TorchDynamo to capture a graph.
2. AOTAutograd decides whether the graph is inference or training.
3. If inference:
   - trace forward graph,
   - compile it,
   - run a runtime wrapper around it.
4. If training:
   - trace joint graph,
   - split it into fw and bw,
   - compile both,
   - create a custom autograd node that knows how to call compiled backward,
   - wrap it in runtime logic for mutations and aliases.
5. Backend like Gaudi receives FX graphs and returns executable callables, often boxed.

That is the core of `torch.compile` as we discussed.

---
## 📚 References
- Local PyTorch source:
  - [aot_autograd.py](https://github.com/pytorch/pytorch/tree/main/torch/_functorch/aot_autograd.py)
  - [dispatch_and_compile_graph.py](https://github.com/pytorch/pytorch/tree/main/torch/_functorch/_aot_autograd/dispatch_and_compile_graph.py)
  - [jit_compile_runtime_wrappers.py](https://github.com/pytorch/pytorch/tree/main/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py)
  - [runtime_wrappers.py](https://github.com/pytorch/pytorch/tree/main/torch/_functorch/_aot_autograd/runtime_wrappers.py)
- Local Gaudi backend source:
  - [backends.py](https://github.com/HabanaAI/gaudi-pytorch-bridge/tree/1.23.0/python_packages/habana_frameworks/torch/dynamo/compile_backend/backends.py)
  - [compilers.py](https://github.com/HabanaAI/gaudi-pytorch-bridge/tree/1.23.0/python_packages/habana_frameworks/torch/dynamo/compile_backend/compilers.py)
  - [partition_fn.py](https://github.com/HabanaAI/gaudi-pytorch-bridge/tree/1.23.0/python_packages/habana_frameworks/torch/dynamo/compile_backend/partition_fn.py)
- PyTorch docs:
  - https://pytorch.org/docs/stable/generated/torch.compile.html
  - https://pytorch.org/docs/stable/fx.html
  - https://pytorch.org/docs/stable/notes/extending.html

---
*"Compilation in PyTorch is not just graph lowering. It is graph capture, semantic preservation, autograd integration and runtime repair, all working together."*
