## Commands

```
uv sync                                        # numpy, ml_dtypes, and the dev group
uv run pytest                                  # the whole suite, under a minute
uv run pytest tests/test_attention.py::test_validation   # one test
uv run pytest -k "attention and cuda"          # one slice of it
uv run ruff format limn tests examples         # run on every file you touch
uv run ruff check limn tests examples
uv run pyright limn tests                      # standard mode, configured in pyproject
uv run python -m limn.sdpa                     # the fused-attention reference checks itself
uv run python examples/bench_attention.py      # fused against composed, forward and backward
```

`git config core.hooksPath .githooks` turns on the hooks: commit messages must match
`type: subject` (feat, fix, docs, refactor, test, speedup, chore), and pushes run ruff and the
suite. The `c` device needs `cc` on PATH; the `cuda` device needs an NVIDIA driver plus NVRTC,
from a toolkit or from `uv sync --extra cuda`. Tests that need either are skipped without them,
so a green run on a machine with neither proves less than it looks.

## Architecture

Tensor methods never compute. They build a DAG of `Node`s over 19 primitive ops, and the graph
runs only when someone asks for bytes. The path from a method call to a kernel crosses six files
and is easier to change once you can see all of it:

    tensor.py    builds Nodes; autograd is Tensors holding a grad_fn, so gradients are lazy graphs too
    ops.py       the closed op set, the Node, and Custom (the CUSTOM arg)
    view.py      shape, strides, offset, mask; reshape/permute/expand/pad/shrink compose into one View
    schedule.py  cuts the DAG into kernels at CUT_OPS; everything elementwise fuses into its consumer
    codegen.py   lowers a kernel to the printable loop-nest IR
    jit.py       plans a graph once per structure, caches on graph_key, owns the assign transaction
    device.py    the numpy interpreter, one numpy call per node
    backend_c.py, cuda_emit.py + backend_cuda.py    render the IR and JIT it

Four invariants carry more weight than any single file:

**The numpy device is the reference, not a backend.** Every compiled backend is diffed against
it (`conftest.check`). A new op goes into `device.py` first. Being a reference costs it the
shapes fusion exists for: its matmul builds the whole (m, n, k) intermediate, so diff compiled
kernels at sizes whose product fits, not whose operands do.

**The op set is closed.** `sub`, `div`, `matmul`, `softmax` and the rest are composed in
`tensor.py`. Adding a primitive means touching every backend, so nearly always the answer is to
compose instead.

**`CUSTOM` is the one escape hatch, and it is opt-in in both directions.** It names a kernel a
device supplies whole. A device that does not register the name never sees the node: the
frontend composes the op from primitives instead, which is why `limn/sdpa.py` plus the fused
block in `cuda_emit.py` can be skipped entirely while reading. A CUSTOM kernel may write more
than one buffer, and then it is one node per output sharing srcs and params; `jit.plan_of`
merges the siblings into one call, which is the only place in the stack that knows they are
siblings.

**`realize()` mutates nodes in place.** A computed sink becomes a `BUFFER` node holding its
result, so a later graph over the same Tensor loads instead of recomputing. Code that holds a
Node across a realize is looking at a different node than it built. The transaction rule sits
next to this: every sink is computed before any `ASSIGN` commits, so an optimizer step reads
pre-assign parameters.

## Style

**Every line of code has to earn its place.** Prefer composing what exists to adding a helper,
and prefer deleting to adding. Before finishing, look for the version of the change that is
shorter without being denser to read.

Prose is the exception: the tree runs at roughly two lines of docstring and comment per line of
code, on purpose, because the framework is meant to be read. Docstrings stay, and they state the
constraint the code cannot, never what the next line does. A comment a reader could delete
without losing anything should not exist.

Match the surrounding code: dense, one-liner-friendly, short concrete sentences.

## Testing

Bug fixes get a regression test that fails without the fix. Kernel work is held to an oracle
rather than to itself: the fused attention is diffed against the composed form, against torch,
and against slicing the hidden keys away, and the cuda kernels against the numpy ones. A test
that would pass against a broken kernel is not a test.
