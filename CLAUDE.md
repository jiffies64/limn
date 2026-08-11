## Commands

```
uv sync                                        # numpy, ml_dtypes, and the dev group
uv run pytest                                  # the whole suite, under a minute
uv run pytest tests/test_attention.py::test_validation   # one test
uv run pytest -k "attention and cuda"          # one slice of it
uv run ruff format limn tests examples         # run on every file you touch
uv run ruff check limn tests examples
uv run pyright                                 # bare: CI type-checks the whole tree, examples too
uv run python -m limn.sdpa                     # the fused-attention reference checks itself
uv run python examples/bench_attention.py      # fused against composed, forward and backward
```

`git config core.hooksPath .githooks` turns on the hooks: commit messages must match
`type: subject` (feat, fix, docs, refactor, test, speedup, chore), and pushes run ruff and the
suite. Merge branches into main by rebasing, never with a merge commit.

The `c` device needs `cc` on PATH; the `cuda` device needs an NVIDIA driver plus NVRTC, from a
toolkit or from `uv sync --extra cuda`. Tests that need either are skipped without them, so a
green run on a machine with neither proves less than it looks.

## Invariants

Tensor methods never compute. They build a DAG of `Node`s over a closed op set, and the graph
runs only when someone asks for bytes. Four rules the code cannot state for itself:

**The numpy device is the reference, not a backend.** Every compiled backend is diffed against
it (`conftest.check`). A new op goes into `device.py` first. Being a reference costs it the
shapes fusion exists for: its matmul builds the whole (m, n, k) intermediate, so diff compiled
kernels at sizes whose product fits, not whose operands do.

**The op set is closed.** `sub`, `div`, `matmul`, `softmax` and the rest are composed in
`tensor.py`. Adding a primitive means touching every backend, so nearly always the answer is to
compose instead.

**`CUSTOM` is the one escape hatch, and it is opt-in in both directions.** It names a kernel a
device supplies whole. A device that registers no kernel for the name never sees the node: the
frontend composes the op from primitives instead. A CUSTOM kernel may write more than one
buffer, and then it is one node per output sharing srcs and params; `jit.plan_of` merges the
siblings into one call, which is the only place in the stack that knows they are siblings.

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
rather than to itself: the fused attention against the composed form and against torch, the
cuda kernels against the numpy ones, the checkpoints against the `safetensors` library. A test
that would pass against a broken kernel is not a test.
