# limn

*limn (verb): to outline in clear sharp detail; to delineate.*

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)
![Runtime dependency: numpy](https://img.shields.io/badge/runtime%20dependency-numpy-013243?logo=numpy&logoColor=white)
![Backends: numpy and c](https://img.shields.io/badge/backends-numpy%20%C2%B7%20c-555555)

A deep learning framework built to be read. The whole stack is here: lazy tensors over a
closed set of 19 primitive ops, reverse-mode autograd, a scheduler that fuses the graph into
kernels, an IR you can print, and a C backend that JIT-compiles it, in about 2,000 lines of
Python with numpy as the only runtime dependency. In spirit it sits between micrograd and
tinygrad: small enough to read in one sitting, real enough that a matmul comes out the other
end as one fused loop nest with the stride-1 dim innermost.

```python
from limn import Tensor

x = Tensor.randn((4, 3), requires_grad=True)
loss = ((x @ Tensor.randn((3, 2))).relu() ** 2).sum()
loss.backward()          # gradients are lazy graphs too
print(x.grad.numpy())    # nothing computes until here
```

## Getting started

```
git clone https://github.com/jiffies64/limn.git
cd limn
uv sync
uv run pytest                          # the whole suite, about ten seconds
uv run python examples/train_mlp.py    # AdamW on a toy regression; loss must drop 20x
```

`uv sync` installs numpy plus the dev group (pytest, ruff, CPU-only torch). The `c` device
also needs a C compiler on `PATH` as `cc`.

## Architecture

Tensor methods never compute. They build a DAG over the primitive ops, and the graph runs
only when someone asks for bytes (`.numpy()`, `.item()`, `.realize()`). Autograd lives at the
same layer: `backward()` walks the recorded graph in reverse, and the gradients it builds are
lazy graphs like everything else.

```mermaid
flowchart LR
    T["Tensor API + autograd<br>tensor.py"] --> G["op graph, 19 primitives<br>ops.py, view.py"]
    G --> N["numpy interpreter<br>device.py"]
    G --> S["scheduler, fused kernels<br>schedule.py"]
    S --> I["loop-nest IR<br>codegen.py"]
    I --> C["C JIT<br>backend_c.py"]
```

Three rules hold it together:

- **The op set is closed.** `sub`, `div`, `matmul`, `softmax`, `relu`, comparisons: all
  composed in `tensor.py` from the 19 primitives, so a backend implements those and gets
  everything else for free.
- **The numpy device is the permanent reference.** It interprets the graph one numpy call per
  node, no fusion, no cleverness. Every compiled backend is diffed against it.
- **Layout is arithmetic, not data.** A `View` is (shape, strides, offset, mask). `reshape`,
  `permute`, `expand`, `pad`, and `shrink` compose into a single `View`, cost nothing when
  built, and lower to index arithmetic inside the kernels.

### The op set

| group | ops |
|---|---|
| sources | `BUFFER`, `CONST` |
| movement | `VIEW` |
| elementwise unary | `NEG`, `EXP`, `LOG`, `SQRT`, `RECIP`, `CAST` |
| elementwise binary | `ADD`, `MUL`, `CMPLT` |
| elementwise ternary | `WHERE` |
| reduce | `SUM`, `MAX` |
| indexed | `GATHER`, `SCATTER` |
| barriers | `CONTIGUOUS`, `ASSIGN` |

Two dtypes: `float32` and `int32`.

## Reading the schedule

`limn.schedule` cuts the DAG into kernels: elementwise work fuses into whatever consumes it,
and a cut falls wherever a value has to exist in memory (a reduce, a copy, an assign,
anything a view addresses). `limn.codegen` lowers each kernel into a loop nest, turning views
into index arithmetic. The result prints, and it is exactly what the compiled backends run:

```python
from limn import Tensor
from limn.codegen import ir

print(ir(Tensor.randn((4, 5)) @ Tensor.randn((5, 3))))
```

```
k0  SUM  loop[4, 3, 5]
    in0 = buf0  float32[4, 5]
    in1 = buf1  float32[5, 3]
    out = buf2  float32[4, 3, 1]
  LOOP    i0 < 4
    LOOP    i1 < 3
      CONST   v0 = 0.0 : float32
      STORE   out[i0*3 + i1] = v0
    ENDLOOP i1
  ENDLOOP i0
  LOOP    i0 < 4
    LOOP    r2 < 5
      LOOP    i1 < 3
        LOAD    v1 = in0[i0*5 + r2] : float32
        LOAD    v2 = in1[i1 + r2*3] : float32
        ARITH   v3 = MUL v1, v2 : float32
        ACCUM   out[i0*3 + i1] = ADD out[i0*3 + i1], v3
      ENDLOOP i1
    ENDLOOP r2
  ENDLOOP i0

sink 0 = buf2  float32[4, 3]
```

Two things the dump shows:

- **A whole matmul is one kernel.** Broadcast, multiply, and reduce fuse into a single nest,
  and dropping the reduce's kept dim afterwards moves no data: `sink 0`, the (4, 3) answer,
  aliases the (4, 3, 1) buffer instead of copying it.
- **Loop order is chosen, not inherited.** The innermost loop decides how the nest walks
  memory, so `loop_order` moves the dim that is stride-1 in the most buffers there: `i1`, not
  the reduce axis `r2`. A reduce axis with loops inside it cannot keep its running totals in
  a register, so this nest fills the output with the reduce identity first and folds into it
  with `ACCUM`.

## Backends

`set_device` picks the executor. Buffers are host bytes on every current device, so tensors
created before a switch keep working after it.

```python
import numpy as np
from limn import Tensor, set_device

set_device("c")            # "numpy" is the default, and stays the reference
x = Tensor(np.random.rand(64, 32).astype(np.float32))
print((x @ x.transpose()).sum().item())
```

| device | pipeline | requires |
|---|---|---|
| `numpy` | interprets the op graph, one numpy call per node | nothing |
| `c` | schedule → loop-nest IR → C source → `cc -O3 -march=native` → ctypes | a C compiler |

The C device caches twice: shared libraries by source hash, execution plans by graph
structure. A training loop at fixed shapes schedules, emits, and compiles on the first step;
every step after goes straight to the compiled kernels. Selecting `"c"` with no compiler
installed fails at `set_device`, not later inside a subprocess.

### Adding one

The IR is the contract: twelve opcodes (loops, loads, arithmetic, accumulators, stores) with
all index arithmetic made explicit. A backend is one rendering of that instruction stream
plus buffer management, which is exactly what `backend_c.py` is for C.

## nn and optim

`limn.nn` holds `Linear`, `LayerNorm`, `Embedding`, and a `parameters()` walker that collects
every trainable tensor reachable from a module's attributes. `limn.optim` holds `SGD` (with
momentum) and `AdamW`. An optimizer step is a batch of `ASSIGN` graphs committed in one
`realize()`, so every update expression reads pre-step values and update order cannot matter.
Semantics match `torch.nn` and `torch.optim` down to weight layouts, LayerNorm's biased
variance, and AdamW's decoupled weight decay; the tests hold them to it step for step.

## Correctness

Every layer answers to an oracle above it:

| layer | oracle | how |
|---|---|---|
| ops + autograd | PyTorch (CPU, test-only) | a seeded fuzzer builds 300 random DAGs (movement, broadcasting, reduces, matmul), runs them forward and backward in both frameworks, and requires agreement to 1e-4; failures print a reproducer |
| scheduler + codegen | the numpy device | tests interpret the lowered IR instruction by instruction and diff the numbers, so the printed nest means what it says (strides, masks, reduce identities) |
| `c` backend | the numpy device | a shared graph corpus runs on both devices and is diffed at 1e-5 |

## Layout

```
limn/
  ops.py         the 19 primitives and the Node DAG
  view.py        layout algebra: shape, strides, offset, mask
  tensor.py      user API: broadcasting, composed ops, autograd
  device.py      the device protocol and the numpy reference interpreter
  schedule.py    cuts the graph into fused kernels
  codegen.py     lowers kernels to the printable loop-nest IR
  backend_c.py   renders the IR as C, compiles it, calls it through ctypes
  nn.py          Linear, LayerNorm, Embedding
  optim.py       SGD, AdamW
tests/           one file per layer, a 300-case autograd fuzzer, an IR interpreter
examples/        train_mlp.py, a regression that must reach 5% of its starting loss
```
