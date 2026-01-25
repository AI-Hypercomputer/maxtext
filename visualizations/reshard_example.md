# Setup

## Tokens / activations
Assume we have a batch of two sequences of tokens:
- a, b
- A, B

Where e.g. a is the value across the whole model dimension, such that if M=2, a = a0|a1

We will decorate these values to indicate then transformations from attention (e.g. a') and MLP (e.g. a" and a^)

## Indexing and sharding

We will, for convenience, in general make array dimensions the same length as the sharding (e.g. 2 sequences often sharded on DP=2, 8 experts, often sharded on EP=8)

B=0 means batch element (sequence) 0

B/DP means B[atch] sharded on DP. 

B=0/DP=0 means batch element 0 on DP shard 0.

B/DP=0 is shorthand for the above.

# Flow

## Inputs

<span style="color:yellow">▶ Attention mesh [DP=2, CP=2, TP=2] ◀</span>

Token embeddings enter the model as `[B/DP=2, S/CP=2, M/TP=2]`.

```python
attention_mesh = Mesh(devices, axis_names=('dp', 'cp', 'tp'))
sharding = NamedSharding(attention_mesh, P('dp', 'cp', 'tp'))

# x has shape [B=2, S=2, M=2]
x = jax.device_put(x, sharding)
```

## Attention

<span style="color:yellow">▶ Attention mesh [DP=2, CP=2, TP=2] ◀</span>

### Attention inputs

Receives `[B/DP=2, S/CP=2, M/TP=2]` from inputs (first layer) or MoE combine (subsequent layers):

| | S/CP=0 | S/CP=1 |
|---|---|---|
| B/DP=0 | M/TP=2: {a0\|a1} | M/TP=2: {b0\|b1} |
| B/DP=1 | M/TP=2: {A0\|A1} | M/TP=2: {B0\|B1} |

Given default device mappings for attention mesh:

| DP | CP | TP | Device |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 0 | 0 | 1 | 1 |
| 0 | 1 | 0 | 2 |
| 0 | 1 | 1 | 3 |
| 1 | 0 | 0 | 4 |
| 1 | 0 | 1 | 5 |
| 1 | 1 | 0 | 6 |
| 1 | 1 | 1 | 7 |


We have:
```
├── B/DP=0
│   ├── S/CP=0
│   │   ├── M/TP=0: a0  [Device 0]
│   │   └── M/TP=1: a1  [Device 1]
│   └── S/CP=1
│       ├── M/TP=0: b0  [Device 2]
│       └── M/TP=1: b1  [Device 3]
└── B/DP=1
    ├── S/CP=0
    │   ├── M/TP=0: A0  [Device 4]
    │   └── M/TP=1: A1  [Device 5]
    └── S/CP=1
        ├── M/TP=0: B0  [Device 6]
        └── M/TP=1: B1  [Device 7]
```

### Attention outputs
We apply attention such that, e.g.
- a0->a'0
- B1->B'1

| | S/CP=0 | S/CP=1 |
|---|---|---|
| B/DP=0 | M/TP=2: {a'0\|a'1} | M/TP=2: {b'0\|b'1} |
| B/DP=1 | M/TP=2: {A'0\|A'1} | M/TP=2: {B'0\|B'1} |

```
├── B/DP=0
│   ├── S/CP=0
│   │   ├── M/TP=0: a'0  [Device 0]
│   │   └── M/TP=1: a'1  [Device 1]
│   └── S/CP=1
│       ├── M/TP=0: b'0  [Device 2]
│       └── M/TP=1: b'1  [Device 3]
└── B/DP=1
    ├── S/CP=0
    │   ├── M/TP=0: A'0  [Device 4]
    │   └── M/TP=1: A'1  [Device 5]
    └── S/CP=1
        ├── M/TP=0: B'0  [Device 6]
        └── M/TP=1: B'1  [Device 7]
```

## MoE
Total experts=8, TopK=2, C=1, EP=8
(i.e. each expert is on its own shard)

<span style="color:yellow">▶ Attention mesh [DP=2, CP=2, TP=2] ◀</span>

### Expert selection

Assume tokens routed as follows:

| Token | Expert 1 | Expert 2 |
|---|---|---|
| a' | E0 | E1 |
| b' | E0 | E2 |
| A' | E2 | E3 |
| B' | E4 | E5 |

Then b' will not go to E0 because of capacity factor:

| Token | Expert 1 | Expert 2 |
|---|---|---|
| a' | E0 | E1 |
| b' | *(dropped)* | E2 |
| A' | E2 | E3 |
| B' | E4 | E5 |

### Dispatch

```
[B=2/DP=2, S=2/CP=2, M=2/TP=2] @ [B=2/DP, S=2/CP, E=8, C=1] -> [E, B/DP=2, C=1, M/TP=2]
```

Now we have:

| | E0 | E1 | E2 | E3 | E4 | E5 | E6 | E7 |
|---|---|---|---|---|---|---|---|---|
| B/DP=0 | M/TP=2: {a'0\|a'1} | M/TP=2: {a'0\|a'1} | M/TP=2: {b'0\|b'1} | — | — | — | — | — |
| B/DP=1 | — | — | M/TP=2: {A'0\|A'1} | M/TP=2: {A'0\|A'1} | M/TP=2: {B'0\|B'1} | M/TP=2: {B'0\|B'1} | — | — |

Expanding with device IDs (see device mappings of attention mesh, above):
```
├── E=0
│   ├── B/DP=0
│   │   ├── M/TP=0: a'0  [Device 0, 2]
│   │   └── M/TP=1: a'1  [Device 1, 3]
│   └── B/DP=1
│       ├── M/TP=0: —    [Device 4, 6]
│       └── M/TP=1: —    [Device 5, 7]
├── E=1
│   ├── B/DP=0
│   │   ├── M/TP=0: a'0  [Device 0, 2]
│   │   └── M/TP=1: a'1  [Device 1, 3]
│   └── ...
└── ...
```

<span style="color:yellow">▶ **MoE mesh [EP=8]** ◀</span>

Now we reshard to get to:

| | E/EP=0 | E/EP=1 | E/EP=2 | E/EP=3 | E/EP=4 | E/EP=5 | E/EP=6 | E/EP=7 |
|---|---|---|---|---|---|---|---|---|
| B=0 | M: {a'0\|a'1} | M: {a'0\|a'1} | M: {b'0\|b'1} | — | — | — | — | — |
| B=1 | — | — | M: {A'0\|A'1} | M: {A'0\|A'1} | M: {B'0\|B'1} | M: {B'0\|B'1} | — | — |

Using code such as the following:

```python
moe_mesh = Mesh(devices, axis_names=('ep',))
expert_inputs = jax.lax.with_sharding_constraint(
    expert_inputs, NamedSharding(moe_mesh, P('ep', None, None, None))
)
```

We have default device mappings for MoE mesh as follows:

| EP | Device |
|---|---|
| 0 | 0 |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
| 4 | 4 |
| 5 | 5 |
| 6 | 6 |
| 7 | 7 |

Giving the following device mappings:
```
├── E/EP=0
│   ├── B=0
│   │   ├── M=0: a'0  [Device 0]
│   │   └── M=1: a'1  [Device 0]
│   └── B=1
│       ├── M=0: —    [Device 0]
│       └── M=1: —    [Device 0]
├── E/EP=1
│   ├── B=0
│   │   ├── M=0: a'0  [Device 1]
│   │   └── M=1: a'1  [Device 1]
│   └── ...
└── ...
```

### Expert processing

Each expert processes its assigned tokens. Using a", b", A", B" to denote post-expert values:

| | E/EP=0 | E/EP=1 | E/EP=2 | E/EP=3 | E/EP=4 | E/EP=5 | E/EP=6 | E/EP=7 |
|---|---|---|---|---|---|---|---|---|
| B=0 | M: {a"0\|a"1} | M: {a"0\|a"1} | M: {b"0\|b"1} | — | — | — | — | — |
| B=1 | — | — | M: {A"0\|A"1} | M: {A"0\|A"1} | M: {B"0\|B"1} | M: {B"0\|B"1} | — | — |

So:
```
├── E/EP=0
│   ├── B=0
│   │   ├── M=0: a"0  [Device 0]
│   │   └── M=1: a"1  [Device 0]
│   └── B=1
│       ├── M=0: —    [Device 0]
│       └── M=1: —    [Device 0]
├── E/EP=1
│   ├── B=0
│   │   ├── M=0: a"0  [Device 1]
│   │   └── M=1: a"1  [Device 1]
│   └── ...
└── ...
```

<span style="color:yellow">▶ **Attention mesh [DP=2, CP=2, TP=2]** ◀</span>

### Combine

First we reshard to get to `[E=8, B/DP=2, C=1, M/TP=2]`:

| | E0 | E1 | E2 | E3 | E4 | E5 | E6 | E7 |
|---|---|---|---|---|---|---|---|---|
| B/DP=0 | M/TP=2: {a"0\|a"1} | M/TP=2: {a"0\|a"1} | M/TP=2: {b"0\|b"1} | — | — | — | — | — |
| B/DP=1 | — | — | M/TP=2: {A"0\|A"1} | M/TP=2: {A"0\|A"1} | M/TP=2: {B"0\|B"1} | M/TP=2: {B"0\|B"1} | — | — |

Using code like the following:

```python
expert_outputs = jax.lax.with_sharding_constraint(
    expert_outputs, NamedSharding(attention_mesh, P(None, 'dp', None, 'tp'))
)
```

Note that we cannot yet make use of the CP axis.

Expanding with device IDs we now have (see device mappings above):
```
├── E=0
│   ├── B/DP=0
│   │   ├── M/TP=0: a"0  [Device 0, 2]
│   │   └── M/TP=1: a"1  [Device 1, 3]
│   └── B/DP=1
│       ├── M/TP=0: —    [Device 4, 6]
│       └── M/TP=1: —    [Device 5, 7]
├── E=1
│   ├── B/DP=0
│   │   ├── M/TP=0: a"0  [Device 0, 2]
│   │   └── M/TP=1: a"1  [Device 1, 3]
│   └── ...
└── ...
```

Now we execute the combine as follows:

```
[E=8, B/DP=2, C=1, M/TP=2] @ [B/DP=2, S, E, C=1] -> [B/DP=2, S/CP=2, M/TP=2]
```

We add sharding on CP with another sharding constraint as follows:

```python
output = jax.lax.with_sharding_constraint(
    output, NamedSharding(attention_mesh, P('dp', 'cp', 'tp'))
)
```

The combine mask contains routing weights (softmax of gate logits for selected experts). For token a routed to E0 and E1 with respective weights w0=0.6 and w1=0.4:

```
a^ = 0.6 · a"(from E0) + 0.4 · a"(from E1)
```

After combine, we have `[B/DP=2, S/CP=2, M/TP=2]`, ready for the next attention layer:

| | S/CP=0 | S/CP=1 |
|---|---|---|
| B/DP=0 | M/TP=2: {a^0\|a^1} | M/TP=2: {b^0\|b^1} |
| B/DP=1 | M/TP=2: {A^0\|A^1} | M/TP=2: {B^0\|B^1} |

```
├── B/DP=0
│   ├── S/CP=0
│   │   ├── M/TP=0: a^0  [Device 0]
│   │   └── M/TP=1: a^1  [Device 1]
│   └── S/CP=1
│       ├── M/TP=0: b^0  [Device 2]
│       └── M/TP=1: b^1  [Device 3]
└── B/DP=1
    ├── S/CP=0
    │   ├── M/TP=0: A^0  [Device 4]
    │   └── M/TP=1: A^1  [Device 5]
    └── S/CP=1
        ├── M/TP=0: B^0  [Device 6]
        └── M/TP=1: B^1  [Device 7]
```
