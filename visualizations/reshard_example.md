# Setup

## Tokens / activations
Assume we have a batch of two sequences of tokens:
- a, b
- A, B

Where e.g. a is the value across the whole model dimension, such that if M=2, a = a0|a1

## Sharding

B/DP means B[atch] sharded on DP. B/DP=0 is shorthand for batch element 0 on DP shard 0

We will, for convenience, in general make array dimensions the same length as the sharding (e.g. 2 sequences often sharded on DP=2, 8 experts, often sharded on EP=8)

# Attention

## MoE mesh

TODO: capture the transition from the MoE mesh

## Attention mesh [DP=2, CP=2, TP=2]

### Attention inputs

```
[B=2/DP=2, S=2/CP=2, M=2/TP=2]
```
```
├── B/DP=0
│   ├── S/CP=0
│   │   ├── M/TP=0: a0
│   │   └── M/TP=1: a1
│   └── S/CP=1
│       ├── M/TP=0: b0
│       └── M/TP=1: b1
└── B/DP=1
    ├── S/CP=0
    │   ├── M/TP=0: A0
    │   └── M/TP=1: A1
    └── S/CP=1
        ├── M/TP=0: B0
        └── M/TP=1: B1
```

### Attention outputs
We apply attention such that 
- a0->a'0
- B1->B'1

# MoE
Total experts=8, TopK=2, C=1, EP=8
(i.e. each expert is on its own shard)

## Attention mesh

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
[B=2/DP=2, S=2/CP=2, M=2/TP=2] @ [B=2/DP, S=2/CP, E=8, C=1] -> [E, B/DP, C=1, M/TP]
```

Now we have:

| | E0 | E1 | E2 | E3 | E4 | E5 | E6 | E7 |
|---|---|---|---|---|---|---|---|---|
| B/DP=0 | a'0\|a'1 | a'0\|a'1 | b'0\|b'1 | — | — | — | — | — |
| B/DP=1 | — | — | A'0\|A'1 | A'0\|A'1 | B'0\|B'1 | B'0\|B'1 | — | — |

With device mappings:

| dp | cp | tp | Device |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 0 | 0 | 1 | 1 |
| 0 | 1 | 0 | 2 |
| 0 | 1 | 1 | 3 |
| 1 | 0 | 0 | 4 |
| 1 | 0 | 1 | 5 |
| 1 | 1 | 0 | 6 |
| 1 | 1 | 1 | 7 |

So, expanding the table above and including device IDs:
```
├── E=0
│   ├── B/DP=0
│   │   ├── M=0: a'0  [Device 0, 2]
│   │   └── M=1: a'1  [Device 1, 3]
│   └── B/DP=1
│       ├── M=0: —    [Device 4, 6]
│       └── M=1: —    [Device 5, 7]
├── E=1
│   ├── B/DP=0
│   │   ├── M=0: a'0  [Device 0, 2]
│   │   └── M=1: a'1  [Device 1, 3]
│   └── ...
└── ...
```

## MoE mesh [EP=8]

Now we reshard to get to:

| | E/EP=0 | E/EP=1 | E/EP=2 | E/EP=3 | E/EP=4 | E/EP=5 | E/EP=6 | E/EP=7 |
|---|---|---|---|---|---|---|---|---|
| B=0 | a'0\|a'1 | a'0\|a'1 | b'0\|b'1 | — | — | — | — | — |
| B=1 | — | — | A'0\|A'1 | A'0\|A'1 | B'0\|B'1 | B'0\|B'1 | — | — |

So:
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
