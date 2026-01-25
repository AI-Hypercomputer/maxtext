# Setup

Assume we have a batch of two sequences of tokens:
- a, b
- A, B

Where e.g. a is the value across the whole model dimension, such that if M=2, a = a0|a1

# Attention

## MoE mesh

TODO: capture the transition from the MoE mesh

## Attention mesh
Attention: DP=2, CP=2, TP=2

### Attention inputs

`[B=2/DP=2, S=2/CP=2, M=2/TP=2]`
```
B/DP=0: 
    S/CP=0: 
        M/TP=0: a0
        M/TP=1: a1
    S/CP=1: 
        M/TP=0: b0
        M/TP=1: b1
B/DP=0: 
    S/CP=0: 
        M/TP=0: A0
        M/TP=1: A1
    S/CP=1: 
        M/TP=0: B0
        M/TP=1: B1
```

### Attention outputs
We apply attention such that a0->a'0, B1->B'1

# MoE
Total experts=8, TopK=2, C=1, EP=8
(i.e. each expert is on its own shard)

## Attention mesh

Assume tokens routed as follows:
a' -> E0, E1
b' -> E0, E2
A' -> E2, E3
B' -> E4, E5

Then b will not go to E0 because of capacity factor:
a' -> E0, E1
b' -> None, E2
A' -> E2, E3
B' -> E4, E5

### Dispatch

[B=2/DP=2, S=2/CP=2, M=2/TP=2] @ [B=2/DP, S=2/CP, E=8, C=1] -> [E, B/DP, C=1, M/TP]

Now we have:
            E0          E1          E2          E3          E4          E5          E6          E7
B/DP=0:     M/TP=a0|a1  M/TP=a0|a1  M/TP=b0|b1   
B/DP=1:                             M/TP=A0|A1  M/TP=A0|A1  M/TP=B0|B1  M/TP=B0|B1

With device mappings:
```
mesh[dp, cp, tp] → device_id   
[0, 0, 0] → 0
[0, 0, 1] → 1
[0, 1, 0] → 2
[0, 1, 1] → 3        
[1, 0, 0] → 4    
[1, 0, 1] → 5
[1, 1, 0] → 6
[1, 1, 1] → 7                                                                                                              
```

So, expanding the table above and including device IDs:
```
E=0:
    B/DP=0:
        M=0: a'0     [Device=0, 2]
        M=1  a'1     [Device=1, 3]
    B/DP=1:
        M=0: None    [Device=4, 6]
        M=1: None    [Device=5, 7]
E=1:
    B/DP=0:
        M=0: a'0     [Device=0, 2]
        M=1: a'1     [Device=1, 3]
    
...
```

## MoE mesh

Now we reshard to get to:

            E/EP=0        E/EP1         E2            E3            E4          E5            E6.           E7
B=0:        M=a'0|a'1     M=a'0|a'1     M=b'0|b'1   
B=1:                                    M=A'0|A'1     M=A'0|A'1     M=B'0|B'1   M=B'0|B'1

So:

E/EP=0:
    B=0:
        M=0: a'0     [Device 0]
        M=1  a'1     [Device 0]
    B=1:
        M=0: None    [Device 0]
        M=1: None    [Device 0]
E/EP=1:
    B=0:
        M=0: a'0     [Device 1]
        M=1: a'1     [Device 1]
