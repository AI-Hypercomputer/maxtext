# Mini-twin: <model>

<!-- The mini-twin is the small configuration every CPU test runs: same layer
types and code paths as the full model, real hidden sizes where CPU memory
allows, real weights sliced from the checkpoint. Random-init toy models have
hidden real bugs before. -->

| Field | Value | Why |
| :---- | :---- | :-- |
| Layer stacks and kept ids | <stack: ids> | <covers every distinct layer type, e.g. each attention or block variant> |
| Scan geometry | <prefix, scanned blocks (>= 2), suffix> | Model-level scan parity geometry |
| Sequence length / batch | <n / n> | <routed models: seq >= 2 * top-k; packed sequences: >= 2 docs + 0-pad gaps> |
| Sizes kept real | <hidden, heads, experts, vocab ...> | <shrunk sizes only shrink; list each> |
| Weights | <real slice via slice_hf_mini.py, or random init with reason> | |
| Slice command | `<exact command>` | Re-creates the mini-twin; hashes in mini_manifest.json |
| Peak memory | <measured RSS> | Measured on the smallest config first |

Overrides that change numerics (block sizes, dtypes, kernel choices) are
deviations and go in the design handoff, not here.
