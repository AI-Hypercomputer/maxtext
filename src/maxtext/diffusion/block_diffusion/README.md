# Block-Diffusion Training Primitives

Block diffusion generates or trains a sequence as bounded blocks rather than
strictly one token at a time. Within the active block, tokens may use
bidirectional context; completed blocks remain fixed context for later blocks.
For background, see [Block Diffusion: Interpolating Between Autoregressive and
Diffusion Language Models](https://arxiv.org/abs/2503.09573).

This package contains model-independent primitives for preparing a training
batch. Model attention, loss integration, and generation are added separately.

## Terminology

- **Canvas**: The fixed-length token slots being refined. A slot contains a
  clean token or the configured mask token.
- **Block**: A bounded region of the canvas that is corrupted and reconstructed
  together. Earlier blocks are not modified while a later block is active.
- **Corruption**: The forward noising step. `corrupt_tokens` independently masks
  eligible tokens in each logical block and guarantees that every nonempty
  eligible block contributes at least one training target.
- **Validity mask**: Identifies real tokens and excludes padding from corruption
  and supervision.
- **Target loss mask**: Identifies token positions supervised for one sampled
  corruption. It is distinct from the validity and corruption masks.
- **Logit alignment**: Defines which model-output position predicts each clean
  target. `same_position` uses the logit at the target position. `shifted` uses
  the preceding logical position, matching models whose logits predict the next
  token.

## Supported Contracts

The initial primitives intentionally support two explicit combinations:

- `same_position` with `all_masked`: every eligible slot in a block may be
  corrupted, and its same-position logit predicts the clean token.
- `shifted` with `seed_and_mask`: the first slot seeds each block and the
  remaining slots may be corrupted; shifted logits predict clean targets.

Unsupported combinations fail during batch preparation instead of silently
using ambiguous target semantics.
