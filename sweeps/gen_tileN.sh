#!/bin/bash
# Generate the tile-sweep-at-N* manifest. Usage: bash sweeps/gen_tileN.sh <NSTAR>
# Rows = chunked image + moe_n_chunks=NSTAR + each high-value tile override, HIGH prio.
set -uo pipefail
N="${1:?usage: gen_tileN.sh NSTAR}"
CHUNK=gcr.io/cloud-tpu-multipod-dev/integrate_v2:chunked-0617
OUT=sweeps/manifest_tileN.tsv
cat > "$OUT" <<ROWS
# Tile sweep at sweet-spot N=$N (per-chunk M=131072/$N). chunked image + baseline flags.
ds-tN${N}-wim512	$CHUNK	flags_A.txt	moe_n_chunks=$N wi_tile_fwd_batch_seq=512	high
ds-tN${N}-wim1024	$CHUNK	flags_A.txt	moe_n_chunks=$N wi_tile_fwd_batch_seq=1024	high
ds-tN${N}-win2048	$CHUNK	flags_A.txt	moe_n_chunks=$N wi_tile_fwd_mlp_dim=2048	high
ds-tN${N}-wim512n2048	$CHUNK	flags_A.txt	moe_n_chunks=$N wi_tile_fwd_batch_seq=512 wi_tile_fwd_mlp_dim=2048	high
ds-tN${N}-spq1024	$CHUNK	flags_A.txt	moe_n_chunks=$N sa_block_q=1024 sa_block_kv=1024 sa_block_kv_compute=1024	high
ROWS
echo "wrote $OUT for N=$N:"; column -t -s$'\t' "$OUT" 2>/dev/null | sed 's/  */ /g'
echo "fire with: bash sweeps/submit_sweep.sh $OUT"
