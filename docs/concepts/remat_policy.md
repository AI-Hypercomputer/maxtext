# Remat Policy and Host Offloading

For large-scale model training, accelerator memory is a limited resource and we
often make trade-offs such as activation re-materialization to trade off compute
cycles for accelerator memory resources. Host offload is another technique we
recently introduced in the XLA compiler to leverage host DRAM to offload
activations computed during the forward pass and reuse them during the backward
pass for gradient computation; this saves activation recomputation cycles.

Maxtext provides a parameter called `remat_policy`. This parameter allows
offloading activation memory to host, HBM or recomputing on backward pass.

Activations in the forward pass are also needed in the backward pass. There are
three options for where in memory these activations are accessible for the
backward pass:

1. In HBM (MaxText remat policy "minimal")
2. On host (MaxText remat policy "minimal_offloaded")
3. Activations are re-computed during the backward pass (MaxText remat policy "full")

We can choose different remat policies for different activations (e.g. the FF
activations versus the QKV proj activations), which allows us to optimize memory
usage vs compute trade-offs: Generally we want to use all of our HBM. Both host
offloading (option 2) and re-computing (Aka remat, option 3), use as little HBM
as possible - which is faster depends on model sizes, device compute speed and
host to device speed.
