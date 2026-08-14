Fix for Gemma 4 E2B PLE Dense Layers Checkpointing
To properly enable checkpointing and rematerialization for the per_layer_input_gate and per_layer_projection outputs inside the Gemma 4 Small models, the new checkpoint string tags must be integrated cleanly across MaxText's configuration definitions, model code, and rematerialization helpers.

Follow these explicit instructions to complete the fix.

1. Apply checkpoint_name Wraps in gemma4_small.py
Wrap the output of both the gate layer and the proj layer using jax.ad_checkpoint.checkpoint_name. MaxText canonically wraps the dense projections before any following activations or normalizations are applied (e.g. before gelu or RMSNorm).

File: src/maxtext/models/gemma4_small.py Context: Inside the __call__ method of the Gemma4SmallDecoderLayer class, where the PLE condition if self.per_layer_input_gate is not None and per_layer_input is not None: executes.

Code Snippet Modification:

python

    if self.per_layer_input_gate is not None and per_layer_input is not None:
      residual = h
      gate = self.per_layer_input_gate(h)
      # [NEW] Wrap the gate output
      gate = checkpoint_name(gate, "per_layer_input_gate")
      gate = jax.nn.gelu(gate.astype(jnp.float32), approximate=True).astype(cfg.dtype)
      gated = gate * per_layer_input.astype(cfg.dtype)
      proj = self.per_layer_projection(gated)
      # [NEW] Wrap the proj output
      proj = checkpoint_name(proj, "per_layer_projection")
      proj = self.post_per_layer_input_norm(proj)
      proj = nn.with_logical_constraint(proj, self.activation_axis_names)
      h = residual + proj
2. Append Tags to Predefined REMAT Configurations
Append the new tags to the array mappings of predefined policies in both the NNX and Linen decoder files so they don't break when basic presets are active.

A. Update minimal_policy (NNX & Linen)
The minimal_policy explicitly specifies what to cache string-by-string. Both PLE dense projections must be saved.

Files: src/maxtext/layers/nnx_decoders.py & src/maxtext/layers/decoders.py Context: In minimal_policy(self, ...)

Code Snippet Modification:

python

  def minimal_policy(self, with_context=False, with_quantization=False):
    """Helper for creating minimal checkpoint policies."""
    names = [
        # ... existing names ...
        "mlpwi",
        "mlpwo",
        # [NEW] Add the Gemma 4 E2B PLE gate names
        "per_layer_input_gate", 
        "per_layer_projection",
    ]
B. Update save_dot_except_mlpwi (NNX & Linen)
This policy intentionally recomputes the input of the FF block (mlpwi), but caches the overarching output (mlpwo). You must add the output mapping counterpart for PLE, but explicitly exclude the input one to maintain strict mathematical intent.

Files: src/maxtext/layers/nnx_decoders.py & src/maxtext/layers/decoders.py Context: In the elif cfg.remat_policy == "save_dot_except_mlpwi": conditional block.

Code Snippet Modification:

python

      elif cfg.remat_policy == "save_dot_except_mlpwi":
        policy = jax.checkpoint_policies.save_only_these_names(
            # ... qkv projections
            "out_proj",
            "mlpwo",
            # [NEW] Add the PLE counterpart to mlpwo!
            "per_layer_projection",
        )
(Leave policies like save_qkv_proj or save_dot_except_mlp untouched, as omitting both PLE tensors aligns with their caching goals).

3. Append Tags to minimal_offloaded in maxtext_utils.py
When using offloading schemes for rematerialization, MaxText delegates the name list to a utility function. It must similarly be integrated into the predefined minimal_offloaded set.

File: src/maxtext/utils/maxtext_utils.py Context: Inside the get_save_and_offload_names function, under the condition if config.remat_policy == "minimal_offloaded":.

Code Snippet Modification:

python

  if config.remat_policy == "minimal_offloaded":
    return [], [
        # ... existing names ...
        "mlpwi",
        "mlpwo",
        # [NEW] Add the Gemma 4 E2B PLE gate names
        "per_layer_input_gate",
        "per_layer_projection",
    ]
4. Declare Custom Policy Fields in types.py (CRITICAL)
In MaxText, users can define remat_policy: 'custom' and set individual properties to device, remat, or offload (e.g. out_proj=device). If you add new checkpoint variables, you must expose them as RematLocation attributes in the Pydantic type model representing configurations, and populate them onto tensors_on_device / tensors_to_offload. Without this step, users can't override the offloading options and defaults will break custom policies.

File: src/maxtext/configs/types.py

A. Add to RematAndOffload class schema Context: In the RematAndOffload class definitions.

Code Snippet Modification:

python

  mla_kv: RematLocation = Field(
      RematLocation.REMAT,
      description="Remat policy for the mla's key and value projection.",
  )
  # [NEW] Fields for Gemma 4 PLE gates
  per_layer_input_gate: RematLocation = Field(
      RematLocation.REMAT,
      description="Remat policy for the Gemma 4 E2B PLE gate.",
  )
  per_layer_projection: RematLocation = Field(
      RematLocation.REMAT,
      description="Remat policy for the Gemma 4 E2B PLE projection.",
  )
B. Add to internal tensors array during class initializations (Occurs in two places) Context: There are two methods (e.g. model_post_init and __init__) that formulate the tensors array under the if self.remat_policy == "custom": check. Add them at the end of both tensors = [...] arrays!

Code Snippet Modification:

python

    if self.remat_policy == "custom":
      tensors = [
          "decoder_layer_input",
          # ... existing names ...
          "attention_out",
          "out_proj",
          # [NEW] Push inside both `tensors` lists
          "per_layer_input_gate",
          "per_layer_projection",
      ]
5. Expose as Defaults in base.yml
Since the checkpoint policies map dynamically to values specified in the global run configuration YAML, the fields must be declared with their baseline configuration values so the system recognizes them uniformly.

File: src/maxtext/configs/base.yml Context: Under the # Choose 'remat_policy' section configuring parameters like query_proj.

Code Snippet Modification:

yaml

mla_q: 'remat'
mla_kv: 'remat'
attention_out: 'remat'
engram: 'remat'
# [NEW] Default baseline behavior
per_layer_input_gate: 'remat'
per_layer_projection: 'remat'