<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>XLA Temporary Compile Memory (Tmem) in MaxText/JAX: A Deep Dive</title>
<style>
  :root {
    --bg: #0d1117;
    --fg: #c9d1d9;
    --accent: #58a6ff;
    --accent2: #f0883e;
    --accent3: #7ee787;
    --code-bg: #161b22;
    --border: #30363d;
    --subtle: #8b949e;
    --red: #f85149;
    --purple: #d2a8ff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--fg);
    line-height: 1.7;
    max-width: 960px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
  }
  h1 { font-size: 2rem; color: var(--accent); margin-bottom: 0.5rem; line-height: 1.2; }
  h2 { font-size: 1.5rem; color: var(--accent2); margin-top: 2.5rem; margin-bottom: 1rem;
       border-bottom: 1px solid var(--border); padding-bottom: 0.4rem; }
  h3 { font-size: 1.2rem; color: var(--accent3); margin-top: 1.8rem; margin-bottom: 0.6rem; }
  h4 { font-size: 1.05rem; color: var(--purple); margin-top: 1.4rem; margin-bottom: 0.4rem; }
  p { margin-bottom: 1rem; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  code {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    background: var(--code-bg);
    padding: 0.15em 0.4em;
    border-radius: 4px;
    font-size: 0.88em;
  }
  pre {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    overflow-x: auto;
    margin-bottom: 1.2rem;
    font-size: 0.85em;
    line-height: 1.5;
  }
  pre code { background: none; padding: 0; }
  .subtitle { color: var(--subtle); font-size: 1rem; margin-bottom: 2rem; }
  .callout {
    border-left: 3px solid var(--accent);
    background: rgba(88,166,255,0.06);
    padding: 0.8rem 1.2rem;
    margin: 1.2rem 0;
    border-radius: 0 6px 6px 0;
  }
  .callout-warn {
    border-left-color: var(--accent2);
    background: rgba(240,136,62,0.06);
  }
  .callout-ok {
    border-left-color: var(--accent3);
    background: rgba(126,231,135,0.06);
  }
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 1.2rem 0;
    font-size: 0.92em;
  }
  th, td {
    padding: 0.6rem 0.8rem;
    border: 1px solid var(--border);
    text-align: left;
  }
  th { background: var(--code-bg); color: var(--accent); font-weight: 600; }
  .diagram-container {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1.2rem;
    margin: 1.5rem 0;
    overflow-x: auto;
  }
  .metric-box {
    display: inline-block;
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    margin: 0.3rem;
    text-align: center;
  }
  .metric-box .value { font-size: 1.8rem; font-weight: 700; color: var(--accent); }
  .metric-box .label { font-size: 0.8rem; color: var(--subtle); }
  .metric-box.red .value { color: var(--red); }
  .metric-box.green .value { color: var(--accent3); }
  .toc { background: var(--code-bg); border-radius: 8px; padding: 1rem 1.5rem; margin: 1.5rem 0; }
  .toc ol { padding-left: 1.5rem; }
  .toc li { margin: 0.3rem 0; }
  .diff-add { color: var(--accent3); }
  .diff-del { color: var(--red); }
  svg text { font-family: 'Inter', sans-serif; }
</style>
</head>
<body>

<h1>XLA Temporary Compile Memory (Tmem) in MaxText/JAX</h1>
<p class="subtitle">
  A deep technical analysis using the DeepSeek-proxy <code>ds-proxy-se2-e256-h4096</code> config as a running example.<br>
  Audience: JAX/XLA practitioners working with pipeline parallelism, SPMD, and HLO optimization.
</p>

<div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 2rem;">
  <div class="metric-box red">
    <div class="value">29.9 GB</div>
    <div class="label">tmem BEFORE</div>
  </div>
  <div class="metric-box green">
    <div class="value">20.4 GB</div>
    <div class="label">tmem AFTER</div>
  </div>
  <div class="metric-box">
    <div class="value">32%</div>
    <div class="label">reduction</div>
  </div>
</div>

<div class="toc">
<strong>Contents</strong>
<ol>
  <li><a href="#s1">The Model Configuration: ds-proxy-se2-e256-h4096</a></li>
  <li><a href="#s2">What is Temp Memory (tmem) in XLA?</a></li>
  <li><a href="#s3">The Compilation Pipeline: Python to Buffer Assignment</a></li>
  <li><a href="#s4">Remat Policy and Checkpoint Boundaries</a></li>
  <li><a href="#s5">Pipeline Parallelism and Tmem: The Core Problem</a></li>
  <li><a href="#s6">Optimization 1: Replacing ppermute/shard_map with Local Ops</a></li>
  <li><a href="#s7">Optimization 2: Removing Unnecessary Sharding Constraints</a></li>
  <li><a href="#s8">Optimization 3: dynamic_slice_in_dim vs. gather</a></li>
  <li><a href="#s9">Optimization 4: pipeline_save_decoder_layer_input</a></li>
  <li><a href="#s10">Optimization 5: float32_weight_sum and MoE Accumulation</a></li>
  <li><a href="#s11">Optimization 6: skip_trivial_specs</a></li>
  <li><a href="#s12">Putting It All Together: The Savings Breakdown</a></li>
  <li><a href="#s13">Memory Layout Diagram</a></li>
</ol>
</div>

<!-- ================================================================== -->
<h2 id="s1">1. The Model Configuration: ds-proxy-se2-e256-h4096</h2>

<p>
The running example throughout this article is a DeepSeek-proxy model that exercises pipeline parallelism with MoE (Mixture of Experts). This is not a full DeepSeek model; it uses the Mixtral decoder block with global attention instead of MLA, serving as a cheaper proxy for tmem experiments. The key parameters:
</p>

<pre><code># Model architecture
base_emb_dim: 4096
base_num_query_heads: 32
base_num_kv_heads: 8
base_num_decoder_layers: 64
head_dim: 128
mlp_activations: ["silu", "linear"]
vocab_size: 102400

# MoE configuration
decoder_block: mixtral
num_experts: 16
num_experts_per_tok: 4
shared_experts: 2
base_mlp_dim: 2048
base_moe_mlp_dim: 2048

# Pipeline parallelism
ici_pipeline_parallelism: 8
num_layers_per_pipeline_stage: 4
num_pipeline_microbatches: 64
per_device_batch_size: 24
max_target_length: 4096

# Rematerialization
remat_policy: full
pipeline_save_decoder_layer_input: false</code></pre>

<p>From these parameters, the pipeline schedule is derived:</p>

<ul>
  <li><code>num_stages = ici_pipeline_parallelism = 8</code></li>
  <li><code>pipeline_parallel_layers = num_decoder_layers = 64</code> (since <code>decoder_block</code> is not <code>deepseek</code>)</li>
  <li><code>num_pipeline_repeats = 64 / (8 * 4) = 2</code> (circular pipeline with 2 repeats)</li>
  <li><code>microbatches_per_stage = 64 / 8 = 8</code></li>
  <li><code>pipeline_fsdp_ag_per_repeat = false</code> &rarr; uses the <code>Pipeline</code> class (not <code>CircularPipeline</code>)</li>
  <li><code>micro_batch_size_to_train_on = num_devices * per_device_batch_size = 8 * 24 = 192</code> (on a single 8-GPU node, all 8 devices are pipeline stages)</li>
  <li><code>pipeline_microbatch_size = micro_batch_size_to_train_on / num_pipeline_microbatches = 192 / 64 = 3</code></li>
</ul>

<p>
The <code>Pipeline</code> class (as opposed to <code>CircularPipeline</code>) is instantiated because <code>pipeline_fsdp_ag_per_repeat</code> is <code>false</code>. This class handles circular repeats internally via <code>vmap_parallel_gather</code> to select the correct repeat's weights each iteration, without the separate BSW (Buffer Sliding Window) prefetch mechanism of <code>CircularPipeline</code>.
</p>

<div class="callout">
<strong>Why a proxy?</strong> The real DeepSeek-V2 (60 layers) and V3 (61 layers) use Multi-head Latent Attention (MLA) and a specialized decoder block. This proxy replaces those with standard global attention and the Mixtral decoder, keeping the MoE router and pipeline geometry identical while being cheaper to compile and profile.
</div>

<!-- ================================================================== -->
<h2 id="s2">2. What is Temp Memory (tmem) in XLA?</h2>

<p>
When you call <code>compiled.memory_analysis()</code> on a JAX-compiled function, it returns a <code>CompiledMemoryStats</code> object with four fields:
</p>

<pre><code>compiled = p_train_step.lower(*lower_args).compile(compiler_options=...)
stats = compiled.memory_analysis()

stats.argument_size_in_bytes   # Size of all input arguments (params, optimizer state, batch)
stats.output_size_in_bytes     # Size of all outputs (updated state, metrics)
stats.temp_size_in_bytes       # Size of all temporary buffers  <-- THIS IS TMEM
stats.alias_size_in_bytes      # Overlap between arguments and outputs (in-place updates)
stats.host_temp_size_in_bytes  # Host-side temporaries (for offloading)</code></pre>

<p>
The total device memory required is <code>argument + output + temp - alias</code>. MaxText prints this in <code>max_utils.print_compiled_memory_stats()</code>.
</p>

<p>
"Temp memory" is the peak amount of HBM needed for all intermediate buffers that are neither function inputs nor outputs. These include:
</p>

<ul>
  <li><strong>Forward-pass activations</strong> saved for the backward pass (checkpoint/remat boundaries)</li>
  <li><strong>Collective communication buffers</strong> (send/recv buffers for <code>all-reduce</code>, <code>ppermute</code>, <code>all-gather</code>)</li>
  <li><strong>Intermediate computation results</strong> that must be live simultaneously (e.g., both inputs and outputs of a matmul during the backward pass)</li>
  <li><strong>Sharding constraint materialization</strong> (explicit copies forced by <code>with_sharding_constraint</code> or <code>reshard</code>)</li>
</ul>

<h3>XLA's BufferAssignment Pass</h3>

<p>
The temp memory size is determined by XLA's <code>BufferAssignment</code> pass, which runs after all HLO optimization passes. The sequence:
</p>

<ol>
  <li><strong>HLO optimization passes</strong> transform the graph: CSE, constant folding, algebraic simplification, layout assignment, fusion</li>
  <li><strong>Memory space assignment</strong> decides which buffers go to HBM vs. host memory</li>
  <li><strong>Scheduling</strong> (<code>LatencyHidingScheduler</code> on GPU) determines the execution order, trying to overlap async collectives with compute</li>
  <li><strong>Buffer assignment</strong> analyzes live ranges of all HLO values in the scheduled order and assigns physical memory offsets. Buffers with non-overlapping live ranges can share the same physical memory</li>
  <li>The <em>peak</em> of the buffer assignment waterfall is <code>temp_size_in_bytes</code></li>
</ol>

<div class="callout">
<strong>Key insight:</strong> Tmem is not about the total number of intermediate values. It is about the <em>peak simultaneous liveness</em> of those values after scheduling. Two large activations that are never live at the same time share memory. The scheduler's ordering directly affects tmem, which is why <code>LatencyHidingScheduler</code> matters: it may reorder operations to overlap communication, but this can increase the number of simultaneously live buffers and thus increase tmem.
</div>

<!-- Diagram: XLA Compilation Pipeline -->
<div class="diagram-container">
<svg viewBox="0 0 900 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:900px;">
  <!-- Pipeline stages -->
  <defs>
    <marker id="arrow" viewBox="0 0 10 7" refX="10" refY="3.5" markerWidth="8" markerHeight="6" orient="auto-start-auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#58a6ff"/>
    </marker>
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="120%">
      <feDropShadow dx="1" dy="2" stdDeviation="2" flood-opacity="0.3"/>
    </filter>
  </defs>

  <!-- Stage boxes -->
  <rect x="10" y="50" width="110" height="55" rx="8" fill="#1f2937" stroke="#58a6ff" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="65" y="72" text-anchor="middle" fill="#58a6ff" font-size="11" font-weight="600">Python</text>
  <text x="65" y="90" text-anchor="middle" fill="#8b949e" font-size="9">train_step()</text>

  <line x1="125" y1="77" x2="150" y2="77" stroke="#58a6ff" stroke-width="1.5" marker-end="url(#arrow)"/>

  <rect x="155" y="50" width="110" height="55" rx="8" fill="#1f2937" stroke="#f0883e" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="210" y="72" text-anchor="middle" fill="#f0883e" font-size="11" font-weight="600">JAX Tracing</text>
  <text x="210" y="90" text-anchor="middle" fill="#8b949e" font-size="9">jax.jit &rarr; Jaxpr</text>

  <line x1="270" y1="77" x2="295" y2="77" stroke="#58a6ff" stroke-width="1.5" marker-end="url(#arrow)"/>

  <rect x="300" y="50" width="110" height="55" rx="8" fill="#1f2937" stroke="#7ee787" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="355" y="72" text-anchor="middle" fill="#7ee787" font-size="11" font-weight="600">Lowering</text>
  <text x="355" y="90" text-anchor="middle" fill="#8b949e" font-size="9">.lower() &rarr; HLO</text>

  <line x1="415" y1="77" x2="440" y2="77" stroke="#58a6ff" stroke-width="1.5" marker-end="url(#arrow)"/>

  <rect x="445" y="50" width="130" height="55" rx="8" fill="#1f2937" stroke="#d2a8ff" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="510" y="72" text-anchor="middle" fill="#d2a8ff" font-size="11" font-weight="600">XLA Compiler</text>
  <text x="510" y="90" text-anchor="middle" fill="#8b949e" font-size="9">.compile() &rarr; passes</text>

  <line x1="580" y1="77" x2="605" y2="77" stroke="#58a6ff" stroke-width="1.5" marker-end="url(#arrow)"/>

  <rect x="610" y="50" width="140" height="55" rx="8" fill="#1f2937" stroke="#f85149" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="680" y="68" text-anchor="middle" fill="#f85149" font-size="11" font-weight="600">BufferAssignment</text>
  <text x="680" y="85" text-anchor="middle" fill="#8b949e" font-size="9">live range analysis</text>
  <text x="680" y="98" text-anchor="middle" fill="#8b949e" font-size="9">&rarr; temp_size_in_bytes</text>

  <!-- Breakdown box below XLA Compiler -->
  <rect x="350" y="130" width="350" height="75" rx="6" fill="#161b22" stroke="#30363d" stroke-width="1" stroke-dasharray="4,3"/>
  <text x="370" y="150" fill="#8b949e" font-size="10">XLA passes that affect tmem:</text>
  <text x="380" y="165" fill="#c9d1d9" font-size="9">1. HLO optimization (CSE, fusion, layout)</text>
  <text x="380" y="178" fill="#c9d1d9" font-size="9">2. LatencyHidingScheduler (async overlap)</text>
  <text x="380" y="191" fill="#c9d1d9" font-size="9">3. BufferAssignment (peak liveness = tmem)</text>

  <line x1="510" y1="110" x2="510" y2="125" stroke="#30363d" stroke-width="1" stroke-dasharray="3,3"/>
</svg>
</div>

<!-- ================================================================== -->
<h2 id="s3">3. The Compilation Pipeline: Python to Buffer Assignment</h2>

<p>
Let us trace the full path from the Python <code>train_step</code> function to the XLA buffer assignment, using our ds-proxy config.
</p>

<h3>3.1 Python: train_step</h3>

<p>
In <code>train.py</code>, the training loop calls:
</p>

<pre><code># JIT compilation
p_train_step = jax.jit(train_step, ...)

# Lowering + compilation
lower_args = (state, shaped_batch, init_rng)
compiled = p_train_step.lower(*lower_args).compile(compiler_options=compiler_options)
compiled_stats = compiled.memory_analysis()
max_utils.print_compiled_memory_stats(compiled_stats)

# Execution
state, metrics = p_train_step(state, example_batch, *step_rng_args)</code></pre>

<p>
<code>train_step</code> calls <code>jax.value_and_grad(loss_fn, ...)</code>. Inside <code>loss_fn</code>, the model's <code>__call__</code> invokes the Pipeline module, which runs:
</p>

<pre><code># Pipeline.__call__ (simplified)
inputs = inputs.reshape([num_microbatches, micro_size, seq_len, emb_dim])
loop_state = self.init_states(inputs)

# nn.scan over total_iterations
for iteration in range(total_iterations):
    # get inputs for this iteration (rotated outputs from previous)
    stages_inputs = self.get_iteration_inputs(loop_iteration, state_io, circ_storage, shift)
    stages_inputs = checkpoint_name(stages_inputs, "iteration_input")  # REMAT BOUNDARY

    # get weights for current repeat (circular pipeline)
    stage_weights = self.get_current_stage_weights(pipeline_weights, loop_iteration)

    # run all stages in parallel (vmapped)
    stages_output = vmap_func(decoder_layer, stage_weights, stages_inputs, ...)

    # rotate outputs, update state_io
    loop_state = self.get_new_loop_state(stages_output, loop_state)</code></pre>

<h3>3.2 JAX Tracing: Jaxpr</h3>

<p>
When <code>jax.jit</code> traces <code>train_step</code>, it produces a Jaxpr (JAX expression) representing the computation as a flat sequence of primitive operations. Key things that happen during tracing:
</p>

<ul>
  <li><code>nn.scan</code> over pipeline iterations becomes a <code>jax.lax.scan</code> primitive</li>
  <li><code>nn.remat</code> wraps the scan body in a <code>remat</code> / <code>checkpoint</code> primitive, governed by the <code>save_only_these_names</code> policy</li>
  <li><code>nn.vmap</code> over stages becomes a <code>jax.vmap</code> (batched) primitive</li>
  <li><code>with_sharding_constraint</code> / <code>reshard</code> calls become explicit sharding primitives in the Jaxpr</li>
  <li><code>shard_map</code> + <code>ppermute</code> become SPMD collective primitives</li>
  <li><code>checkpoint_name(x, "decoder_layer_input")</code> attaches a name tag to the value for the remat policy to reference</li>
</ul>

<h3>3.3 Lowering: Jaxpr to HLO</h3>

<p>
<code>.lower()</code> converts the Jaxpr to an XLA <code>HloModule</code>. This phase translates JAX primitives into HLO operations and embeds sharding annotations as metadata on the HLO instructions. Key HLO ops that appear at this stage:
</p>

<ul>
  <li><code>while</code> (from <code>lax.scan</code>) with the scan body as a computation</li>
  <li><code>conditional</code> (from <code>jnp.where</code>) or <code>select</code> (the more common lowered form)</li>
  <li><code>collective-permute</code> (from <code>ppermute</code>) &mdash; before the optimization</li>
  <li><code>slice</code>, <code>concatenate</code>, <code>pad</code> &mdash; after the optimization</li>
  <li><code>dynamic-slice</code> (from <code>dynamic_slice_in_dim</code>)</li>
  <li><code>gather</code> (from <code>x.at[idx].get()</code>) &mdash; before the optimization</li>
  <li><code>all-gather</code> (from FSDP weight gathering via sharding constraints)</li>
  <li><code>custom-call</code> (cuDNN flash attention, NCCL collectives on GPU)</li>
</ul>

<h3>3.4 Compilation: HLO to Executable</h3>

<p>
<code>.compile()</code> runs the full XLA compiler pipeline. The SPMD partitioner runs during this phase, converting logical sharding annotations into physical device assignments and inserting the necessary communication operations (all-gather, all-reduce, collective-permute, etc.). The passes relevant to tmem:
</p>

<ol>
  <li><strong>SPMD partitioning</strong>: Transforms the single-device HLO program into a per-device SPMD program by partitioning tensors according to their sharding annotations and inserting collectives for cross-device communication.</li>
  <li><strong>HLO optimization passes</strong>: CSE (common subexpression elimination) can merge duplicate sharding copies. Fusion combines element-wise ops into single kernels, reducing intermediate buffers.</li>
  <li><strong>Layout assignment</strong>: Determines physical memory layout (row-major, column-major, tiled) for each buffer. Different layouts for the same logical tensor require a layout-change copy, adding to tmem.</li>
  <li><strong>LatencyHidingScheduler (LHS)</strong>: On GPU, LHS reorders operations to overlap async collectives (NCCL all-reduce, all-gather) with compute (matmuls). This is critical for performance but can increase tmem: overlapping means both the collective's buffer and the compute's operands are live simultaneously.</li>
  <li><strong>Buffer assignment</strong>: Walks the scheduled instruction sequence, tracking which buffers are "live" (produced but not yet consumed by their last user). Assigns physical memory offsets. The peak of the live set is <code>temp_size_in_bytes</code>.</li>
</ol>

<div class="callout-warn callout">
<strong>The tension:</strong> The LatencyHidingScheduler improves throughput by overlapping communication and computation, but this increases the number of simultaneously live buffers. Reducing tmem without hurting LHS overlap is the central challenge of this work.
</div>

<!-- ================================================================== -->
<h2 id="s4">4. Remat Policy and Checkpoint Boundaries</h2>

<p>
Rematerialization (remat, or gradient checkpointing) is the primary mechanism for controlling tmem from the Python level. The idea: instead of saving all forward-pass activations for the backward pass, discard some and recompute them during backpropagation.
</p>

<h3>4.1 How save_only_these_names Works</h3>

<p>
MaxText's pipeline uses <code>jax.checkpoint_policies.save_only_these_names(...)</code> to define which activations survive into the backward pass:
</p>

<pre><code># In Pipeline.get_pipeline_remat_policy():
names_to_save = ["iteration_input"]
if self.config.pipeline_save_decoder_layer_input:
    names_to_save.append("decoder_layer_input")
save_input_policy = jax.checkpoint_policies.save_only_these_names(*names_to_save)</code></pre>

<p>
These names are attached to tensors in the forward pass via <code>jax.ad_checkpoint.checkpoint_name()</code>:
</p>

<pre><code># In Pipeline.run_one_iteration():
stages_inputs = self.get_iteration_inputs(...)
stages_inputs = jax.ad_checkpoint.checkpoint_name(stages_inputs, "iteration_input")

# In MixtralDecoderLayer.__call__():
inputs = checkpoint_name(inputs, "decoder_layer_input")</code></pre>

<p>
When JAX encounters a <code>remat</code> boundary during differentiation, it checks each intermediate value against the policy. Named values matching the policy are saved; everything else is discarded and recomputed from the nearest saved checkpoint.
</p>

<h3>4.2 The decoder_layer_input Checkpoint: Memory vs. Compute</h3>

<p>
Each decoder layer in the Mixtral block tags its input:
</p>

<pre><code># mixtral.py, MixtralDecoderLayer.__call__():
inputs = shard(inputs)
inputs = checkpoint_name(inputs, "decoder_layer_input")
# ... attention, MoE, residual ...</code></pre>

<p>
When <code>"decoder_layer_input"</code> is in the saved names list, XLA allocates memory to hold this activation throughout the forward pass, keeping it alive until the backward pass consumes it. For our config, the allocation per saved checkpoint is:
</p>

<pre><code>checkpoint_size = num_stages * num_layers_per_stage * [micro_size, seq_len, emb_dim] * sizeof(bf16)
                = 8 * 4 * [3, 4096, 4096] * 2 bytes</code></pre>

<p>
This is the dominant contributor to tmem in pipeline configurations, because these checkpoints must be live simultaneously across all pipeline iterations (the scan carries them forward).
</p>

<!-- Diagram: Remat checkpoint boundary -->
<div class="diagram-container">
<svg viewBox="0 0 850 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:850px;">
  <defs>
    <marker id="arr2" viewBox="0 0 10 7" refX="10" refY="3.5" markerWidth="7" markerHeight="5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#c9d1d9"/>
    </marker>
  </defs>

  <text x="20" y="25" fill="#58a6ff" font-size="13" font-weight="600">Forward Pass (one pipeline iteration, one stage)</text>

  <!-- Forward boxes -->
  <rect x="20" y="40" width="120" height="40" rx="5" fill="#1a3a1a" stroke="#7ee787" stroke-width="1.5"/>
  <text x="80" y="65" text-anchor="middle" fill="#7ee787" font-size="10" font-weight="600">iteration_input</text>
  <text x="80" y="52" text-anchor="middle" fill="#8b949e" font-size="8">SAVED</text>

  <line x1="145" y1="60" x2="170" y2="60" stroke="#c9d1d9" stroke-width="1" marker-end="url(#arr2)"/>

  <rect x="175" y="40" width="130" height="40" rx="5" fill="#1a1a2e" stroke="#d2a8ff" stroke-width="1.5"/>
  <text x="240" y="55" text-anchor="middle" fill="#d2a8ff" font-size="10">decoder_layer 0</text>
  <text x="240" y="68" text-anchor="middle" fill="#8b949e" font-size="8">RMSNorm+Attn+MoE</text>

  <line x1="310" y1="60" x2="335" y2="60" stroke="#c9d1d9" stroke-width="1" marker-end="url(#arr2)"/>

  <!-- Conditional save box -->
  <rect x="340" y="40" width="130" height="40" rx="5" fill="#2a1a1a" stroke="#f0883e" stroke-width="1.5" stroke-dasharray="4,3"/>
  <text x="405" y="55" text-anchor="middle" fill="#f0883e" font-size="10">decoder_layer_input</text>
  <text x="405" y="68" text-anchor="middle" fill="#8b949e" font-size="8">SAVED or RECOMPUTED</text>

  <line x1="475" y1="60" x2="500" y2="60" stroke="#c9d1d9" stroke-width="1" marker-end="url(#arr2)"/>

  <rect x="505" y="40" width="130" height="40" rx="5" fill="#1a1a2e" stroke="#d2a8ff" stroke-width="1.5"/>
  <text x="570" y="55" text-anchor="middle" fill="#d2a8ff" font-size="10">decoder_layer 1</text>
  <text x="570" y="68" text-anchor="middle" fill="#8b949e" font-size="8">RMSNorm+Attn+MoE</text>

  <line x1="640" y1="60" x2="665" y2="60" stroke="#c9d1d9" stroke-width="1" marker-end="url(#arr2)"/>
  <text x="690" y="65" fill="#8b949e" font-size="10">... layers 2, 3</text>

  <!-- Backward pass -->
  <text x="20" y="115" fill="#f85149" font-size="13" font-weight="600">Backward Pass</text>

  <!-- Case 1: saved -->
  <text x="20" y="140" fill="#7ee787" font-size="11">Case A: pipeline_save_decoder_layer_input = true</text>
  <rect x="30" y="150" width="380" height="35" rx="5" fill="#1a3a1a" stroke="#7ee787" stroke-width="1"/>
  <text x="220" y="172" text-anchor="middle" fill="#c9d1d9" font-size="9">Backward uses saved decoder_layer_input activations from HBM.</text>
  <text x="430" y="172" fill="#f85149" font-size="10" font-weight="600">high tmem</text>

  <!-- Case 2: recomputed -->
  <text x="20" y="210" fill="#f0883e" font-size="11">Case B: pipeline_save_decoder_layer_input = false</text>
  <rect x="30" y="220" width="380" height="35" rx="5" fill="#2a1a1a" stroke="#f0883e" stroke-width="1"/>
  <text x="220" y="237" text-anchor="middle" fill="#c9d1d9" font-size="9">Backward recomputes from iteration_input. Extra FLOPs, zero extra HBM.</text>
  <text x="430" y="237" fill="#7ee787" font-size="10" font-weight="600">0 GB tmem</text>

  <!-- Arrow showing tradeoff -->
  <text x="520" y="200" fill="#58a6ff" font-size="11" font-weight="600">Trade-off:</text>
  <text x="520" y="215" fill="#8b949e" font-size="9">Save: fast backward, high tmem</text>
  <text x="520" y="230" fill="#8b949e" font-size="9">Recompute: slow backward, low tmem</text>
</svg>
</div>

<p>
When <code>pipeline_save_decoder_layer_input = true</code> (the default), each of the <code>num_layers_per_pipeline_stage</code> layers per stage saves its input. With 4 layers per stage and 8 stages, that is 32 checkpoint buffers of shape <code>[3, 4096, 4096]</code> in bf16. These must remain live across the full forward-then-backward lifetime of the scan, making them the dominant tmem contributor.
</p>

<p>
Setting <code>pipeline_save_decoder_layer_input = false</code> removes <code>"decoder_layer_input"</code> from the policy. Now the only saved checkpoint is <code>"iteration_input"</code> (one per scan iteration, at the input to the entire stage, not per-layer). During the backward pass, the decoder layer inputs are recomputed by re-running the forward computation from the iteration input. This trades extra FLOPs (roughly one extra forward pass per stage) for eliminating the per-layer checkpoint allocations.
</p>

<!-- ================================================================== -->
<h2 id="s5">5. Pipeline Parallelism and Tmem: The Core Problem</h2>

<p>
Pipeline parallelism in MaxText divides the model's decoder layers across <code>num_stages</code> devices along the <code>stage</code> mesh axis. Each pipeline iteration processes one microbatch through all stages simultaneously (via vmap). The key data structures:
</p>

<ul>
  <li><strong><code>state_io</code></strong>: <code>[num_stages, microbatches_per_stage, micro_size, seq_len, emb_dim]</code> &mdash; holds inputs and collects outputs</li>
  <li><strong><code>shift</code></strong>: <code>[num_stages, micro_size, seq_len, emb_dim]</code> &mdash; passes each stage's output to the next stage's input</li>
  <li><strong><code>circ_storage</code></strong>: <code>[num_stages, num_microbatches, micro_size, seq_len, emb_dim]</code> &mdash; circular buffer for repeats (when <code>num_microbatches > num_stages</code>)</li>
</ul>

<p>
After each iteration, the pipeline must <em>rotate</em> the outputs: stage <em>i</em>'s output becomes stage <em>i+1</em>'s input. Stage 0 receives a fresh microbatch from <code>state_io</code> (or from <code>circ_storage</code> in circular mode). The <code>state_io</code> itself must be rotated left: each stage's slot shifts up to make room for the last stage's output at the bottom.
</p>

<p>
These rotation operations are the second-largest tmem contributor after remat checkpoints, and the way they are implemented has a profound impact on XLA's buffer allocation.
</p>

<!-- ================================================================== -->
<h2 id="s6">6. Optimization 1: Replacing ppermute/shard_map with Local Ops</h2>

<p>
This is the single largest tmem reduction after the remat checkpoint change.
</p>

<h3>6.1 Before: shard_map + ppermute (CircularPipeline Style)</h3>

<p>
The <code>CircularPipeline</code> class (and the <code>Pipeline</code> class before the optimization) implements rotation using inter-device collectives:
</p>

<pre><code>@jax.shard_map(mesh=self.mesh, in_specs=self.stages_in_spec,
               out_specs=self.stages_in_spec, check_vma=True)
def _rotate_right(arr):
    stage_size = jax.lax.axis_size("stage")
    perm = [(i, (i + 1) % stage_size) for i in range(stage_size)]
    return jax.lax.ppermute(arr, axis_name="stage", perm=perm)

@jax.shard_map(mesh=self.mesh, in_specs=self.stages_in_spec,
               out_specs=self.stages_in_spec, check_vma=True)
def _shift_right(arr):
    stage_idx = jax.lax.axis_index("stage")
    stage_size = jax.lax.axis_size("stage")
    perm = [(i, (i + 1) % stage_size) for i in range(stage_size)]
    arr = jax.lax.ppermute(arr, axis_name="stage", perm=perm)
    return jnp.where(stage_idx == 0, jnp.zeros_like(arr), arr)</code></pre>

<p>
This generates <code>collective-permute</code> HLO operations, which have two tmem costs:
</p>

<ol>
  <li><strong>NCCL/RCCL buffer allocation</strong>: A collective permute requires a send buffer and receive buffer that are simultaneously live. Each is the size of one stage's activation shard: <code>[micro_size, seq_len, emb_dim]</code>. On GPU, these are allocated by NCCL and must persist until the collective completes.</li>
  <li><strong><code>shard_map</code> with <code>check_vma=True</code></strong>: The <code>check_vma</code> parameter enables JAX's Varying Manual Axes (VMA) type system within the <code>shard_map</code> body. VMA tracks whether values are <em>varying</em> (different across devices on a given mesh axis) or <em>invariant</em> (identical). When enabled, it validates that the output's varying/invariant status matches the declared <code>out_specs</code>, and enables more efficient autodiff by skipping unnecessary collectives for invariant values. However, the VMA tracking can also cause the compiler to insert explicit copies to enforce the declared output sharding, adding buffer allocation to tmem.</li>
</ol>

<p>
For the <code>state_io</code> rotation, the same pattern was used with <code>_shift_left</code> inside another <code>shard_map</code> block:
</p>

<pre><code>@jax.shard_map(mesh=self.mesh,
    in_specs=(self.state_io_spec, self.stages_in_spec, self.stages_in_spec, P()),
    out_specs=self.state_io_spec, check_vma=True)
def _update_state_io(state_in, stream_slice, output, stream_buf_idx):
    stage_size = jax.lax.axis_size("stage")
    stream_slice = _shift_left(stream_slice, stage_size, output)  # another ppermute
    stream_slice = jnp.expand_dims(stream_slice, 1)
    return jax.lax.dynamic_update_slice_in_dim(state_in, stream_slice, stream_buf_idx, axis=1)</code></pre>

<h3>6.2 After: Local slice + concatenate (Pipeline Class)</h3>

<p>
The optimized <code>Pipeline</code> class replaces all collectives with purely local array operations:
</p>

<pre><code>def _rotate_right(arr):
    # [0,1,...,N-1] -> [N-1,0,1,...,N-2]
    last = jax.lax.slice_in_dim(arr, self.num_stages - 1, self.num_stages, axis=0)
    except_last = jax.lax.slice_in_dim(arr, 0, self.num_stages - 1, axis=0)
    return jnp.concatenate([last, except_last], axis=0)

def _shift_right(arr):
    # [0,1,...,N-1] -> [0, 0,1,...,N-2]  (zero-fill first position)
    padding = [[1, 0]] + [[0, 0]] * (arr.ndim - 1)
    # Use lax.slice to guarantee the gradient is a pad.
    return jax.lax.slice(jnp.pad(arr, padding), [0] * arr.ndim, arr.shape)</code></pre>

<p>
And the state_io update:
</p>

<pre><code>def _update_state_io(state_in, stream_slice, output):
    # Shift left by 1, fill last position with output
    padding = [[0, 1]] + [[0, 0]] * (stream_slice.ndim - 1)
    stream_slice = jax.lax.slice_in_dim(
        jnp.pad(stream_slice, padding), 1, stream_slice.shape[0] + 1, axis=0)
    stream_slice = jnp.where(
        jax.lax.broadcasted_iota("int32", stream_slice.shape, 0) == self.num_stages - 1,
        output, stream_slice)
    stream_slice = jnp.expand_dims(stream_slice, 1)
    return jax.lax.dynamic_update_slice_in_dim(state_in, stream_slice, stream_buf_idx, axis=1)</code></pre>

<h3>6.3 Why This Works: XLA's View of Local vs. Collective Ops</h3>

<p>
The key insight is that <code>lax.slice_in_dim</code> and <code>jnp.concatenate</code> along axis 0 (the stage axis) are operations on the logically full <code>[num_stages, ...]</code> array. The array is <em>sharded</em> along the stage axis (each device holds its own stage's shard), but the SPMD partitioner (which runs during <code>.compile()</code>) transforms these logical full-array operations into per-device local operations. For a slice+concat rotation pattern with statically known indices, XLA can often implement the per-device portion as buffer views (pointer arithmetic) with zero copies, or at most a single local memcpy. This is far cheaper than a multi-device collective that requires separate send/recv buffers and NCCL coordination.
</p>

<!-- Diagram: Pipeline rotation before vs. after -->
<div class="diagram-container">
<svg viewBox="0 0 850 400" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:850px;">
  <defs>
    <marker id="arr3" viewBox="0 0 10 7" refX="10" refY="3.5" markerWidth="7" markerHeight="5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#f85149"/>
    </marker>
    <marker id="arr4" viewBox="0 0 10 7" refX="10" refY="3.5" markerWidth="7" markerHeight="5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#7ee787"/>
    </marker>
  </defs>

  <!-- BEFORE -->
  <text x="20" y="25" fill="#f85149" font-size="14" font-weight="700">BEFORE: shard_map + ppermute</text>
  <text x="20" y="42" fill="#8b949e" font-size="10">Each stage is on a separate device. ppermute is an inter-device collective.</text>

  <!-- Device boxes -->
  <rect x="20" y="55" width="85" height="35" rx="4" fill="#2a1a1a" stroke="#f85149"/>
  <text x="62" y="77" text-anchor="middle" fill="#f85149" font-size="9" font-weight="600">Device 0</text>
  <rect x="120" y="55" width="85" height="35" rx="4" fill="#2a1a1a" stroke="#f85149"/>
  <text x="162" y="77" text-anchor="middle" fill="#f85149" font-size="9" font-weight="600">Device 1</text>
  <rect x="220" y="55" width="85" height="35" rx="4" fill="#2a1a1a" stroke="#f85149"/>
  <text x="262" y="77" text-anchor="middle" fill="#f85149" font-size="9" font-weight="600">Device 2</text>
  <text x="320" y="77" fill="#8b949e" font-size="12">...</text>
  <rect x="350" y="55" width="85" height="35" rx="4" fill="#2a1a1a" stroke="#f85149"/>
  <text x="392" y="77" text-anchor="middle" fill="#f85149" font-size="9" font-weight="600">Device 7</text>

  <!-- Arrows showing ppermute communication -->
  <path d="M 105 72 C 110 50, 118 50, 120 72" stroke="#f85149" fill="none" stroke-width="1.5" marker-end="url(#arr3)"/>
  <path d="M 205 72 C 210 50, 218 50, 220 72" stroke="#f85149" fill="none" stroke-width="1.5" marker-end="url(#arr3)"/>
  <path d="M 435 72 C 460 45, 30 45, 20 72" stroke="#f85149" fill="none" stroke-width="1" stroke-dasharray="4,3" marker-end="url(#arr3)"/>

  <!-- NCCL buffers -->
  <rect x="20" y="100" width="85" height="25" rx="3" fill="#3a1a1a" stroke="#f85149" stroke-dasharray="3,2"/>
  <text x="62" y="117" text-anchor="middle" fill="#f85149" font-size="8">send buf</text>
  <rect x="120" y="100" width="85" height="25" rx="3" fill="#3a1a1a" stroke="#f85149" stroke-dasharray="3,2"/>
  <text x="162" y="117" text-anchor="middle" fill="#f85149" font-size="8">recv buf</text>
  <text x="225" y="117" fill="#8b949e" font-size="8">+ send/recv for each device pair</text>

  <rect x="500" y="55" width="320" height="80" rx="6" fill="#161b22" stroke="#f85149" stroke-width="1"/>
  <text x="515" y="75" fill="#f85149" font-size="10" font-weight="600">HLO ops generated:</text>
  <text x="525" y="92" fill="#c9d1d9" font-size="9">collective-permute (NCCL ring)</text>
  <text x="525" y="106" fill="#c9d1d9" font-size="9">+ shard_map staging copies</text>
  <text x="525" y="120" fill="#c9d1d9" font-size="9">= 2x activation_shard tmem per call</text>

  <!-- AFTER -->
  <text x="20" y="170" fill="#7ee787" font-size="14" font-weight="700">AFTER: local slice + concatenate</text>
  <text x="20" y="187" fill="#8b949e" font-size="10">The logical [num_stages, ...] array is sharded along the stage axis. XLA's SPMD partitioner transforms these ops into per-device local operations.</text>

  <!-- Single device view -->
  <rect x="20" y="200" width="415" height="55" rx="6" fill="#1a2a1a" stroke="#7ee787"/>
  <text x="30" y="218" fill="#7ee787" font-size="10" font-weight="600">Logical array: [8, micro_size, seq_len, emb_dim] (sharded along axis 0)</text>

  <!-- Slice visualization -->
  <rect x="30" y="228" width="45" height="18" rx="2" fill="#2a3a2a" stroke="#7ee787" stroke-width="0.8"/>
  <text x="52" y="240" text-anchor="middle" fill="#c9d1d9" font-size="7">s0</text>
  <rect x="80" y="228" width="45" height="18" rx="2" fill="#2a3a2a" stroke="#7ee787" stroke-width="0.8"/>
  <text x="102" y="240" text-anchor="middle" fill="#c9d1d9" font-size="7">s1</text>
  <rect x="130" y="228" width="45" height="18" rx="2" fill="#2a3a2a" stroke="#7ee787" stroke-width="0.8"/>
  <text x="152" y="240" text-anchor="middle" fill="#c9d1d9" font-size="7">s2</text>
  <text x="186" y="240" fill="#8b949e" font-size="7">...</text>
  <rect x="200" y="228" width="45" height="18" rx="2" fill="#3a4a3a" stroke="#f0883e" stroke-width="1.2"/>
  <text x="222" y="240" text-anchor="middle" fill="#f0883e" font-size="7" font-weight="600">s7</text>

  <text x="260" y="240" fill="#7ee787" font-size="10">&rarr;</text>

  <rect x="280" y="228" width="45" height="18" rx="2" fill="#3a4a3a" stroke="#f0883e" stroke-width="1.2"/>
  <text x="302" y="240" text-anchor="middle" fill="#f0883e" font-size="7" font-weight="600">s7</text>
  <rect x="330" y="228" width="45" height="18" rx="2" fill="#2a3a2a" stroke="#7ee787" stroke-width="0.8"/>
  <text x="352" y="240" text-anchor="middle" fill="#c9d1d9" font-size="7">s0</text>
  <rect x="380" y="228" width="45" height="18" rx="2" fill="#2a3a2a" stroke="#7ee787" stroke-width="0.8"/>
  <text x="402" y="240" text-anchor="middle" fill="#c9d1d9" font-size="7">s1</text>

  <rect x="500" y="200" width="320" height="55" rx="6" fill="#161b22" stroke="#7ee787" stroke-width="1"/>
  <text x="515" y="220" fill="#7ee787" font-size="10" font-weight="600">HLO ops generated:</text>
  <text x="525" y="237" fill="#c9d1d9" font-size="9">slice + concatenate (or pad + slice)</text>
  <text x="525" y="250" fill="#c9d1d9" font-size="9">= 0 extra tmem (buffer view / single copy)</text>

  <!-- Gradient consideration -->
  <text x="20" y="290" fill="#d2a8ff" font-size="12" font-weight="600">Backward-pass gradient consideration</text>

  <rect x="20" y="300" width="810" height="85" rx="6" fill="#161b22" stroke="#d2a8ff" stroke-width="1"/>
  <text x="35" y="320" fill="#c9d1d9" font-size="10">The _shift_right function uses <tspan fill="#f0883e">jax.lax.slice(jnp.pad(arr, padding), ...)</tspan> instead of slice_in_dim.</text>
  <text x="35" y="338" fill="#c9d1d9" font-size="10">This is deliberate: <tspan fill="#d2a8ff">slice_in_dim</tspan> may lower to a <tspan fill="#f85149">gather</tspan> in some cases, whose gradient is a <tspan fill="#f85149">scatter</tspan>.</text>
  <text x="35" y="356" fill="#c9d1d9" font-size="10"><tspan fill="#7ee787">lax.slice</tspan> with static indices always generates a <tspan fill="#7ee787">pad</tspan> gradient, which is allocation-free.</text>
  <text x="35" y="374" fill="#8b949e" font-size="9">The gradient of pad(x, [[1,0], ...]) followed by slice is: pad(grad, [[0,1], ...]) &mdash; no scatter, no extra tmem.</text>
</svg>
</div>

<p>
The critical subtlety is in <code>_shift_right</code>: it uses <code>jax.lax.slice</code> (not <code>slice_in_dim</code>) on the padded result. This guarantees that the backward-pass gradient is a <code>pad</code> operation rather than a <code>scatter</code>. The <code>scatter</code> gradient of a <code>gather</code> requires allocating an output buffer and performing a non-trivial index-based write, while the <code>pad</code> gradient is a simple buffer extension (often a view) with zero allocation.
</p>

<h3>6.4 Why Not Use Local Ops in CircularPipeline?</h3>

<p>
The <code>CircularPipeline</code> class still uses <code>shard_map</code> + <code>ppermute</code> in its <code>advance_circular_buffers</code> method. This is because <code>CircularPipeline</code> is designed for the <code>pipeline_fsdp_ag_per_repeat = true</code> path, where the BSW (Buffer Sliding Window) mechanism requires explicit per-device operations for weight prefetching via <code>shard_map</code>. Converting it would require restructuring the BSW logic, which is a separate effort.
</p>

<p>
Our ds-proxy config uses the <code>Pipeline</code> class (since <code>pipeline_fsdp_ag_per_repeat = false</code>), so it benefits fully from the local-ops optimization.
</p>

<!-- ================================================================== -->
<h2 id="s7">7. Optimization 2: Removing Unnecessary Sharding Constraints</h2>

<p>
Several intermediate tensors in the pipeline had explicit sharding constraints that forced XLA to materialize copies:
</p>

<h3>7.1 Removed Constraints on shift and first_stage_in</h3>

<p>
In the original code, <code>get_iteration_inputs</code> applied sharding constraints to intermediate values:
</p>

<pre><code><span class="diff-del">- shift = self._maybe_shard_with_logical(shift, self.stages_in_logical)</span>
<span class="diff-del">- first_stage_in = self._maybe_shard_with_logical(first_stage_in, self.stages_in_logical)</span></code></pre>

<p>
<code>shift</code> is the rotated output from the previous iteration, and <code>first_stage_in</code> is the input for stage 0 (either from <code>state_io</code> or <code>circ_storage</code>). Both are constructed from tensors that already have the correct sharding from their source operations. Applying <code>_maybe_shard_with_logical</code> inserts <code>with_sharding_constraint</code> ops in the Jaxpr, which force XLA to validate and potentially copy the tensor to match the declared sharding. In the optimized code, the final <code>stages_in</code> result still gets the sharding constraint (line 170 of the current pipeline.py), so the intermediates do not need it.
</p>

<h3>7.2 Removed Constraint on microbatches_processed</h3>

<pre><code><span class="diff-del">- microbatches_processed = self._maybe_shard_with_name(</span>
<span class="diff-del">-     microbatches_processed, NamedSharding(self.mesh, P("stage")))</span></code></pre>

<p>
<code>microbatches_processed</code> is a tiny <code>int32[num_stages]</code> tensor used only to compute <code>microbatch_ids</code> and <code>repeat_ids</code>. Forcing it onto the <code>"stage"</code> mesh axis generated an explicit sharding op for what is essentially a scalar computation. Without the constraint, XLA treats it as an unsharded scalar, with no allocation overhead.
</p>

<!-- ================================================================== -->
<h2 id="s8">8. Optimization 3: dynamic_slice_in_dim vs. gather</h2>

<p>
The <code>vmap_gather</code> and <code>vmap_parallel_gather</code> functions select weights for each stage based on the current repeat ID. The original implementation used NumPy-style advanced indexing:
</p>

<pre><code><span class="diff-del">- def _gather_one(x, i):</span>
<span class="diff-del">-     idx = tuple(i if d == ids_dim else slice(None) for d in range(x.ndim))</span>
<span class="diff-del">-     return x.at[idx].get(out_sharding=replicated_sharding)</span></code></pre>

<p>
This generates an XLA <code>gather</code> operation, which:
</p>

<ol>
  <li>Allocates a gather descriptor table (index mapping)</li>
  <li>Allocates the output buffer</li>
  <li>The <code>out_sharding=replicated_sharding</code> hint forces a collective broadcast to replicate the result across devices, adding an <code>all-gather</code> collective buffer</li>
</ol>

<p>
The replacement uses <code>dynamic_slice_in_dim</code>:
</p>

<pre><code><span class="diff-add">+ def _gather_one(x, i):</span>
<span class="diff-add">+     return jnp.squeeze(jax.lax.dynamic_slice_in_dim(x, i, 1, ids_dim), ids_dim)</span></code></pre>

<p>
<code>dynamic_slice_in_dim</code> generates an XLA <code>dynamic-slice</code> op. When the slice size is statically known (here it is 1), XLA can often implement this as an in-place view with zero allocation. Critically, the <code>out_sharding</code> hint is also removed, eliminating the broadcast collective.
</p>

<!-- ================================================================== -->
<h2 id="s9">9. Optimization 4: pipeline_save_decoder_layer_input</h2>

<p>
As discussed in Section 4, this is the single largest tmem reduction. The change in code is minimal:
</p>

<pre><code># Pipeline.get_pipeline_remat_policy()

<span class="diff-del">- save_input_policy = jax.checkpoint_policies.save_only_these_names(</span>
<span class="diff-del">-     "iteration_input", "decoder_layer_input")</span>

<span class="diff-add">+ names_to_save = ["iteration_input"]</span>
<span class="diff-add">+ if self.config.pipeline_save_decoder_layer_input:</span>
<span class="diff-add">+     names_to_save.append("decoder_layer_input")</span>
<span class="diff-add">+ save_input_policy = jax.checkpoint_policies.save_only_these_names(*names_to_save)</span></code></pre>

<h3>9.1 The Reviewer's Concern and the Flag Solution</h3>

<p>
The original PR removed <code>"decoder_layer_input"</code> unconditionally for all pipeline users. Reviewer feedback flagged this as risky: any model using <code>ici_pipeline_parallelism > 1</code> would silently get extra recomputation in the backward pass, potentially increasing step time for configurations that were not tmem-constrained.
</p>

<p>
The solution: a new config field <code>pipeline_save_decoder_layer_input</code> with:
</p>

<ul>
  <li><strong>Default <code>true</code></strong>: preserves upstream behavior for all existing users</li>
  <li><strong>Explicit <code>false</code></strong>: opts in to the tmem reduction (used in <code>ds-proxy-se2-e256-h4096.yml</code> and the AOT compile test)</li>
</ul>

<p>
The field is defined in <code>types.py</code> alongside the other pipeline remat controls:
</p>

<pre><code>pipeline_save_decoder_layer_input: bool = Field(
    True,
    description=(
        "Whether to save 'decoder_layer_input' activations in the pipeline remat policy. "
        "Setting to False reduces temporary memory (tmem) during pipeline execution at the cost "
        "of recomputing decoder layer inputs in the backward pass."
    ),
)</code></pre>

<h3>9.2 Quantifying the Savings</h3>

<p>
With <code>pipeline_save_decoder_layer_input = true</code>, the saved activations per device are:
</p>

<pre><code># Per scan iteration, per stage, per layer:
activation_size = pipeline_microbatch_size * max_target_length * emb_dim * sizeof(bf16)
               = 3 * 4096 * 4096 * 2 = ~96 MB

# Total across all layers per stage:
per_stage = num_layers_per_pipeline_stage * activation_size = 4 * 96 MB = ~384 MB

# These are live across the full scan (all iterations):
# The scan carries them as part of the remat checkpoint</code></pre>

<p>
The exact savings from setting <code>pipeline_save_decoder_layer_input = false</code> depend on how many of these checkpoint buffers are simultaneously live after XLA's buffer assignment optimizes their placement. This is the dominant contributor to the overall 9.5 GB reduction, but the precise amount cannot be isolated from the other optimizations because buffer assignment is a global optimization: removing one large buffer changes the live ranges of all surrounding buffers, potentially allowing them to share memory slots they could not before.
</p>

<!-- ================================================================== -->
<h2 id="s10">10. Optimization 5: float32_weight_sum and MoE Accumulation</h2>

<p>
The MoE (Mixture of Experts) layer accumulates routed expert outputs. With <code>float32_weight_sum = true</code> (the old default), the accumulation buffer is cast to float32 before summing:
</p>

<pre><code># Conceptual MoE output accumulation:
# With float32_weight_sum=True:
accum = jnp.zeros([batch, seq_len, emb_dim], dtype=jnp.float32)  # fp32 buffer
for expert_output, weight in zip(expert_outputs, routing_weights):
    accum += weight * expert_output.astype(jnp.float32)
result = accum.astype(jnp.bfloat16)

# With float32_weight_sum=False:
accum = jnp.zeros([batch, seq_len, emb_dim], dtype=jnp.bfloat16)  # bf16 buffer
for expert_output, weight in zip(expert_outputs, routing_weights):
    accum += weight * expert_output  # stays in bf16</code></pre>

<p>
The float32 accumulation buffer is a separate allocation from the bf16 expert outputs, adding tmem per device for <code>emb_dim=4096</code>. Setting <code>float32_weight_sum = false</code> keeps everything in bf16, using the same buffer for accumulation and output.
</p>

<div class="callout-warn callout">
<strong>Precision trade-off:</strong> The float32 accumulation provides slightly better numerical stability for the MoE output. For most training configurations (especially with bf16 training), the precision difference is negligible, but it may matter for convergence-sensitive runs.
</div>

<p>
The default was changed in both <code>base.yml</code> and <code>types.py</code>:
</p>

<pre><code># base.yml
<span class="diff-del">- float32_weight_sum: true</span>
<span class="diff-add">+ float32_weight_sum: false</span>

# types.py
float32_weight_sum: bool = Field(
<span class="diff-del">-   True,</span>
<span class="diff-add">+   False,</span>
    description="Whether to use fp32 for MoE expert weight summation; true adds ~2 GB f32 temp per device.",
)</code></pre>

<!-- ================================================================== -->
<h2 id="s11">11. Optimization 6: skip_trivial_specs</h2>

<p>
A small but clean optimization in the sharding utility:
</p>

<pre><code># In PipelineBase._maybe_shard_with_logical():
return maybe_shard_with_logical(
    inputs, logical_axes,
    shard_mode=self.config.shard_mode,
    mesh=self.mesh,
    rules=self.config.logical_axis_rules,
    debug_sharding=self.config.debug_sharding,
    extra_stack_level=1,
<span class="diff-add">+   skip_trivial_specs=True,</span>
)</code></pre>

<p>
The implementation in <code>sharding.py</code>:
</p>

<pre><code>def maybe_shard_with_logical(inputs, logical_axes, mesh, shard_mode, ..., skip_trivial_specs=False):
    named_sharding = create_sharding(mesh, logical_axes, rules=rules)

    if skip_trivial_specs and all(ax is None or ax == () for ax in named_sharding.spec):
        return inputs  # <-- early return, no sharding op emitted

    return maybe_shard_with_name(inputs, named_sharding, ...)</code></pre>

<p>
When a <code>PartitionSpec</code> has all-<code>None</code> axes (fully replicated), the <code>with_sharding_constraint</code> / <code>reshard</code> call is semantically a no-op. But the compiler may still insert an explicit copy to "enforce" the trivial sharding. <code>skip_trivial_specs=True</code> short-circuits this path, preventing the sharding op from appearing in the Jaxpr at all.
</p>

<p>
This also affects the <code>shard</code> function inside <code>MixtralDecoderLayer.__call__</code>:
</p>

<pre><code># mixtral.py
def shard(x):
    return maybe_shard_with_logical(
        x, self.activation_axis_names,
        mesh=self.mesh, shard_mode=cfg.shard_mode,
        rules=cfg.logical_axis_rules,
        skip_trivial_specs=True,  # added
    )</code></pre>

<!-- ================================================================== -->
<h2 id="s12">12. Putting It All Together: The Savings Breakdown</h2>

<p>
The following table presents the <em>standalone impact</em> of each optimization: the approximate tmem reduction if only that single optimization were applied to the baseline. Because XLA's buffer assignment is a global optimization, these standalone estimates are <strong>not additive</strong>. When multiple optimizations are applied together, the total reduction (9.5 GB) is less than the sum of standalone impacts. This is expected: once one buffer is eliminated, XLA may reuse its memory slot for a different buffer, meaning that eliminating the second buffer produces less incremental savings than it would in isolation.
</p>

<table>
  <thead>
    <tr>
      <th>Optimization</th>
      <th>Mechanism</th>
      <th>Standalone Impact</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>pipeline_save_decoder_layer_input = false</code></td>
      <td>Eliminates per-layer decoder checkpoint saves across all scan iterations</td>
      <td style="color:#7ee787;">largest single contributor</td>
    </tr>
    <tr>
      <td>Replace <code>ppermute</code>/<code>shard_map</code> with local <code>slice</code>+<code>concat</code></td>
      <td>Eliminates collective buffers and shard_map staging copies</td>
      <td style="color:#7ee787;">second largest contributor</td>
    </tr>
    <tr>
      <td><code>float32_weight_sum = false</code></td>
      <td>Eliminates fp32 accumulation buffer in MoE</td>
      <td style="color:#7ee787;">moderate</td>
    </tr>
    <tr>
      <td>Remove sharding constraints on <code>shift</code>, <code>first_stage_in</code>, <code>microbatches_processed</code></td>
      <td>Eliminates forced intermediate tensor materializations</td>
      <td style="color:#7ee787;">moderate</td>
    </tr>
    <tr>
      <td><code>dynamic_slice_in_dim</code> replacing <code>gather</code> + replicated hint</td>
      <td>Eliminates gather descriptor table and broadcast collective buffer</td>
      <td style="color:#7ee787;">small</td>
    </tr>
    <tr>
      <td><code>skip_trivial_specs = True</code></td>
      <td>Eliminates spurious no-op sharding copies</td>
      <td style="color:#7ee787;">small</td>
    </tr>
    <tr style="font-weight:700; border-top: 2px solid var(--accent);">
      <td>Combined total (measured)</td>
      <td></td>
      <td style="color:#58a6ff;">29.9 GB &rarr; 20.4 GB (9.5 GB / 32% reduction)</td>
    </tr>
  </tbody>
</table>

<div class="callout-ok callout">
<strong>Why the standalone impacts do not sum to 9.5 GB:</strong> XLA's buffer assignment assigns memory offsets to all intermediate buffers based on their live ranges. When buffer A is removed, buffer B (which had an overlapping live range) may now fit into A's old memory slot. If buffer B is later also removed, its elimination provides zero <em>incremental</em> savings, even though its <em>standalone</em> impact would have been significant. The 9.5 GB is the empirically measured combined result of applying all optimizations together.
</div>

<!-- ================================================================== -->
<h2 id="s13">13. Memory Layout Diagram</h2>

<p>
The following diagram shows the HBM memory layout for one device during a training step, before and after the optimizations:
</p>

<div class="diagram-container">
<svg viewBox="0 0 850 520" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:850px;">
  <!-- BEFORE -->
  <text x="20" y="25" fill="#f85149" font-size="14" font-weight="700">BEFORE (29.9 GB tmem)</text>

  <!-- Memory bar -->
  <rect x="20" y="40" width="380" height="30" rx="3" fill="#1a1a2e" stroke="#30363d"/>
  <text x="210" y="60" text-anchor="middle" fill="#8b949e" font-size="10">Argument memory (params, optimizer state, batch)</text>

  <rect x="20" y="75" width="30" height="30" rx="3" fill="#1a2e1a" stroke="#30363d"/>
  <text x="35" y="95" text-anchor="middle" fill="#8b949e" font-size="7">out</text>

  <!-- Temp memory breakdown -->
  <rect x="55" y="75" width="345" height="30" rx="3" fill="#2a1a1a" stroke="#f85149" stroke-width="1.5"/>
  <text x="227" y="95" text-anchor="middle" fill="#f85149" font-size="10" font-weight="600">Temp memory: 29.9 GB</text>

  <!-- Breakdown of tmem -->
  <rect x="55" y="115" width="170" height="25" rx="3" fill="#3a1a1a" stroke="#f85149" stroke-width="0.8"/>
  <text x="140" y="132" text-anchor="middle" fill="#f85149" font-size="9">decoder_layer_input checkpoints</text>

  <rect x="230" y="115" width="115" height="25" rx="3" fill="#3a1a2a" stroke="#d2a8ff" stroke-width="0.8"/>
  <text x="287" y="132" text-anchor="middle" fill="#d2a8ff" font-size="9">ppermute buffers</text>

  <rect x="350" y="115" width="50" height="25" rx="3" fill="#2a2a1a" stroke="#f0883e" stroke-width="0.8"/>
  <text x="375" y="132" text-anchor="middle" fill="#f0883e" font-size="8">fp32 accum</text>

  <rect x="55" y="145" width="90" height="25" rx="3" fill="#1a2a2a" stroke="#58a6ff" stroke-width="0.8"/>
  <text x="100" y="162" text-anchor="middle" fill="#58a6ff" font-size="9">sharding copies</text>

  <rect x="150" y="145" width="80" height="25" rx="3" fill="#1a1a3a" stroke="#8b949e" stroke-width="0.8"/>
  <text x="190" y="162" text-anchor="middle" fill="#8b949e" font-size="9">gather+bcast</text>

  <rect x="235" y="145" width="90" height="25" rx="3" fill="#1a2a1a" stroke="#7ee787" stroke-width="0.8"/>
  <text x="280" y="162" text-anchor="middle" fill="#7ee787" font-size="9">other activations</text>

  <!-- Alias -->
  <rect x="20" y="180" width="50" height="20" rx="3" fill="#1a1a1a" stroke="#30363d" stroke-dasharray="3,2"/>
  <text x="45" y="194" text-anchor="middle" fill="#8b949e" font-size="8">alias</text>

  <!-- AFTER -->
  <text x="20" y="240" fill="#7ee787" font-size="14" font-weight="700">AFTER (20.4 GB tmem)</text>

  <rect x="20" y="255" width="380" height="30" rx="3" fill="#1a1a2e" stroke="#30363d"/>
  <text x="210" y="275" text-anchor="middle" fill="#8b949e" font-size="10">Argument memory (unchanged)</text>

  <rect x="20" y="290" width="30" height="30" rx="3" fill="#1a2e1a" stroke="#30363d"/>
  <text x="35" y="310" text-anchor="middle" fill="#8b949e" font-size="7">out</text>

  <rect x="55" y="290" width="235" height="30" rx="3" fill="#1a2a1a" stroke="#7ee787" stroke-width="1.5"/>
  <text x="172" y="310" text-anchor="middle" fill="#7ee787" font-size="10" font-weight="600">Temp memory: 20.4 GB</text>

  <!-- Breakdown of tmem after -->
  <rect x="55" y="330" width="100" height="25" rx="3" fill="#1a2a1a" stroke="#7ee787" stroke-width="0.8"/>
  <text x="105" y="347" text-anchor="middle" fill="#7ee787" font-size="9">iteration_input ckpt</text>

  <rect x="160" y="330" width="60" height="25" rx="3" fill="#1a2a1a" stroke="#7ee787" stroke-width="0.8"/>
  <text x="190" y="347" text-anchor="middle" fill="#7ee787" font-size="9">local ops</text>

  <rect x="225" y="330" width="65" height="25" rx="3" fill="#1a2a1a" stroke="#7ee787" stroke-width="0.8"/>
  <text x="257" y="347" text-anchor="middle" fill="#7ee787" font-size="9">other activ.</text>

  <rect x="20" y="365" width="50" height="20" rx="3" fill="#1a1a1a" stroke="#30363d" stroke-dasharray="3,2"/>
  <text x="45" y="379" text-anchor="middle" fill="#8b949e" font-size="8">alias</text>

  <!-- What was eliminated -->
  <text x="450" y="240" fill="#58a6ff" font-size="12" font-weight="600">What was eliminated:</text>

  <rect x="450" y="255" width="370" height="140" rx="6" fill="#161b22" stroke="#30363d"/>

  <line x1="465" y1="275" x2="475" y2="275" stroke="#f85149" stroke-width="2"/>
  <text x="485" y="279" fill="#c9d1d9" font-size="10">decoder_layer_input checkpoints</text>
  <text x="485" y="292" fill="#8b949e" font-size="8">Recomputed from iteration_input in backward pass</text>

  <line x1="465" y1="307" x2="475" y2="307" stroke="#f85149" stroke-width="2"/>
  <text x="485" y="311" fill="#c9d1d9" font-size="10">ppermute collective buffers</text>
  <text x="485" y="324" fill="#8b949e" font-size="8">Replaced by local slice+concat (zero-copy)</text>

  <line x1="465" y1="339" x2="475" y2="339" stroke="#f85149" stroke-width="2"/>
  <text x="485" y="343" fill="#c9d1d9" font-size="10">fp32 MoE accumulator</text>
  <text x="485" y="356" fill="#8b949e" font-size="8">Now bf16 (same buffer as expert output)</text>

  <line x1="465" y1="371" x2="475" y2="371" stroke="#f85149" stroke-width="2"/>
  <text x="485" y="375" fill="#c9d1d9" font-size="10">Sharding copies + gather buffers</text>
  <text x="485" y="388" fill="#8b949e" font-size="8">Removed constraints, dynamic_slice_in_dim</text>

  <!-- Formula -->
  <rect x="20" y="410" width="810" height="90" rx="8" fill="#161b22" stroke="#58a6ff" stroke-width="1"/>
  <text x="40" y="435" fill="#58a6ff" font-size="12" font-weight="600">Total device memory equation:</text>
  <text x="40" y="458" fill="#c9d1d9" font-size="11" font-family="monospace">
    total = argument_size + output_size + temp_size - alias_size
  </text>
  <text x="40" y="478" fill="#8b949e" font-size="10">
    The optimizations reduce only temp_size. argument_size (model weights + optimizer state) and output_size
  </text>
  <text x="40" y="493" fill="#8b949e" font-size="10">
    (updated state + metrics) are unchanged. alias_size (in-place updates of state) is also unchanged.
  </text>
</svg>
</div>

<!-- ================================================================== -->
<h2>Appendix: The Full Call Chain</h2>

<p>
For reference, the complete call chain from training entry point to the tmem-relevant operations:
</p>

<pre><code># Entry point
train.train_loop()
  &rarr; p_train_step = jax.jit(train_step, ...)
  &rarr; p_train_step.lower(*args).compile()  # produces XLA HloModule
  &rarr; compiled.memory_analysis()            # reports temp_size_in_bytes

# Inside train_step (during tracing):
train_step()
  &rarr; jax.value_and_grad(loss_fn, ...)
    &rarr; model(inputs, ...)                  # Linen __call__
      &rarr; Pipeline.__call__(inputs, ...)
        &rarr; init_states(inputs)              # allocates state_io, shift, circ_storage
        &rarr; nn.scan(run_iteration_scannable, length=total_iterations)
          &rarr; nn.remat(run_iteration_scannable, policy=get_pipeline_remat_policy())
            &rarr; run_one_iteration()
              &rarr; get_iteration_inputs()     # constructs stages_in from state_io/shift
              &rarr; checkpoint_name(stages_in, "iteration_input")   # REMAT SAVE POINT
              &rarr; get_current_stage_weights() # vmap_parallel_gather with dynamic_slice
              &rarr; vmap_func(decoder_layer, weights, inputs, ...)
                &rarr; MixtralDecoderLayer.__call__()
                  &rarr; checkpoint_name(inputs, "decoder_layer_input")  # CONDITIONAL SAVE
                  &rarr; pre_self_attention_layer_norm (RMSNorm)
                  &rarr; self_attention (cuDNN flash attention)
                  &rarr; post_self_attention_layer_norm (RMSNorm)
                  &rarr; MoeBlock_0 (RoutedMoE with float32_weight_sum control)
                  &rarr; residual + dropout
              &rarr; get_new_loop_state()
                &rarr; _rotate_right() or _shift_right()  # local slice+concat
                &rarr; _update_state_io()                  # local pad+slice+where

# After all XLA passes:
BufferAssignment
  &rarr; live range analysis on scheduled HLO
  &rarr; peak liveness = temp_size_in_bytes</code></pre>

<!-- ================================================================== -->
<h2>Appendix: Why These Optimizations Do Not Affect Numerical Correctness</h2>

<p>
Each optimization falls into one of two categories:
</p>

<h4>Semantics-preserving rewrites (no numerical change):</h4>
<ul>
  <li><strong>slice+concat replacing ppermute</strong>: Both implement the same permutation <code>[0,1,...,N-1] &rarr; [N-1,0,...,N-2]</code>. The local version operates on the logically full array (which is sharded across devices); XLA's SPMD partitioner handles the per-device transformation. The result is bit-identical.</li>
  <li><strong>dynamic_slice_in_dim replacing gather</strong>: Both select one element along a dimension. <code>dynamic_slice_in_dim(x, i, 1, dim)</code> and <code>x.at[idx].get()</code> produce the same output.</li>
  <li><strong>Removing sharding constraints</strong>: Sharding constraints are hints to the compiler, not computations. Removing them does not change any computed value.</li>
  <li><strong>skip_trivial_specs</strong>: Skipping a no-op (all-None) sharding constraint changes nothing.</li>
</ul>

<h4>Deliberate precision/compute trade-offs (documented in config):</h4>
<ul>
  <li><strong>pipeline_save_decoder_layer_input = false</strong>: Trades compute for memory. Activations are recomputed instead of saved. The recomputed values are bit-identical to the saved ones (assuming deterministic execution). Step time increases slightly; tmem decreases significantly.</li>
  <li><strong>float32_weight_sum = false</strong>: Keeps MoE accumulation in bf16 instead of casting to fp32. This introduces small numerical differences in the MoE output due to reduced precision in the summation. For bf16 training, this is typically acceptable.</li>
</ul>

<!-- ================================================================== -->
<h2>Appendix: Applying These Optimizations to Your Config</h2>

<p>
To enable the tmem reduction in any MaxText pipeline config:
</p>

<pre><code># Add to your .yml config:
pipeline_save_decoder_layer_input: false  # Major tmem reduction, adds recompute cost
float32_weight_sum: false                 # Savings from eliminating fp32 accum buffer

# These are already applied by default in the codebase:
# - Local slice+concat (Pipeline class)
# - dynamic_slice_in_dim (Pipeline class)
# - skip_trivial_specs (PipelineBase._maybe_shard_with_logical)
# - Removed unnecessary sharding constraints</code></pre>

<p>
If you are using <code>CircularPipeline</code> (i.e., <code>pipeline_fsdp_ag_per_repeat: true</code>), the local-ops optimization does not apply. That code path still uses <code>shard_map</code> + <code>ppermute</code> for stage rotation.
</p>

<p>
To verify the tmem reduction, inspect the compile-time log output:
</p>

<pre><code>Total memory size: X.X GB, Output size: X.X GB, Temp size: 20.4 GB,
Argument size: X.X GB, Host temp size: X.X GB.</code></pre>

</body>
</html>
