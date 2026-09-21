
Code branch: https://github.com/AI-Hypercomputer/maxtext/compare/shuningjin-emb

1. We use the folder `sandbox` for temporary files

2. do not commit change 

3. use tpu-testing skill

4. pyink
- read the file changed in https://github.com/AI-Hypercomputer/maxtext/compare/shuningjin-emb
- run commands like
```
pre-commit run --files \
check_qwix_interception.py \
docs/reference/core_concepts/quantization.md \
src/maxtext/configs/base.yml \
src/maxtext/configs/types.py \
src/maxtext/layers/quantizations.py \
tests/unit/quantizations_test.py
```

5. Test plan

(a)
```
python sandbox/check_qwix_interception.py
```
save the output to sandbox/check_qwix_interception_log.txt, only retain the line contains the word "DEBUG" or "[QWIX]"


(b) unit test
```
JAX_PLATFORMS=cpu pytest tests/unit/quantizations_test.py -k "LogitsProjQwixInterceptionTest"
```

6. Rationale: Quantizing Logits Projection for Both Main and MTP

- **Shared Head Architecture**: In DeepSeek-V3 and MaxText, MTP does not define a separate vocabulary projection layer. It explicitly reuses the main model's output head (`mtp_logits = self.decoder.apply_output_head(...)`). Targeting `decoder/logits_dense.*` naturally intercepts both invocations:
  - `op=dot_general0`: Main model forward logits projection.
  - `op=dot_general1`: MTP block forward logits projection.
- **Multiplied FLOP Savings**: The vocabulary projection GEMM is executed $(1 + M)$ times per forward pass (where $M$ is `mtp_num_layers`). Quantizing `logits_proj` cuts the arithmetic cost of this heavy GEMM by $\approx 2\times$ across the main pass and all MTP layers.
- **Gradient & Numerical Parity**: The single shared weight tensor receives gradients from both main and MTP heads during training. Quantizing both paths ensures consistent numerical representation and prevents gradient mismatch between FP8 and BF16 regimes.
- **Speculative Decoding Alignment**: At inference time, if MTP is used for speculative draft generation, having both heads in FP8 matches latency profiles and numerical precision between draft and target tokens.