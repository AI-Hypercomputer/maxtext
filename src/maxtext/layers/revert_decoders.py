import re

with open('/usr/local/google/home/mattdonati/common_files/maxtext/src/maxtext/layers/decoders.py', 'r') as f:
    content = f.read()

content = content.replace(
    "case DecoderBlockType.GEMMA4_SMALL:\n        return [gemma4_small.Gemma4SmallScannableBlockToLinen]",
    "case DecoderBlockType.GEMMA4_SMALL:\n        # PLE input + KV-share donor threading requires per-layer-index state,\n        # which is not expressible inside ``nn.scan``.\n        return [gemma4_small.Gemma4SmallDecoderLayerToLinen]"
)

with open('/usr/local/google/home/mattdonati/common_files/maxtext/src/maxtext/layers/decoders.py', 'w') as f:
    f.write(content)
print("Reverted successfully")
