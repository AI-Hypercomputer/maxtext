with open("src/maxtext/experimental/agent/ckpt_validation_pipeline/forward_pass_validator.py", "r") as f:
    content = f.read()

target1 = """        if name == "pre_self_attention_layer_norm" and "input_layernorm" in node:
          node = node["input_layernorm"]
          continue
        if name == "post_self_attention_layer_norm" and "post_attention_layernorm" in node:
          node = node["post_attention_layernorm"]
          continue
        if name == "self_attention" and "attention" in node:
          node = node["attention"]
          continue
        if name == "input_layernorm" and "pre_self_attention_layer_norm" in node:
          node = node["pre_self_attention_layer_norm"]
          continue
        if name == "post_attention_layernorm" and "post_self_attention_layer_norm" in node:
          node = node["post_self_attention_layer_norm"]
          continue
        if name == "attention" and "self_attention" in node:
          node = node["self_attention"]
          continue"""

replacement1 = """        SYNONYMS = {
            "pre_self_attention_layer_norm": ["input_layernorm", "pre_attention_norm"],
            "post_self_attention_layer_norm": ["post_attention_layernorm", "post_attention_norm"],
            "self_attention": ["attention"],
            "input_layernorm": ["pre_self_attention_layer_norm", "pre_attention_norm"],
            "post_attention_layernorm": ["post_self_attention_layer_norm", "post_attention_norm"],
            "attention": ["self_attention"],
            "mlp": ["feed_forward", "ffn"],
            "feed_forward": ["mlp", "ffn"],
            "ffn": ["mlp", "feed_forward"],
        }
        
        found_synonym = False
        if name in SYNONYMS:
          for syn in SYNONYMS[name]:
            if syn in node:
              node = node[syn]
              found_synonym = True
              break
        if found_synonym:
          continue"""

if target1 not in content:
    print("Failed to find target1")
content = content.replace(target1, replacement1)

target2 = """      if to_linen:
        key_map = {
            "pre_self_attention_layer_norm": "input_layernorm",
            "post_self_attention_layer_norm": "post_attention_layernorm",
            "self_attention": "attention",
        }
      else:
        key_map = {
            "input_layernorm": "pre_self_attention_layer_norm",
            "post_attention_layernorm": "post_self_attention_layer_norm",
            "attention": "self_attention",
        }"""

replacement2 = """      SYNONYMS = {
          "pre_self_attention_layer_norm": ["input_layernorm", "pre_attention_norm"],
          "post_self_attention_layer_norm": ["post_attention_layernorm", "post_attention_norm"],
          "self_attention": ["attention"],
          "mlp": ["feed_forward", "ffn"],
      }
      
      if to_linen:
        key_map = {k: v[0] for k, v in SYNONYMS.items()}
      else:
        key_map = {}
        for k, v in SYNONYMS.items():
          for syn in v:
            key_map[syn] = k"""

if target2 not in content:
    print("Failed to find target2")
content = content.replace(target2, replacement2)

with open("src/maxtext/experimental/agent/ckpt_validation_pipeline/forward_pass_validator.py", "w") as f:
    f.write(content)
print("done")
