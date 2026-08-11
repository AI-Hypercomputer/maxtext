import os

def replace_in_file(filepath, old, new):
    if not os.path.exists(filepath):
        print(f"Not found: {filepath}")
        return
    with open(filepath, "r") as f:
        content = f.read()
    if old in content:
        content = content.replace(old, new)
        with open(filepath, "w") as f:
            f.write(content)
            print(f"Patched {filepath}")
    else:
        print(f"Text not found in {filepath}!")

# 1) types.py - Deepseek fields
old_types_1 = """  routed_bias_update_rate: float = Field(0.0, description="Update rate applied to the router bias term.")
  mlp_bias: bool = Field("""
new_types_1 = """  routed_bias_update_rate: float = Field(0.0, description="Update rate applied to the router bias term.")
  log_moe_bias_norms: bool = Field(False, description="Whether to log the norms of MoE router biases.")
  mlp_bias: bool = Field("""
replace_in_file("src/maxtext/configs/types.py", old_types_1, new_types_1)

# 2) types.py - Deepseek rules
old_types_2 = """      if self.routed_bias and self.routed_bias_update_rate > 0.0 and self.decoder_block != DecoderBlockType.DEEPSEEK:
        raise ValueError("Loss-free load balancing is only supported for the DeepSeek decoder block.")"""
new_types_2 = """      if (
          self.routed_bias
          and self.routed_bias_update_rate > 0.0
          and self.decoder_block not in (DecoderBlockType.DEEPSEEK, DecoderBlockType.DEEPSEEK4)
      ):
        raise ValueError("Loss-free load balancing is only supported for the DeepSeek decoder block.")
      if not self.pure_nnx and self.routed_bias and self.decoder_block == DecoderBlockType.DEEPSEEK4:
        raise ValueError(
            "Auxiliary-loss-free routed bias for DeepSeek V4 is only supported in pure NNX mode. "
            "Please set pure_nnx=True or disable routed_bias."
        )"""
replace_in_file("src/maxtext/configs/types.py", old_types_2, new_types_2)

# 3) wait_for_airflow_run.py
old_wait = """  while time.monotonic() < deadline:
    credentials.refresh(request)
    response = requests.get("""
new_wait = """  while time.monotonic() < deadline:
    if not credentials.valid:
      credentials.refresh(request)
    response = requests.get("""
replace_in_file("src/maxtext/experimental/agent/ckpt_validation_pipeline/agent_sidecar/fixer/tools/wait_for_airflow_run.py", old_wait, new_wait)

# 4) forward_compile_validator.py
old_fw = """  except Exception as e:  # pylint: disable=broad-exception-caught"""
new_fw = """  except BaseException as e:  # pylint: disable=broad-exception-caught"""
replace_in_file("src/maxtext/experimental/agent/ckpt_validation_pipeline/forward_compile_validator.py", old_fw, new_fw)

# 5) checkpoint_shape_validator.py
old_shp = """      if "key:" in line and "|" in line:
        parts = line.split("|")"""
new_shp = """      if "key:" in line and "|" in line:
        parts = line.split("|", 1)"""
replace_in_file("src/maxtext/experimental/agent/ckpt_validation_pipeline/checkpoint_shape_validator.py", old_shp, new_shp)

# 6) Dockerfile
old_docker = """# Set git global identity for commits and associate local repo with origin/main history
RUN git config --global user.email "overwatch-agent@google.com" && \\
    git config --global user.name "Overwatch Agent" && \\
    git init -q && \\
    git remote add origin https://github.com/AI-Hypercomputer/maxtext.git && \\
    git fetch origin main --depth=1 && \\
    git checkout -b main && \\
    git reset origin/main"""
new_docker = """# Set git global identity for commits and associate local repo with origin/main history
RUN git config --global user.email "overwatch-agent@google.com" && \\
    git config --global user.name "Overwatch Agent" """
replace_in_file("src/maxtext/experimental/agent/ckpt_validation_pipeline/agent_sidecar/Dockerfile", old_docker, new_docker)


# 7) pyconfig.py
old_pycfg = """    if isinstance(new_value, str) and new_value.lower() == "none":
      field_info = valid_fields.get(key)
      if not (field_info and field_info.annotation is str):
        new_value = None"""
new_pycfg = """    if isinstance(new_value, str) and new_value.lower() == "none":
      field_info = valid_fields.get(key)
      if field_info:
        ann_str = str(field_info.annotation)
        # Convert to None ONLY if the field allows it or isn't exclusively a string union that forbids None
        if field_info.annotation is str:
          pass
        elif "str" in ann_str and "None" not in ann_str:
          pass
        else:
          new_value = None
      else:
        new_value = None"""
replace_in_file("src/maxtext/configs/pyconfig.py", old_pycfg, new_pycfg)

