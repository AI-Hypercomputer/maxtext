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

# 1) pyconfig.py
old_pycfg = """    if isinstance(new_value, str) and new_value.lower() == "none":
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
new_pycfg = """    if isinstance(new_value, str) and new_value.lower() == "none":
      field_info = valid_fields.get(key)
      if field_info:
        ann = field_info.annotation
        import typing
        import types as python_types
        def _allows_none(annotation) -> bool:
          if annotation is None or annotation is type(None) or annotation is typing.Any:
            return True
          origin = typing.get_origin(annotation)
          if origin in (typing.Union, getattr(python_types, "UnionType", None)):
            return any(arg is type(None) or arg is None or arg is typing.Any for arg in typing.get_args(annotation))
          return False

        if _allows_none(ann):
          new_value = None
      else:
        new_value = None"""
replace_in_file("src/maxtext/configs/pyconfig.py", old_pycfg, new_pycfg)

# 2) wait_for_airflow_run.py
old_wait = """    response = requests.get(
        url, headers={"Authorization": f"Bearer {credentials.token}", "Accept": "application/json"}, timeout=30
    )
    if response.status_code != 200:
      raise RuntimeError(f"Airflow status failed ({response.status_code}): {response.text}")
    payload = response.json()
    state = str(payload.get("state", "")).lower()
    if state in TERMINAL_STATES:
      result = {"ok": state == "success", "dag_id": dag_id, "dag_run_id": dag_run_id, "state": state}
      print(json.dumps(result))
      return result"""
new_wait = """    try:
      response = requests.get(
          url, headers={"Authorization": f"Bearer {credentials.token}", "Accept": "application/json"}, timeout=30
      )
      if response.status_code != 200:
        if response.status_code in {502, 503, 504}:
          import time
          time.sleep(poll_seconds)
          continue
        raise RuntimeError(f"Airflow status failed ({response.status_code}): {response.text}")
      payload = response.json()
      state = str(payload.get("state", "")).lower()
      if state in TERMINAL_STATES:
        result = {"ok": state == "success", "dag_id": dag_id, "dag_run_id": dag_run_id, "state": state}
        print(json.dumps(result))
        return result
    except (requests.RequestException, Exception) as e:
      # Log warning but don't fail, allowing subsequent poll iterations to retry
      pass"""
replace_in_file("src/maxtext/experimental/agent/ckpt_validation_pipeline/agent_sidecar/fixer/tools/wait_for_airflow_run.py", old_wait, new_wait)

# 3) decode_validator.py
old_dec = """      else:
        returncode = proc.returncode
        stdout_str = "".join(stdout_lines)
        stderr_str = "Redirected to stdout\""""
new_dec = """      else:
        returncode = proc.returncode
        reader_thread.join(timeout=10)  # Ensure all stdout is read up to EOF
        stdout_str = "".join(stdout_lines)
        stderr_str = "Redirected to stdout\""""
replace_in_file("src/maxtext/experimental/agent/ckpt_validation_pipeline/decode_validator.py", old_dec, new_dec)

# 4) forward_pass_validator.py
old_fwp = """  import jax  # pylint: disable=import-outside-toplevel

  _orig_array_delete = getattr(jax.Array, "delete", None)
  if _orig_array_delete is not None:
    # WARNING: Suppressing array deletion can increase device memory pressure and risk TPU OOM
    jax.Array.delete = lambda self: None

  import transformers  # pylint: disable=import-outside-toplevel"""
new_fwp = """  import jax  # pylint: disable=import-outside-toplevel

  import transformers  # pylint: disable=import-outside-toplevel"""
replace_in_file("src/maxtext/experimental/agent/ckpt_validation_pipeline/forward_pass_validator.py", old_fwp, new_fwp)

# 5) send_email.py
old_email = """    except UnicodeDecodeError:
      logger.warning(f"Binary attachment detected at {attachment_path}. Only UTF-8 text files are currently supported.")

  data = json.dumps"""
new_email = """    except UnicodeDecodeError:
      logger.warning(f"Binary attachment detected at {attachment_path}. Only UTF-8 text files are currently supported.")
    except Exception as e:
      logger.warning(f"Failed to read attachment at {attachment_path}: {e}. Sending email without attachment.")

  data = json.dumps"""
replace_in_file("src/maxtext/experimental/agent/ckpt_validation_pipeline/send_email.py", old_email, new_email)

