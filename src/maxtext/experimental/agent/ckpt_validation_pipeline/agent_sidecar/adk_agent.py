import os
import time
import subprocess
import logging
import json
from pathlib import Path
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _send_message_with_retry(chat, prompt, max_retries=3, sleep_seconds=30):
  """Sends a message to Gemini with retry and a 30-second sleep on 429 rate-limit/quota errors."""
  for attempt in range(1, max_retries + 1):
    try:
      return chat.send_message(prompt)
    except Exception as e:
      err_str = str(e).lower()
      retry_keywords = ["429", "resource_exhausted", "quota", "503", "unavailable", "500", "internal server error", "502", "bad gateway", "504", "gateway timeout"]
      if any(k in err_str for k in retry_keywords):
        if attempt < max_retries:
          logger.warning(
              f"Received API rate-limit/server error (attempt {attempt}/{max_retries}). Sleeping {sleep_seconds}s before retry..."
          )
          time.sleep(sleep_seconds)
          continue
      raise e


# Helper to run local scripts
def _run_script(script_name: str, args: list[str]) -> str:
  script_path = Path(__file__).parent / "fixer" / "tools" / script_name
  cmd = ["python3", str(script_path)] + args
  logger.info(f"Executing script tool: {script_name} {args}")
  try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    logger.info(f"Tool {script_name} output:\n{result.stdout}")
    return result.stdout
  except subprocess.CalledProcessError as e:
    logger.error(f"Error executing {script_name}:\n{e.stderr}\n{e.stdout}")
    return f"Error executing {script_name}:\n{e.stderr}\n{e.stdout}"


# --- ANALYST TOOLS ---

def _resolve_path(filepath: str) -> str:
  """Helper to resolve paths either absolutely or relative to the repo root."""
  path = Path(filepath)
  if path.is_absolute() or path.exists():
    return str(path)
  
  # Check relative to repo root (6 levels up: src/maxtext/experimental/agent/ckpt_validation_pipeline/agent_sidecar)
  repo_root = Path(__file__).resolve().parents[6]
  
  # Sometimes the agent provides 'maxtext/layers/...' and sometimes 'src/maxtext/layers/...'
  # We can check a few combinations if it doesn't exist directly.
  root_path = repo_root / path
  if root_path.exists():
    return str(root_path)
  if (repo_root / "src" / path).exists():
    return str(repo_root / "src" / path)
  
  return str(root_path)


def read_local_file(filepath: str) -> str:
  """Reads a Python file from the local MaxText repository. Output includes line numbers so you can use edit_file_lines."""
  filepath = _resolve_path(filepath)
  try:
    with open(filepath, "r", encoding="utf-8") as f:
      lines = f.readlines()
      return "".join(f"{i+1}: {line}" for i, line in enumerate(lines))
  except Exception as e:
    return f"Error reading file {filepath}: {e}"


def fetch_reference_code(url: str) -> str:
  """Fetches the raw text from the provided PyTorch reference URLs (e.g., from HuggingFace)."""
  import urllib.request

  try:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
      return response.read().decode("utf-8")
  except Exception as e:
    return f"Error fetching URL {url}: {e}"


def run_shape_analysis(model_name: str, run_id: str) -> str:
  """Runs the analyze_shapes.py script to quickly test mock tensors for Shape Mismatch errors."""
  return _run_script("analyze_shapes.py", ["--model", model_name, "--run_id", run_id])


# --- FIXER TOOLS ---


def edit_file_lines(filepath: str, start_line: int, end_line: int, new_content: str) -> str:
  """Replaces lines from start_line to end_line (1-indexed, inclusive) with new_content.
  If the file does not exist, it will be created.
  To append to the end of the file, use a start_line greater than the total number of lines.
  To insert without deleting, use start_line = end_line + 1.
  """
  filepath = _resolve_path(filepath)
  try:
    import os
    if os.path.exists(filepath):
      with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    else:
      lines = []
      os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    start_idx = max(0, start_line - 1)
    end_idx = max(0, end_line)
    
    if new_content and not new_content.endswith('\n'):
      new_content += '\n'
      
    new_lines = [line + '\n' if not line.endswith('\n') else line for line in new_content.splitlines()]
    
    if start_idx >= len(lines):
      lines.extend(new_lines)
    else:
      lines[start_idx:end_idx] = new_lines
      
    with open(filepath, "w", encoding="utf-8") as f:
      f.writelines(lines)
    return f"Successfully edited {filepath}."
  except Exception as e:
    return f"Error editing file {filepath}: {e}"


def run_linters(filepath: str) -> str:
  """Runs pyink (indentation=2, length=122) and pylint on the modified file to enforce standards."""
  filepath = _resolve_path(filepath)
  return _run_script("run_linters.py", ["--file", filepath])


def manage_github_branch(action: str, branch_name: str) -> str:
  """Creates or checks out a GitHub branch. Action should be 'create' or 'checkout'."""
  return _run_script("github_branch_manager.py", ["--action", action, "--branch", branch_name])


def create_pull_request(base_branch: str, fix_branch_name: str, commit_message: str) -> str:
  """Commits changes, pushes the branch, and uses the GitHub CLI to open a Pull Request."""
  return _run_script(
      "create_pull_request.py", ["--base", base_branch, "--fix_branch", fix_branch_name, "--message", commit_message]
  )


# --- VERIFIER TOOLS ---


def trigger_airflow_dag(branch: str, overrides: str = "", dag_id: str = "") -> str:
  """Triggers the Airflow pipeline (specific sub-DAG or master DAG) to verify the patched branch, optionally passing parameter overrides in conf and a specific dag_id."""
  args = ["--branch", branch]
  if overrides:
    args.extend(["--overrides", overrides])
  if dag_id:
    args.extend(["--dag_id", dag_id])
  return _run_script("trigger_airflow_dag.py", args)


def wait_for_airflow_run(dag_id: str, dag_run_id: str, timeout_seconds: int = 7200) -> str:
  """Waits for an exact Airflow DAG run to reach success or failure."""
  return _run_script(
      "wait_for_airflow_run.py",
      ["--dag_id", dag_id, "--dag_run_id", dag_run_id, "--timeout_seconds", str(timeout_seconds)],
  )


def write_remediation_report(run_id: str, content: str) -> str:
  """Writes the final victory lap markdown report to the root of the project."""
  report_path = Path(__file__).resolve().parents[6] / f"remediation_report_{run_id}.md"
  try:
    with open(report_path, "w", encoding="utf-8") as f:
      f.write(content)
      
    # Upload to the reports bucket so it persists after the Cloud Run job exits
    from google.cloud import storage
    gcs_bucket = os.environ.get("AGENT_TRIGGER_BUCKET", "maxtext-validation-agent-reports")
    if gcs_bucket.startswith("gs://"):
      gcs_bucket = gcs_bucket[5:]
      
    client = storage.Client()
    bucket = client.bucket(gcs_bucket)
    blob = bucket.blob(f"remediation_report_{run_id}.md")
    blob.upload_from_filename(str(report_path), content_type="text/markdown")
    
    return f"Report successfully written locally and uploaded to gs://{gcs_bucket}/remediation_report_{run_id}.md"
  except Exception as e:
    return f"Error writing/uploading report: {e}"


def clear_failed_airflow_task(dag_id: str, task_id: str, new_branch: str = "", base_run_name: str = "", logical_date: str = "") -> str:
  """Clears a failed Airflow task instance to resume an existing DAG run after patching a Python file (Level 2)."""
  if new_branch and base_run_name:
    var_cmd = [
        "gcloud",
        "composer",
        "environments",
        "run",
        "ml-auto-solutions",
        "--location",
        os.environ.get("COMPOSER_LOCATION", "us-central1"),
        "variables",
        "set",
        f"OVERRIDE_BRANCH_{base_run_name}",
        new_branch,
    ]
    logger.info(f"Setting override branch variable in Composer: {var_cmd}")
    try:
      subprocess.run(var_cmd, capture_output=True, text=True, check=True)
    except Exception as e:
      logger.warning(f"Failed to set override branch variable ({e}). Proceeding to clear task...")

  cmd = [
      "gcloud",
      "composer",
      "environments",
      "run",
      "ml-auto-solutions",
      "--location",
      os.environ.get("COMPOSER_LOCATION", "us-central1"),
      "tasks",
      "clear",
      dag_id,
      "-t",
      task_id,
      "-y",
  ]
  if logical_date:
    cmd.extend(["-s", logical_date, "-e", logical_date])
  logger.info(f"Clearing Airflow task: {cmd}")
  try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return f"Successfully cleared task {task_id} on DAG {dag_id} (Override Branch: {new_branch}).\n{result.stdout}"
  except Exception as e:
    return f"Error clearing Airflow task: {e}"


def send_alert_email(subject: str, body: str, recipient: str = "", attachment_path: str = "") -> str:
  """Executes send_email.py to alert the engineering team of remediation status or failures."""
  args = ["--subject", subject, "--body", body]
  recipient = recipient or os.environ.get("ALERT_RECIPIENT") or os.environ.get("USER_EMAIL") or ""
  if recipient:
    args.extend(["--recipient", recipient])
  else:
    # If no recipient is found, we must pass a dummy value because the argument is required
    args.extend(["--recipient", "overwatch-team@google.com"])
    
  if attachment_path:
    args.extend(["--attachment", attachment_path])
  
  script_path = Path(__file__).resolve().parents[1] / "send_email.py"
  cmd = ["python3", str(script_path)] + args
  try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return f"Successfully sent email alert: {result.stdout}"
  except subprocess.CalledProcessError as e:
    return f"Failed to send email alert: {e.stderr}"


# --- AGENT EXECUTION LOOP ---


def _load_prompt_file(filename: str) -> str:
  """Loads a prompt template from fixer/prompts/ directory."""
  prompt_path = Path(__file__).resolve().parent / "fixer" / "prompts" / filename
  try:
    with open(prompt_path, "r", encoding="utf-8") as f:
      return f.read()
  except Exception as e:
    logger.error(f"Error loading prompt file {filename}: {e}")
    return ""


def run_agent_workflow(context: dict, failure_log: str):
  """Executes the 4-Phase Meta-Agent Orchestrator loop using Gemini."""
  os.environ["ORIGINAL_DAG_CONF"] = json.dumps(context)
  run_id = context.get("remediation_key") or context.get("run_name", "unknown_run")
  model_name = context.get("maxtext_model_name", "unknown_model")
  report_source = context.get("report_source", "")
  logger.info("Starting 4-Phase workflow for remediation key: %s", run_id)

  # Check for manager 1B-token API key first
  api_key = os.environ.get("GEMINI_API_KEY")
  if api_key:
    logger.info("Initializing GenAI Client using GEMINI_API_KEY (1B token quota)...")
    client = genai.Client(api_key=api_key)
    model_id = os.environ.get("OVERWATCH_MODEL_ID", "gemini-3.1-pro-preview-customtools")
  else:
    logger.info("Initializing GenAI Client using Vertex AI default credentials...")
    client = genai.Client(
        vertexai=True, project="tpu-prod-env-multipod", location=os.environ.get("VERTEX_LOCATION", "global")
    )
    model_id = os.environ.get("OVERWATCH_MODEL_ID", "gemini-3.1-pro-preview-customtools")

  maxtext_branch = context.get("maxtext_branch") or os.environ.get("MAXTEXT_BRANCH", "main")
  hf_ref_code_url = context.get("hf_ref_code_url") or os.environ.get("HF_REF_CODE_URL", "")
  hf_config_url = context.get("hf_config_url") or os.environ.get("HF_CONFIG_URL", "")
  maxtext_overrides = context.get("maxtext_overrides", {})
  airflow_dag_id = context.get("airflow_dag_id") or os.environ.get("TARGET_DAG_ID", "")
  airflow_task_id = context.get("airflow_task_id", "")
  airflow_run_id = context.get("airflow_run_id", "")
  safe_run_id = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in run_id)[:80]
  new_branch = f"fix-validation-pipeline-{model_name}-{safe_run_id}"

  # --- PHASE 1: ANALYST SUBAGENT (01_diagnose.txt) ---
  logger.info("Phase 1: Invoking Analyst subagent to generate structured JSON One-Pager...")
  analyst_template = _load_prompt_file("01_diagnose.txt")
  analyst_prompt = analyst_template.format(
      model_name=model_name,
      run_id=run_id,
      maxtext_branch=maxtext_branch,
      hf_ref_code_url=hf_ref_code_url,
      hf_config_url=hf_config_url,
      failure_log=failure_log,
      maxtext_overrides=json.dumps(maxtext_overrides, indent=2),
      airflow_dag_id=airflow_dag_id,
      airflow_task_id=airflow_task_id,
      airflow_run_id=airflow_run_id,
  )


  analyst_tools = [read_local_file, fetch_reference_code, run_shape_analysis, send_alert_email]
  analyst_chat = client.chats.create(
      model=model_id,
      config=types.GenerateContentConfig(
          tools=analyst_tools,
          temperature=0.2,
          response_mime_type="application/json",
          automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=35),
      ),
  )
  analyst_response = _send_message_with_retry(analyst_chat, analyst_prompt)

  # --- PHASE 2: REVIEW PHASE (Meta-Agent Validation) ---
  logger.info("Phase 2: Reviewing Analyst JSON One-Pager plan...")
  plan_json = {}
  try:
    # LLMs frequently leak markdown blocks even with response_mime_type="application/json"
    raw_text = (analyst_response.text or "").strip()
    if raw_text.startswith("```json"):
      raw_text = raw_text[7:]
    if raw_text.startswith("```"):
      raw_text = raw_text[3:]
    if raw_text.endswith("```"):
      raw_text = raw_text[:-3]
    plan_json = json.loads(raw_text.strip())
    logger.info(f"Analyst diagnosis: {plan_json.get('diagnosis')}")
    failing_file = plan_json.get("failing_file", "")
    remediation_level = plan_json.get("remediation_level", "")
    if failing_file and remediation_level != "level_1_config" and not Path(failing_file).exists():
      logger.warning(f"Note: failing_file '{failing_file}' not found at exact root path; Fixer will locate.")
  except Exception as e:
    logger.error("Analyst returned invalid JSON: %s", e)
    raise ValueError("Unsafe to patch because Analyst output was not valid JSON") from e

  # --- PHASE 2.5: OVERSEER SURVEILLANCE LOOP (meta_agent.txt) ---
  overseer_instruction = ""
  run_state = {}
  try:
    from monitor.state_manager import get_run_state
    run_state = get_run_state(run_id)
    if run_state.get("retries", 0) >= 2 and len(run_state.get("attempts", [])) >= 2:
      logger.info("Phase 2.5: Invoking Overwatch Overseer (meta_agent.txt) to inspect recursive failure loop...")
      meta_template = _load_prompt_file("meta_agent.txt")
      overseer_chat = client.chats.create(
          model=model_id,
          config=types.GenerateContentConfig(
              system_instruction=meta_template,
              temperature=0.2,
          ),
      )
      overseer_prompt = (
          f"Analyze attempt history for run_id '{run_id}' and synthesize corrective instruction:\n"
          f"{json.dumps(run_state['attempts'], indent=2)}"
      )
      overseer_res = _send_message_with_retry(overseer_chat, overseer_prompt)
      overseer_instruction = f"\n- OVERSEER SURVEILLANCE INTERVENTION:\n  {overseer_res.text.strip()}\n"
      logger.info(f"Overseer intervention synthesized:\n{overseer_instruction}")
  except Exception as e:
    logger.warning(f"Overseer surveillance loop skipped ({e}). Proceeding with primary plan...")

  max_agent_calls = int(os.environ.get("MAX_AGENT_CALLS", "35"))

  remediation_level = plan_json.get("remediation_level", "level_2_code")
  config_overrides = plan_json.get("config_overrides", {})
  fixer_response_text = ""

  if remediation_level == "level_1_config":
    logger.info("Level 1 Config Repair detected. Short-circuiting Phase 3 code patching and git branch creation.")
    fixer_response_text = f"Level 1 Config Repair: Overrides identified = {json.dumps(config_overrides)}"
    new_branch = maxtext_branch
  else:
    # --- PHASE 3: FIXER SUBAGENT (02_patch.txt) ---
    logger.info("Phase 3: Invoking Fixer subagent to execute structured plan...")
    fixer_template = _load_prompt_file("02_patch.txt")
    fixer_system = (
        f"{fixer_template}\n\n"
        "META-AGENT STRICT CONSTRAINTS:\n"
        "- For forward pass or eval verification tasks, if the error is 'RuntimeError: Array has been deleted', DO NOT edit normalizations.py or any nnx code. Fix via maxtext_overrides (remat_policy=none).\n"
        "- For training verification tasks, if 'RuntimeError: Array has been deleted' occurs, report it as an upstream MaxText NNX framework issue.\n"
        f"- Target branch to fork from: '{maxtext_branch}'. Newly created fix branch will be '{new_branch}'.\n"
        f"{overseer_instruction}"
        f"- Here is the Analyst Structured One-Pager Plan:\n{json.dumps(plan_json, indent=2)}"
    )

    fixer_tools = [
        read_local_file,
        fetch_reference_code,
        run_shape_analysis,
        edit_file_lines,
        run_linters,
        manage_github_branch,
        create_pull_request,
    ]
    fixer_chat = client.chats.create(
        model=model_id,
        config=types.GenerateContentConfig(
            system_instruction=fixer_system,
            tools=fixer_tools,
            temperature=0.2,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=max_agent_calls),
        ),
    )
    fixer_prompt = f"Execute the Analyst structured plan for run_id {run_id} and model {model_name}."
    fixer_response = _send_message_with_retry(fixer_chat, fixer_prompt)
    fixer_response_text = fixer_response.text or ""
    logger.info(f"Fixer Phase completed: {fixer_response_text[:200]}...")

    # --- PHASE 3.5: OVERSEER OUTPUT & FIX HALLUCINATION GUARD (meta_agent.txt) ---
    logger.info("Phase 3.5: Overwatch Overseer inspecting Fixer output for hallucinations, API validity, and syntactic regressions...")
    try:
      meta_template = _load_prompt_file("meta_agent.txt")
      overseer_guard_chat = client.chats.create(
          model=model_id,
          config=types.GenerateContentConfig(
              system_instruction=meta_template,
              temperature=0.2,
          ),
      )
      overseer_audit_prompt = (
          f"Audit the following Fixer output against the original error log and Analyst plan.\n"
          f"1. Did the Fixer hallucinate non-existent JAX/NNX APIs or edit irrelevant files?\n"
          f"2. Are there syntactic regressions (pyink/pylint/SyntaxError)?\n"
          f"3. If valid, respond with 'VALID_FIX'. Otherwise, output a specific correction prompt for the Fixer.\n\n"
          f"Fixer Output:\n{fixer_response_text[:2000]}"
      )
      audit_res = _send_message_with_retry(overseer_guard_chat, overseer_audit_prompt).text.strip()
      if "VALID_FIX" not in audit_res or "SyntaxError" in fixer_response_text:
        logger.warning(f"Overseer detected hallucination or syntax issue in Fixer output! Directing repair:\n{audit_res}")
        fixer_repair_prompt = f"Overseer Intervention: Repair your fix immediately based on this audit:\n{audit_res}\nRun run_linters after repairing."
        fixer_response = _send_message_with_retry(fixer_chat, fixer_repair_prompt)
        fixer_response_text = fixer_response.text or ""
        logger.info(f"Fixer Syntactic & Hallucination Repair completed: {fixer_response_text[:200]}...")
      else:
        logger.info("Overseer verified Fixer output: VALID_FIX.")
    except Exception as e:
      logger.warning(f"Overseer Fixer audit skipped ({e}). Proceeding to verification...")

  # --- PHASE 4: VERIFIER SUBAGENT (03_verify.txt) ---
  logger.info("Phase 4: Invoking Verifier subagent to trigger pipeline and write remediation report...")
  base_run_name = context.get("dag_conf", {}).get("run_name", "default_run")
  verifier_template = _load_prompt_file("03_verify.txt")

  # Determine the correct sub-DAG ID based on report source filename
  dag_id_to_trigger = airflow_dag_id
  if report_source.endswith("_forward_pass.json"):
    dag_id_to_trigger = "dag_verify_forward_pass"
  elif report_source.endswith("_decoding.json"):
    dag_id_to_trigger = "dag_verify_decoding"
  elif report_source.endswith("_shape.json"):
    dag_id_to_trigger = "dag_verify_checkpoint_shape"
  elif report_source.endswith("_forward_compile.json"):
    dag_id_to_trigger = "dag_verify_forward_compile"

  verifier_system = (
      f"{verifier_template}\n\n"
      "EXPLICIT META-AGENT VERIFICATION INSTRUCTIONS:\n"
      f"1. Target branch: '{new_branch}'.\n"
      f"Original DAG: '{airflow_dag_id}'; task: '{airflow_task_id}'; run: '{airflow_run_id}'.\n"
      f"Attempt Information: You are on attempt {run_state.get('retries', 0) + 1} out of {int(os.environ.get('MAX_RETRIES', '25'))}.\n"
      f"Original overrides: {json.dumps(maxtext_overrides)}\n"
      f"Identified config_overrides: {json.dumps(config_overrides)}\n"
      f"Fixer result: {fixer_response_text[:4000]}\n"
      "2. For Level 2 (Python Code Patch): Call wrapped_clear_failed_airflow_task to resume the existing DAG run.\n"
      f"3. For Level 1 (Config Override): Call trigger_airflow_dag passing branch='{new_branch}', dag_id='{dag_id_to_trigger}', and overrides='{json.dumps(config_overrides)}'.\n"
      "4. You MUST capture the returned dag_id and dag_run_id, then call wait_for_airflow_run to wait for the execution to complete.\n"
      "5. Upon successful verification, call write_remediation_report. Then, call send_alert_email and pass the local path to the generated markdown file into the 'attachment_path' argument."
  )

  logical_date = context.get("airflow_logical_date", "")

  def wrapped_clear_failed_airflow_task(dag_id: str, task_id: str) -> str:
    """Clears a failed Airflow task instance to resume an existing DAG run after a Python Code Patch (Level 2)."""
    return clear_failed_airflow_task(dag_id, task_id, new_branch, base_run_name, logical_date)

  verifier_tools = [
      trigger_airflow_dag,
      wait_for_airflow_run,
      wrapped_clear_failed_airflow_task,
      write_remediation_report,
      send_alert_email,
  ]
  try:
    from monitor.state_manager import record_attempt
    record_attempt(
        run_id,
        status="verification_started",
        branch=new_branch,
        diagnosis=plan_json.get("diagnosis", ""),
        remediation_level=plan_json.get("remediation_level", "unknown"),
        airflow_dag_id=airflow_dag_id,
        airflow_task_id=airflow_task_id,
        airflow_run_id=airflow_run_id,
    )
  except ModuleNotFoundError:
    logger.warning("record_attempt skipped (No module named 'monitor.state_manager')")

  verifier_chat = client.chats.create(
      model=model_id,
      config=types.GenerateContentConfig(
          system_instruction=verifier_system,
          tools=verifier_tools,
          temperature=0.2,
          automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=10),
      ),
  )
  verifier_prompt = f"Verify branch '{new_branch}' for run_id '{run_id}' and write the final Remediation Report."
  verifier_response = _send_message_with_retry(verifier_chat, verifier_prompt)
  verifier_response_text = verifier_response.text or ""
  logger.info("4-Phase agent interaction completed; Airflow terminal state must determine remediation success.")
  return verifier_response_text


if __name__ == "__main__":
  print("Agent ready. To run as a Cloud Run Job, this should be invoked by the poller.")
