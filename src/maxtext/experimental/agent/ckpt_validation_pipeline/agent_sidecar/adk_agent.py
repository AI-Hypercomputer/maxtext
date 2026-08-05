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
      err_str = str(e)
      if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
        if attempt < max_retries:
          logger.warning(
              f"Received 429 rate-limit error (attempt {attempt}/{max_retries}). Sleeping {sleep_seconds}s before retry..."
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


def read_local_file(filepath: str) -> str:
  """Reads a Python file from the local MaxText repository to inspect the architecture."""
  try:
    with open(filepath, "r", encoding="utf-8") as f:
      return f.read()
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


def patch_file(filepath: str, old_text: str, new_text: str) -> str:
  """Replaces specific lines of code in a file. Must provide the exact old text to replace."""
  try:
    with open(filepath, "r", encoding="utf-8") as f:
      content = f.read()

    if old_text not in content:
      return f"Error: old_text not found in {filepath}. Ensure exact match including whitespace."

    content = content.replace(old_text, new_text)

    with open(filepath, "w", encoding="utf-8") as f:
      f.write(content)
    return f"Successfully patched {filepath}."
  except Exception as e:
    return f"Error patching file {filepath}: {e}"


def run_linters(filepath: str) -> str:
  """Runs pyink (indentation=2, length=122) and pylint on the modified file to enforce standards."""
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


def trigger_airflow_dag(branch_name: str, overrides: str = "", dag_id: str = "") -> str:
  """Triggers the Airflow pipeline (specific sub-DAG or master DAG) to verify the patched branch, optionally passing parameter overrides in conf and a specific dag_id."""
  args = ["--branch", branch_name]
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
    return f"Report successfully written to {report_path}"
  except Exception as e:
    return f"Error writing report: {e}"


def clear_failed_airflow_task(dag_id: str, task_id: str, new_branch: str = "", run_id: str = "") -> str:
  """Clears a failed Airflow task instance to resume an existing DAG run after patching a Python file (Level 2)."""
  if new_branch and run_id:
    var_cmd = [
        "gcloud",
        "composer",
        "environments",
        "run",
        "ml-auto-solutions",
        "--location",
        "us-central2",
        "variables",
        "set",
        f"OVERRIDE_BRANCH_{run_id}",
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
      "us-central2",
      "tasks",
      "clear",
      dag_id,
      "-t",
      task_id,
      "-y",
  ]
  logger.info(f"Clearing Airflow task: {cmd}")
  try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return f"Successfully cleared task {task_id} on DAG {dag_id} (Override Branch: {new_branch}).\n{result.stdout}"
  except Exception as e:
    return f"Error clearing Airflow task: {e}"


def send_alert_email(subject: str, body: str, recipient: str = "") -> str:
  """Executes send_email.py to alert the engineering team of remediation status or failures."""
  args = ["--subject", subject, "--body", body]
  if recipient:
    args.extend(["--recipient", recipient])
  return _run_script("send_email.py", args)


# --- AGENT EXECUTION LOOP ---


def _load_prompt_file(filename: str) -> str:
  """Loads a prompt template from fixer/prompts/ directory."""
  prompt_path = Path(__file__).parent / "fixer" / "prompts" / filename
  try:
    with open(prompt_path, "r", encoding="utf-8") as f:
      return f.read()
  except Exception as e:
    logger.error(f"Error loading prompt file {filename}: {e}")
    return ""


def run_agent_workflow(context: dict, failure_log: str):
  """Executes the 4-Phase Meta-Agent Orchestrator loop using Gemini."""
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
          automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=15),
      ),
  )
  analyst_response = _send_message_with_retry(analyst_chat, analyst_prompt)

  # --- PHASE 2: REVIEW PHASE (Meta-Agent Validation) ---
  logger.info("Phase 2: Reviewing Analyst JSON One-Pager plan...")
  plan_json = {}
  try:
    plan_json = json.loads(analyst_response.text)
    logger.info(f"Analyst diagnosis: {plan_json.get('diagnosis')}")
    # Phase 2 Hallucination Guard: Verify failing_file exists
    failing_file = plan_json.get("failing_file", "")
    if failing_file and not Path(failing_file).exists():
      raise ValueError(f"Unsafe diagnosis: failing_file does not exist: {failing_file}")
  except Exception as e:
    logger.error("Analyst returned invalid JSON: %s", e)
    raise ValueError("Unsafe to patch because Analyst output was not valid JSON") from e

  # --- PHASE 2.5: OVERSEER SURVEILLANCE LOOP (meta_agent.txt) ---
  overseer_instruction = ""
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
      patch_file,
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
          automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=20),
      ),
  )
  fixer_prompt = f"Execute the Analyst structured plan for run_id {run_id} and model {model_name}."
  fixer_response = _send_message_with_retry(fixer_chat, fixer_prompt)
  logger.info(f"Fixer Phase completed: {fixer_response.text[:200]}...")

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
        f"Fixer Output:\n{fixer_response.text[:2000]}"
    )
    audit_res = _send_message_with_retry(overseer_guard_chat, overseer_audit_prompt).text.strip()
    if "VALID_FIX" not in audit_res or "SyntaxError" in fixer_response.text:
      logger.warning(f"Overseer detected hallucination or syntax issue in Fixer output! Directing repair:\n{audit_res}")
      fixer_repair_prompt = f"Overseer Intervention: Repair your fix immediately based on this audit:\n{audit_res}\nRun run_linters after repairing."
      fixer_response = _send_message_with_retry(fixer_chat, fixer_repair_prompt)
      logger.info(f"Fixer Syntactic & Hallucination Repair completed: {fixer_response.text[:200]}...")
    else:
      logger.info("Overseer verified Fixer output: VALID_FIX.")
  except Exception as e:
    logger.warning(f"Overseer Fixer audit skipped ({e}). Proceeding to verification...")

  # --- PHASE 4: VERIFIER SUBAGENT (03_verify.txt) ---
  logger.info("Phase 4: Invoking Verifier subagent to trigger pipeline and write remediation report...")
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
      f"1. The newly created fix branch is exactly: '{new_branch}'.\n"
      f"Original DAG: '{airflow_dag_id}'; task: '{airflow_task_id}'; run: '{airflow_run_id}'.\n"
      f"Original overrides: {json.dumps(maxtext_overrides)}\n"
      f"Fixer result: {fixer_response.text[:4000]}\n"
      "2. For Level 2 (Python Code Patch): Call clear_failed_airflow_task to resume the existing DAG run and preserve upstream context.\n"
      f"3. For Level 1 (Config Override): Call trigger_airflow_dag passing branch_name='{new_branch}' and dag_id='{dag_id_to_trigger}'.\n"
      "4. Upon completion, you MUST call write_remediation_report to create remediation_report_{run_id}.md in the project root and send_alert_email to notify the team."
  )

  verifier_tools = [
      trigger_airflow_dag,
      wait_for_airflow_run,
      clear_failed_airflow_task,
      write_remediation_report,
      send_alert_email,
  ]
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
  logger.info("4-Phase agent interaction completed; Airflow terminal state must determine remediation success.")
  return verifier_response.text


if __name__ == "__main__":
  # Mock trigger for local testing
  print("Agent ready. To run as a Cloud Run Job, this should be invoked by the poller.")
