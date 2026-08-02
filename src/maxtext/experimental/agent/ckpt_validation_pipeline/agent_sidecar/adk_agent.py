import os
import subprocess
import logging
from pathlib import Path
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Helper to run local scripts
def _run_script(script_name: str, args: list[str]) -> str:
    script_path = Path(__file__).parent / "fixer" / "tools" / script_name
    cmd = ["python3", str(script_path)] + args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
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
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            return response.read().decode('utf-8')
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
    return _run_script("create_pull_request.py", [
        "--base", base_branch,
        "--fix_branch", fix_branch_name,
        "--message", commit_message
    ])

# --- VERIFIER TOOLS ---

def trigger_airflow_dag(branch_name: str, overrides: str = "") -> str:
    """Triggers the Airflow pipeline to verify the patched branch, optionally passing parameter overrides in conf (as a JSON string or key=val list)."""
    args = ["--branch", branch_name]
    if overrides:
        args.extend(["--overrides", overrides])
    return _run_script("trigger_airflow_dag.py", args)

def write_remediation_report(run_id: str, content: str) -> str:
    """Writes the final victory lap markdown report to the root of the project."""
    report_path = Path(__file__).resolve().parents[6] / f"remediation_report_{run_id}.md"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Report successfully written to {report_path}"
    except Exception as e:
        return f"Error writing report: {e}"

# --- AGENT EXECUTION LOOP ---

def run_agent_workflow(run_id: str, model_name: str, failure_log: str):
    """Executes the ADK native agent loop using Gemini."""
    logger.info(f"Starting native ADK agent workflow for run_id: {run_id}")
    
    # Initialize the GenAI client using Vertex AI (No API key needed, uses Cloud Run Service Account)
    client = genai.Client(
        vertexai=True, 
        project="tpu-prod-env-multipod", 
        location=os.environ.get("VERTEX_LOCATION", "global")
    )
    
    # List of all our defined tools
    adk_tools = [
        read_local_file,
        fetch_reference_code,
        run_shape_analysis,
        patch_file,
        run_linters,
        manage_github_branch,
        create_pull_request,
        trigger_airflow_dag,
        write_remediation_report
    ]
    
    system_instruction = (
        "You are the Overwatch Autonomous Agent. Your job is to debug and fix failures in the MaxText checkpoint "
        "validation pipeline. Follow these strict steps:\n"
        "1. Diagnose the issue using the failure log and read_local_file/fetch_reference_code/run_shape_analysis.\n"
        "2. Create a fix branch using manage_github_branch.\n"
        "3. Apply fixes using patch_file and ensure they pass run_linters.\n"
        "4. Create a PR using create_pull_request.\n"
        "5. Test the branch using trigger_airflow_dag. If the failure is a configuration or rematerialization conflict (e.g. remat_policy='full' with debug_tensors=True), pass parameter overrides via the overrides argument in trigger_airflow_dag (e.g., '{\"maxtext_overrides\": {\"remat_policy\": \"none\"}}').\n"
        "6. Write a final report using write_remediation_report and conclude the task.\n"
        "7. Autonomous Hardware Scaling: If a failure report shows an Out-Of-Memory (OOM) error or HBM allocation failure (ResourceExhaustedError), the model checkpoint (e.g. DeepSeek-671B) is too large for the current TPU cluster slice. Do not edit model math or sharding. Instead, autonomously scale up infrastructure by calling trigger_airflow_dag with a larger reserved TPU cluster: --cluster_name v5p-128-bodaborg-europe-west4-b --project_name cloud-tpu-multipod-dev --zone europe-west4-b."
    )
    
    # Defaulting to Gemini 3 Pro (preview) for advanced reasoning, with environment variable override support
    model_id = os.environ.get("OVERWATCH_MODEL_ID", "gemini-3.5-flash-lite")
    
    prompt = (
        f"Pipeline Run ID: {run_id}\n"
        f"Target Model: {model_name}\n"
        f"Failure Log:\n{failure_log}\n\n"
        "Begin your diagnosis and execute the necessary tools to resolve this issue."
    )
    
    # Start a chat session that will automatically execute tools and loop back to the model
    chat = client.chats.create(
        model=model_id,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=adk_tools,
            temperature=0.2,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=25
            ),
        )
    )
    
    # Send the initial prompt. The SDK will handle the tool call loop automatically in newer versions, 
    # or you can manually loop it. We rely on the ADK's chat interface.
    response = chat.send_message(prompt)
    logger.info("Agent workflow completed.")
    return response.text

if __name__ == "__main__":
    # Mock trigger for local testing
    print("Agent ready. To run as a Cloud Run Job, this should be invoked by the poller.")
