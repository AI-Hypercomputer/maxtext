import subprocess
import sys
from pathlib import Path

def main():
    """
    Installs extra dependencies specified in extra_deps.txt using uv.

    This script looks for 'extra_deps.txt' relative to its own location.
    It executes 'uv pip install -r <path_to_extra_deps.txt> --resolution=lowest'.
    """
    script_dir = Path(__file__).resolve().parent
    
    # Adjust this path if your extra_deps.txt is in a different location,
    # e.g., script_dir / "data" / "extra_deps_from_github.txt"
    extra_deps_file = script_dir / "extra_deps_from_github.txt"

    if not extra_deps_file.exists():
        print(f"Error: '{extra_deps_file}' not found.")
        print("Please ensure 'extra_deps.txt' is in the correct location relative to the script.")
        sys.exit(1)
    # Check if 'uv' is available in the system's PATH
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
    except FileNotFoundError:
        print("Error: 'uv' command not found.")
        subprocess.run(["pip", "install", "uv"], check=True)
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error checking uv version: {e}")
        print(f"Stderr: {e.stderr.decode()}")
        sys.exit(1)

    command = [
        sys.executable,  # Use the current Python executable's pip to ensure the correct environment
        "-m",
        "uv",
        "pip",
        "install",
        "-r",
        str(extra_deps_file),
        "--no-deps",
    ]

    print(f"Installing extra dependencies from '{extra_deps_file}' using uv...")
    print(f"Running command: {' '.join(command)}")

    try:
        # Run the command
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Extra dependencies installed successfully!")
        print("--- Output from uv ---")
        print(process.stdout)
        if process.stderr:
            print("--- Errors/Warnings from uv (if any) ---")
            print(process.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Failed to install extra dependencies.")
        print(f"Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}.")
        print("--- Stderr ---")
        print(e.stderr)
        print("--- Stdout ---")
        print(e.stdout)
        sys.exit(e.returncode)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
