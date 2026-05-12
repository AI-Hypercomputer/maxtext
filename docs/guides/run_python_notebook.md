# Run MaxText Python Notebooks on TPUs

This guide provides clear, step-by-step instructions for running MaxText Python notebooks on TPUs using three supported methods: Google Colab, Visual Studio Code, and a local JupyterLab environment.

## 📑 Table of Contents

- [Prerequisites](#prerequisites)
- [Method 1: Google Colab with TPU](#method-1-google-colab-with-tpu)
- [Method 2: Visual Studio Code with TPU (Recommended)](#method-2-visual-studio-code-with-tpu-recommended)
- [Method 3: Local Jupyter Lab with TPU (Recommended)](#method-3-local-jupyter-lab-with-tpu-recommended)
- [Common Pitfalls & Debugging](#common-pitfalls--debugging)
- [Support & Resources](#support-and-resources)

## Prerequisites

### Get Hugging Face Token

To access model checkpoint from the Hugging Face Hub, you need to authenticate with a personal access token. Follow these steps to get your token:

- **Navigate to the Access Tokens page** in your Hugging Face account settings. You can go there directly by visiting [this url](https://huggingface.co/settings/tokens).

- **Create a new token** by clicking the **"+ Create new token"** button.

- **Give your token a name** and assign it a **`read` role**. The `read` role is sufficient for downloading models.

## Method 1: Google Colab with TPU

This is the fastest way to run MaxText python notebooks without managing infrastructure.

**⚠️ IMPORTANT NOTE ⚠️**
The free tier of Google Colab provides access to `v5e-1 TPU`, but this access is not guaranteed and is subject to availability and usage limits.

Currently, this method only supports the **`sft_qwen3_demo.ipynb`** notebook, which demonstrates Qwen3-0.6B SFT training and evaluation on [OpenAI's GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k). If you want to run other notebooks, please use the other methods below.

Before proceeding, please verify that the specific notebook you are running works reliably on the free-tier TPU resources. If you encounter frequent disconnections or resource limitations, you may need to:

- Upgrade to a Colab Pro or Pro+ subscription for more stable and powerful TPU access.

- Try [Method 2](#method-2-visual-studio-code-with-tpu-recommended) or [Method 3](#method-3-local-jupyter-lab-with-tpu-recommended) with access to a more powerful TPU machine.

### Step 1: Import Notebook into Google Colab

- Visit the [MaxText examples directory](https://github.com/AI-Hypercomputer/maxtext/tree/main/src/maxtext/examples) on Github.

- Find the notebook you want to run (e.g., `sft_qwen3_demo.ipynb`) and copy its URL.

- Go to [Google Colab](https://colab.research.google.com/) and sign in.

- Go to **File** -> **Open Notebook**.

- Select the **GitHub** tab.

- Paste the target `.ipynb` link you copied and press Enter.

### Step 2: Enable TPU Runtime

- Go to **Runtime** → **Change runtime type**.

- Select your desired **TPU** under **Hardware accelerator**.

- Click **Save**.

### Step 3: Run the Notebook

Follow the instructions within the notebook cells to install dependencies and run the training/inference.

## Method 2: Visual Studio Code with TPU (Recommended)

Running Jupyter notebooks in Visual Studio Code (VS Code) provides a powerful, interactive environment that combines the flexibility of notebooks with the robust features of a code editor. Follow these steps to get your environment up and running.

### Step 1: SSH to TPU-VM via VS Code

- Install the [Remote - SSH extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) in VS Code.

- Follow [Connect to a remote host](https://code.visualstudio.com/docs/remote/ssh#_connect-to-a-remote-host) guide to connect to your TPU-VM via VS Code.

### Step 2: Install Necessary Extensions on VS Code

To enable notebook support, you must install two official extensions from the VS Code Marketplace:

- Python Extension: Provides support for the Python language.

- Jupyter Extension: Enables you to create, edit, and run `.ipynb` files directly inside VS Code.

To install, click the `Extensions` icon on the left sidebar (or press `Ctrl+Shift+X` or `Cmd+Shift+X`), search for `Jupyter` and `Python`, and click `Install`.

### Step 3: Install MaxText and Dependencies

To execute post-training notebooks on your TPU-VM, follow the official [MaxText installation guides](install-from-source) and specifically follow `Option 3: Installing [tpu-post-train]`. This will ensure all post-training dependencies are installed inside your virtual environment.

> **Note:** If you have previously installed MaxText with a different option (e.g., `maxtext[tpu]`), we strongly recommend using a fresh virtual environment for `maxtext[tpu-post-train]` to avoid potential library version conflicts.

Login to Hugging Face and provide your access token when prompted:

```bash
hf auth login
```

### Step 4: Install the necessary library for Jupyter

Jupyter requires a kernel to execute code. This kernel is tied to a specific Python environment. Open your terminal inside VS Code and run:

```bash
uv pip install ipykernel
```

### Step 5: Select Kernel in VS Code & Run Notebook

Before you can run the notebook, you must tell VS Code which Python environment to use.

1. Look at the top-right corner of the notebook editor.
2. Click `Select Kernel`.
3. Choose Python Environments and select the virtual environment you created in Step 3.
4. Open [available post-training notebooks in MaxText](#available-examples) inside VS Code and run the jupyter notebook cells.

## Method 3: Local Jupyter Lab with TPU (Recommended)

You can run all of our Python notebooks on a local JupyterLab environment, giving you full control over your computing resources.

### Step 1: SSH to TPU-VM with Port Forwarding

Run the following command on your local machine:

```bash
gcloud compute tpus tpu-vm ssh <TPU_VM_NAME> --zone=<ZONE> -- -L 8888:localhost:8888
```

> **Note**: If you get a "bind: Address already in use" error, it means port 8888 is busy on your local computer. Change the first number to a different port, e.g., -L 9999:localhost:8888. You will then access Jupyter at localhost:9999.

### Step 2: Install JupyterLab

Run the following commands on your TPU-VM:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-dev git -y
pip3 install jupyterlab
```

### Step 3: Install MaxText and Dependencies

To execute post-training notebooks on your TPU-VM, follow the official [MaxText installation guides](install-from-source) and specifically follow `Option 3: Installing [tpu-post-train]`. This will ensure all post-training dependencies are installed inside your virtual environment.

> **Note:** If you have previously installed MaxText with a different option (e.g., `maxtext[tpu]`), we strongly recommend using a fresh virtual environment for `maxtext[tpu-post-train]` to avoid potential library version conflicts.

Login to Hugging Face and provide your access token when prompted:

```bash
hf auth login
```

### Step 4: Register virtual environment as a Jupyter Kernel

Once the environment is set up, you need to register it so JupyterLab can see it as an available kernel. Run the following command on your TPU-VM:

```bash
python3 -m ipykernel install --user --name <VENV_NAME> --display-name "Python3 (MaxText Venv)"
```

### Step 5: Start JupyterLab

Start the Jupyter server on your TPU-VM by running:

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### Step 6: Access the Notebook

- Look at the terminal output for a URL that looks like: `http://127.0.0.1:8888/lab?token=...` and copy it.

- Paste it into your **local computer's browser**.

- **Important:** If you changed the port in Step 1 (e.g., to `9999`), you must manually replace `8888` in the URL with `9999`. Example: `http://127.0.0.1:9999/lab?token=...`.

- Once the interface opens in your browser, click on the current kernel name (e.g., `Python 3 (ipykernel)`).

- In the dropdown menu, select the new kernel you just created: `Python3 (MaxText Venv)`.

- Open [available post-training notebooks in MaxText](#available-examples) and run the jupyter notebook cells.

## Available Examples

### Supervised Fine-Tuning (SFT)

- **`sft_qwen3_demo.ipynb`** → Qwen3-0.6B SFT training and evaluation on [OpenAI's GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k). This notebook is friendly for beginners and runs successfully on Google Colab's free-tier v5e-1 TPU runtime.
- **`sft_llama3_demo_tpu.ipynb`** → Llama3.1-8B SFT training on [Hugging Face ultrachat_200k dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k). We recommend running this on a v5p-8 TPU VM using [Method 2](#method-2-visual-studio-code-with-tpu-recommended) or [Method 3](#method-3-local-jupyter-lab-with-tpu-recommended).

### Reinforcement Learning (GRPO/GSPO) Training

- **`rl_llama3_demo.ipynb`** → GRPO/GSPO training on [OpenAI's GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k). We recommend running this on a v5p-8 TPU VM using [Method 2](#method-2-visual-studio-code-with-tpu-recommended) or [Method 3](#method-3-local-jupyter-lab-with-tpu-recommended).

## Common Pitfalls & Debugging

| Issue                          | Solution                                                     |
| ------------------------------ | ------------------------------------------------------------ |
| ❌ Colab disconnects           | Save checkpoints to GCS regularly                            |
| ❌ "RESOURCE_EXHAUSTED" errors | Use smaller batch size or change BASE_OUTPUTDIRECTORY to GCS |
| ❌ Firewall blocked            | Ensure port 8888 open, or always use SSH tunneling           |

## Support and Resources

- 📘 [MaxText Documentation](https://maxtext.readthedocs.io/)
- 💻 [Google Colab](https://colab.research.google.com)
- ⚡ [Cloud TPU Docs](https://cloud.google.com/tpu/docs)
- 🧩 [Jupyter Lab](https://jupyterlab.readthedocs.io)
