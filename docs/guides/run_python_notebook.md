# Run MaxText Python Notebooks on TPUs

This guide provides clear, step-by-step instructions for getting started with python notebooks on the two most popular platforms: Google Colab and a local JupyterLab environment.

## 📑 Table of Contents

- [Prerequisites](#prerequisites)
- [Method 1: Google Colab with TPU](#method-1-google-colab-with-tpu)
- [Method 2: Visual Studio Code with TPU (Recommended)](#method-2-visual-studio-code-with-tpu-recommended)
- [Method 3: Local Jupyter Lab with TPU (Recommended)](#method-3-local-jupyter-lab-with-tpu-recommended)
- [Common Pitfalls & Debugging](#common-pitfalls--debugging)
- [Support & Resources](#support-and-resources)
- [Contributing](#contributing)

## Prerequisites

Before starting, make sure you have:

- ✅ Basic familiarity with Jupyter, Python, and Git

**For Method 2 (Visual Studio Code) and Method 3 (Local Jupyter Lab) only:**

- ✅ A Google Cloud Platform (GCP) account with billing enabled
- ✅ TPU quota available in your region (check under IAM & Admin → Quotas)
- ✅ `tpu.nodes.create` permission to create a TPU VM
- ✅ gcloud CLI installed locally
- ✅ Firewall rules open for port 8888 (Jupyter) if accessing directly

## Method 1: Google Colab with TPU

This is the fastest way to run MaxText python notebooks without managing infrastructure.

**⚠️ IMPORTANT NOTE ⚠️**
The free tier of Google Colab provides access to `v5e-1 TPU`, but this access is not guaranteed and is subject to availability and usage limits.

Currently, this method only supports the **`sft_qwen3_demo.ipynb`** notebook, which demonstrates Qwen3-0.6B SFT training and evaluation on [OpenAI's GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k). If you want to run other notebooks, please use the local Jupyter Lab setup method.

Before proceeding, please verify that the specific notebook you are running works reliably on the free-tier TPU resources. If you encounter frequent disconnections or resource limitations, you may need to:

- Upgrade to a Colab Pro or Pro+ subscription for more stable and powerful TPU access.

- Move to local Jupyter Lab setup method with access to a powerful TPU machine.

### Step 1: Choose an Example

1.a. Visit the [MaxText examples directory](https://github.com/AI-Hypercomputer/maxtext/tree/main/src/MaxText/examples) on Github.

1.b. Find the notebook you want to run (e.g., `sft_qwen3_demo.ipynb`) and copy its URL.

### Step 2: Import into Colab

2.a. Go to [Google Colab](https://colab.research.google.com/) and sign in.

2.b. Select **File** -> **Open Notebook**.

2.c. Select the **GitHub** tab.

2.d. Paste the target `.ipynb` link you copied in step 1.b and press Enter.

### Step 3: Enable TPU Runtime

3.a. **Runtime** → **Change runtime type**

3.b. Select your desired **TPU** under **Hardware accelerator**

3.c. Click **Save**

### Step 4: Run the Notebook

Follow the instructions within the notebook cells to install dependencies and run the training/inference.

## Method 2: Visual Studio Code with TPU (Recommended)

Running Jupyter notebooks in Visual Studio Code (VS Code) provides a powerful, interactive environment that combines the flexibility of notebooks with the robust features of a code editor. Follow these steps to get your environment up and running.

### Step 1: Set Up TPU VM

In Google Cloud Console, create a standalone TPU VM:

1.a. **Compute Engine** → **TPUs** → **Create TPU**

1.b. Example config:

- **Name:** `maxtext-tpu-node`
- **TPU type:** Choose your desired TPU type
- **Runtime Version:** `tpu-ubuntu2204-base` (or other compatible runtime)

### Step 2: SSH to TPU-VM via VS Code

2.a. Install the [Remote - SSH extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) in VS Code.

2.b. Follow [Connect to a remote host](https://code.visualstudio.com/docs/remote/ssh#_connect-to-a-remote-host) guide to connect to your TPU-VM via VS Code.

### Step 3. Install Necessary Extensions on VS Code

To enable notebook support, you must install two official extensions from the VS Code Marketplace:

- Python Extension: Provides support for the Python language.

- Jupyter Extension: Enables you to create, edit, and run `.ipynb` files directly inside VS Code.

To install, click the `Extensions` icon on the left sidebar (or press `Ctrl+Shift+X` or `Cmd+Shift+X`), search for `Jupyter` and `Python`, and click `Install`.

### Step 4: Install MaxText and Dependencies

To execute post-training notebooks on your TPU-VM, follow the official [MaxText installation guides](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/rl.html#create-virtual-environment-and-install-maxtext-dependencies) to install MaxText and its dependencies inside a dedicated virtual environment.

### Step 5: Install the necessary library for Jupyter

Jupyter requires a kernel to execute code. This kernel is tied to a specific Python environment. Open your terminal inside VS Code and run:

```bash
uv pip install ipykernel
```

### Step 6: Select Kernel in VS Code & Run Notebook

Before you can run the notebook, you must tell VS Code which Python environment to use.

1. Look at the top-right corner of the notebook editor.
1. Click `Select Kernel`.
1. Choose Python Environments and select the virtual environment you created in Step 4.
1. Open [available post-training notebooks in MaxText](#available-examples) inside VS Code and run the jupyter notebook cells.

## Method 3: Local Jupyter Lab with TPU (Recommended)

You can run all of our Python notebooks on a local JupyterLab environment, giving you full control over your computing resources.

### Step 1: Set Up TPU VM

In Google Cloud Console, create a standalone TPU VM:

1.a. **Compute Engine** → **TPUs** → **Create TPU**

1.b. Example config:

- **Name:** `maxtext-tpu-node`
- **TPU type:** Choose your desired TPU type
- **Runtime Version:** `tpu-ubuntu2204-base` (or other compatible runtime)

### Step 2: Connect with Port Forwarding

Run the following command on your local machine:

> **Note**: The `--` separator before the `-L` flag is required. This tunnels the remote port 8888 to your local machine securely.

```bash
gcloud compute tpus tpu-vm ssh maxtext-tpu-node --zone=YOUR_ZONE -- -L 8888:localhost:8888
```

> **Note**: If you get a "bind: Address already in use" error, it means port 8888 is busy on your local computer. Change the first number to a different port, e.g., -L 9999:localhost:8888. You will then access Jupyter at localhost:9999.

### Step 3: Install JupyterLab

Run the following commands on your TPU-VM:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-dev git -y
pip3 install jupyterlab
```

### Step 4: Install MaxText and Dependencies

To execute post-training notebooks on your TPU-VM, follow the official [MaxText installation guides](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/rl.html#create-virtual-environment-and-install-maxtext-dependencies) to install MaxText and its dependencies inside a dedicated virtual environment.

### Step 5: Register virtual environment as a Jupyter Kernel

Once the environment is set up, you need to register it so JupyterLab can see it as an available kernel. Run the following command on your TPU-VM, replacing <virtual env name> with the actual name to your virtual environment:

```bash
python3 -m ipykernel install --user --name <virtual env name> --display-name "Python3 (MaxText Venv)"
```

### Step 6: Start JupyterLab

Start the Jupyter server on your TPU-VM by running:

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### Step 7: Access the Notebook

7.a. Look at the terminal output for a URL that looks like: `http://127.0.0.1:8888/lab?token=...`.

7.b. Copy that URL.

7.c. Paste it into your **local computer's browser**.

- **Important:** If you changed the port in Step 2 (e.g., to `9999`), you must manually replace `8888` in the URL with `9999`.
- *Example:* `http://127.0.0.1:9999/lab?token=...`

7.d. Once the interface opens in your browser, Click on the current kernel name (e.g., `Python 3 (ipykernel)`).

7.e. In the dropdown menu, select the new kernel you just created: `Python3 (MaxText Venv)`.

7.f. Open [available post-training notebooks in MaxText](#available-examples) and run the jupyter notebook cells.

## Available Examples

### Supervised Fine-Tuning (SFT)

- **`sft_qwen3_demo.ipynb`** → Qwen3-0.6B SFT training and evaluation on [OpenAI's GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k). This notebook is friendly for beginners and runs successfully on Google Colab's free-tier v5e-1 TPU runtime.
- **`sft_llama3_demo.ipynb`** → Llama3.1-8B SFT training on [Hugging Face ultrachat_200k dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k). We recommend running this on a v5p-8 TPU VM using the port-forwarding method.

### Reinforcement Learning (GRPO/GSPO) Training

- **`rl_llama3_demo.ipynb`** → GRPO/GSPO training on [OpenAI's GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k). We recommend running this on a v5p-8 TPU VM using the port-forwarding method.

## Common Pitfalls & Debugging

| Issue                          | Solution                                                   |
| ------------------------------ | ---------------------------------------------------------- |
| ❌ TPU runtime mismatch        | Check TPU runtime version matches VM image                 |
| ❌ Colab disconnects           | Save checkpoints to GCS or Drive regularly                 |
| ❌ "RESOURCE_EXHAUSTED" errors | Use smaller batch size or v5e-8 instead of v5e-1           |
| ❌ Firewall blocked            | Ensure port 8888 open, or always use SSH tunneling         |
| ❌ Path confusion              | In Colab use `/content/maxtext`; in TPU VM use `~/maxtext` |

## Support and Resources

- 📘 [MaxText Documentation](https://maxtext.readthedocs.io/)
- 💻 [Google Colab](https://colab.research.google.com)
- ⚡ [Cloud TPU Docs](https://cloud.google.com/tpu/docs)
- 🧩 [Jupyter Lab](https://jupyterlab.readthedocs.io)

## Contributing

If you encounter issues or have improvements for this guide, please:

1. Open an issue on the MaxText repository
1. Submit a pull request with your improvements
1. Share your experience in the discussions

______________________________________________________________________

**Happy Training! 🚀**
