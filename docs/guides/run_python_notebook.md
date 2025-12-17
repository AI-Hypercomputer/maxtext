# Run MaxText Python Notebooks on TPUs

This guide provides clear, step-by-step instructions for getting started with python notebooks on the two most popular platforms: Google Colab and a local JupyterLab environment.

## 📑 Table of Contents

- [Prerequisites](#prerequisites)
- [Method 1: Google Colab with TPU](#method-1-google-colab-with-tpu)
- [Method 2: Local Jupyter Lab with TPU (Recommended)](#method-2-local-jupyter-lab-with-tpu-recommended)
- [Common Pitfalls & Debugging](#common-pitfalls--debugging)
- [Support & Resources](#support-and-resources)
- [Contributing](#contributing)

## Prerequisites

Before starting, make sure you have:

- ✅ Basic familiarity with Jupyter, Python, and Git

**For Method 2 (Local Jupyter Lab) only:**
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

* Upgrade to a Colab Pro or Pro+ subscription for more stable and powerful TPU access.

* Move to local Jupyter Lab setup method with access to a powerful TPU machine.

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

## Method 2: Local Jupyter Lab with TPU (Recommended)

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

### Step 3: Install Dependencies

Run the following commands on your TPU-VM:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-dev git -y
pip3 install jupyterlab
```

### Step 4: Start Jupyter Lab

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### Step 5: Access the Notebook
5.a. Look at the terminal output for a URL that looks like: `http://127.0.0.1:8888/lab?token=...`

5.b. Copy that URL.

5.c. Paste it into your **local computer's browser**.
   * **Important:** If you changed the port in Step 2 (e.g., to `9999`), you must manually replace `8888` in the URL with `9999`.
   * *Example:* `http://127.0.0.1:9999/lab?token=...`


## Available Examples

### Supervised Fine-Tuning (SFT)

- **`sft_qwen3_demo.ipynb`** → Qwen3-0.6B SFT training and evaluation on [OpenAI's GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k). This notebook is friendly for beginners and runs successfully on Google Colab's free-tier v5e-1 TPU runtime.
- **`sft_llama3_demo.ipynb`** → Llama3.1-8B SFT training on [Hugging Face ultrachat_200k dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k). We recommend running this on a v5p-8 TPU VM using the port-forwarding method.

### Reinforcement Learning (GRPO/GSPO) Training

- **`rl_llama3_demo.ipynb`** → GRPO/GSPO training on [OpenAI's GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k). We recommend running this on a v5p-8 TPU VM using the port-forwarding method.

## Common Pitfalls & Debugging

| Issue | Solution |
|-------|----------|
| ❌ TPU runtime mismatch | Check TPU runtime version matches VM image |
| ❌ Colab disconnects | Save checkpoints to GCS or Drive regularly |
| ❌ "RESOURCE_EXHAUSTED" errors | Use smaller batch size or v5e-8 instead of v5e-1 |
| ❌ Firewall blocked | Ensure port 8888 open, or always use SSH tunneling |
| ❌ Path confusion | In Colab use `/content/maxtext`; in TPU VM use `~/maxtext` |

## Support and Resources

- 📘 [MaxText Documentation](https://maxtext.readthedocs.io/)
- 💻 [Google Colab](https://colab.research.google.com)
- ⚡ [Cloud TPU Docs](https://cloud.google.com/tpu/docs)
- 🧩 [Jupyter Lab](https://jupyterlab.readthedocs.io)

## Contributing

If you encounter issues or have improvements for this guide, please:

1. Open an issue on the MaxText repository
2. Submit a pull request with your improvements
3. Share your experience in the discussions

---

**Happy Training! 🚀**