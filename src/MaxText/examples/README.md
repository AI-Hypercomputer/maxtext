# MaxText Examples - Setting the Jupyter Lab or Collab to run them on TPU

This guide provides comprehensive instructions for setting up Jupyter Lab on TPU and connecting it to Google Colab for running MaxText examples.

## 📑 Table of Contents

- [Prerequisites](#prerequisites)
- [Method 1: Google Colab with TPU (Recommended)](#method-1-google-colab-with-tpu-recommended)
- [Method 2: Local Jupyter Lab with TPU](#method-2-local-jupyter-lab-with-tpu)
- [Method 3: Colab + Local Jupyter Lab Hybrid](#method-3-colab--local-jupyter-lab-hybrid)
- [Available Examples](#available-examples)
- [Common Pitfalls & Debugging](#common-pitfalls--debugging)
- [Support & Resources](#support--resources)
- [Contributing](#contributing)

## Prerequisites

Before starting, make sure you have:

- ✅ A Google Cloud Platform (GCP) account with billing enabled
- ✅ TPU quota available in your region (check under IAM & Admin → Quotas)
- ✅ Basic familiarity with Jupyter, Python, and Git
- ✅ gcloud CLI installed locally if you plan to use Method 2 or 3
- ✅ Firewall rules open for port 8888 (Jupyter) if accessing directly

## Method 1: Google Colab with TPU (Recommended)

This is the fastest way to run MaxText without managing infrastructure.

### Step 1: Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Sign in → New Notebook

### Step 2: Enable TPU Runtime

1. **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** → **TPU**
3. Select TPU version:
   - **v5e-8** → recommended for most MaxText examples, but it's a paid option
   - **v5e-1** → free tier option (slower, but works for Qwen-0.6B demos)
4. Click **Save**

### Step 3: Upload & Prepare MaxText

Upload notebooks or mount your GitHub repo

> **Note:** In Colab, the repo root will usually be `/content/maxtext`

**Example:**
```python
!git clone https://github.com/AI-Hypercomputer/maxtext.git
%cd maxtext
```

### Step 4: Run Examples

1. Open `src/MaxText/examples/`
2. Try:
   - `sft_qwen3_demo.ipynb`
   - `sft_llama3_demo.ipynb`
   - `grpo_llama3_demo.ipynb`


> ⚡ **Tip:** If Colab disconnects, re-enable TPU and re-run setup cells. Save checkpoints to GCS or Drive.

## Method 2: Local Jupyter Lab with TPU

This method gives you more control and is better for long training runs.

### Step 1: Set Up TPU VM

In Google Cloud Console:

1. **Compute Engine** → **TPU** → **Create TPU Node**
2. Example config:
   - **Name:** `maxtext-tpu-node`
   - **TPU type:** `v5e-8` (or `v6p-8` for newer hardware)
   - **Runtime Version:** `tpu-ubuntu-alpha-*` (matches your VM image)

### Step 2: Connect to TPU VM

```bash
gcloud compute tpus tpu-vm ssh maxtext-tpu-node --zone=YOUR_ZONE
```

### Step 3: Install Dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-dev git -y
pip3 install jupyterlab
```

### Step 4: Start Jupyter Lab

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Copy the URL with token from terminal

### Step 5: Secure Access

#### Option A: SSH Tunnel (Recommended)

```bash
gcloud compute tpus tpu-vm ssh maxtext-tpu-node --zone=YOUR_ZONE -- -L 8888:localhost:8888
```

Then open → `http://localhost:8888`


## Method 3: Colab + Local Jupyter Lab Hybrid

Set up Jupyter Lab as in step 2.
Use the link for Jupyter Lab as a link for "Connect to a local runtime" in Collab - at the dropdown where you select the runtime.

## Available Examples

### Supervised Fine-Tuning (SFT)

- **`sft_qwen3_demo.ipynb`** → Qwen3-0.6B with ultrachat_200k
- **`sft_llama3_demo.ipynb`** → Llama3 with ultrachat_200k

### GRPO Training

- **`grpo_llama3_demo.ipynb`** → GRPO training on math dataset

## Common Pitfalls & Debugging

| Issue | Solution |
|-------|----------|
| ❌ TPU runtime mismatch | Check TPU runtime version matches VM image (`tpu-ubuntu-alpha-*`) |
| ❌ Colab disconnects | Save checkpoints to GCS or Drive regularly |
| ❌ "RESOURCE_EXHAUSTED" errors | Use smaller batch size or v5e-8 instead of v5e-1 |
| ❌ Firewall blocked | Ensure port 8888 open, or always use SSH tunneling |
| ❌ Path confusion | In Colab use `/content/maxtext`; in TPU VM use `~/maxtext` |

## Support and Resources

- 📘 [MaxText Documentation](https://github.com/AI-Hypercomputer/maxtext)
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