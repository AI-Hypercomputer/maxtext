# MTP + CP 端到端测试复现指南

## 环境

- TPU VM: v6e-4 (2x2 topology, 4 chips), v2-alpha-tpuv6e runtime
- Zone: us-east5-b
- OS: Ubuntu 22.04.5 LTS
- 创建命令: `gcloud compute tpus tpu-vm create <name> --zone=us-east5-b --accelerator-type=v6e-4 --version=v2-alpha-tpuv6e`

## 依赖安装

```bash
# 1. Python 3.12（TPU VM 默认只有 3.10/3.11）
sudo apt-get update -qq
sudo apt-get install -y -qq python3.12 python3.12-dev python3.12-venv

# 2. Clone maxtext
git clone https://github.com/AI-Hypercomputer/maxtext.git ~/maxtext
cd ~/maxtext
git checkout -b feat/mtp-packing-cp-fix f835ffb38

# 3. Apply MTP CP-aware rolling patch + MTP loss fix
#    将本地的 mtp_cp_fix.patch 写入 ~/mtp_cp_fix.patch，然后:
git apply ~/mtp_cp_fix.patch
#    再将本地的 train.py 覆盖到远程:
#    scp src/MaxText/trainers/pre_train/train.py tpu-chiaoant:~/maxtext/src/maxtext/trainers/pre_train/train.py

# 4. 修改 deepseek-custom.yml 关掉 mHC/indexer/engram（和 MTP 不兼容）
python3 -c "
import yaml
with open('src/maxtext/configs/models/deepseek-custom.yml') as f:
    cfg = yaml.safe_load(f)
cfg['use_indexer'] = False
cfg['engram_layers'] = []
cfg['mhc_expansion_rate'] = 1
with open('src/maxtext/configs/models/deepseek-custom.yml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"

# 5. 创建 Python 3.12 venv + 安装依赖
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip packaging
pip install -e .
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install "maxtext[tpu]"
```

## 测试命令

### 测试 1: MTP + packing（单 chip, 无 CP）

验证 segment-aware roll 正确性：

```bash
source ~/maxtext/.venv/bin/activate
cd ~/maxtext

python3 -m maxtext.trainers.pre_train.train \
  src/maxtext/configs/base.yml \
  model_name=deepseek-custom \
  mtp_num_layers=2 \
  packing=true \
  max_segments_per_seq=4 \
  dataset_type=synthetic \
  max_target_length=2048 \
  per_device_batch_size=2 \
  steps=5 \
  run_name=mtp_packing_test \
  enable_checkpointing=false \
  async_checkpointing=false \
  scan_layers=false \
  attention=flash
```

预期: 5 steps 正常完成，loss 下降，无 crash/NaN。

### 测试 2: MTP + CP=2（多 chip, 无 packing）

验证 CP-aware `_shift_left_one_cp_aware` 跨 rank 正确性：

```bash
source ~/maxtext/.venv/bin/activate
cd ~/maxtext

python3 -m maxtext.trainers.pre_train.train \
  src/maxtext/configs/base.yml \
  model_name=deepseek-custom \
  mtp_num_layers=2 \
  packing=false \
  dataset_type=synthetic \
  max_target_length=2048 \
  per_device_batch_size=2 \
  steps=5 \
  run_name=mtp_cp_test \
  enable_checkpointing=false \
  async_checkpointing=false \
  scan_layers=false \
  attention=flash \
  ici_context_parallelism=2 \
  context_parallel_strategy=all_gather \
  context_parallel_load_balance=false
```

预期: 5 steps 正常完成，4 个 TPU 设备都参与（mesh shape 包含 context=2），loss 下降。

## MTP loss 修复

MaxText 上游存在一个 bug：`train.py` 第 203-211 行对 `mtp_losses` 的 `nnx.pop` 重复执行，导致 MTP loss 始终为零。

根因：`mtp_losses` 和 `mtp_acceptance` 是 `nnx.Intermediate` 的子类，第 200 行 `nnx.pop(nnx.Intermediate)` 已经捕获了它们。第 210-211 行再次 `nnx.pop(mtp_losses)` 拿到空 dict，覆盖了正确数据。

修复：将这 9 行替换为从 `intermediate_outputs["mtp_block"]` 中提取并重组：

```python
# MTP sows mtp_losses/mtp_acceptance as custom Variable subclasses of
# Intermediate, already captured by nnx.pop(nnx.Intermediate) above.
# Restructure them under their collection-name keys so calculate_mtp_loss
# and calculate_mtp_acceptance_rate can find them at the expected paths.
if config.mtp_num_layers > 0:
  if "mtp_block" in intermediate_outputs:
    mtp_data = intermediate_outputs.pop("mtp_block")
    intermediate_outputs["mtp_losses"] = {
        "mtp_block": {k: v for k, v in mtp_data.items() if k in ("losses", "weights")}
    }
    intermediate_outputs["mtp_acceptance"] = {
        "mtp_block": {k: v for k, v in mtp_data.items() if k in ("mtp_preds", "mtp_mask")}
    }
```

## 注意事项

| 问题 | 原因 | 解决 |
|---|---|---|
| `packing + CP + synthetic` 被拒绝 | 上游 train_utils.py:262 校验 | 测试 1 只测试 packing，测试 2 只测试 CP |
| deepseek-custom 的 indexer/engram 需要 tokenizer | MLA+indexer 依赖 HF tokenizer 做 sparse attn | 关掉 indexer/engram 用纯 MLA |
| mHC 与 MTP 不兼容 | MTP 输出 [B,T,E] 但 mHC 期望 4D | mhc_expansion_rate=1 关掉 mHC |
| Python 3.10 不支持 MaxText | MaxText requires >= 3.12 | `apt install python3.12` + venv |

## 测试结果 (2026-07-17)

### 测试 1: MTP + packing（单 chip）
```
completed step: 0, loss: 13.490, mtp_loss: 1.227, main_model_loss: 12.262
completed step: 1, loss: 13.377, mtp_loss: 1.224, main_model_loss: 12.153
completed step: 2, loss: 13.280, mtp_loss: 1.221, main_model_loss: 12.059
completed step: 3, loss: 13.218, mtp_loss: 1.219, main_model_loss: 11.999
completed step: 4, loss: 13.193, mtp_loss: 1.218, main_model_loss: 11.975
```

### 测试 2: MTP + CP=2（4 chips）
```
completed step: 0, loss: 13.490, mtp_loss: 1.227, main_model_loss: 12.262
completed step: 1, loss: 13.377, mtp_loss: 1.224, main_model_loss: 12.153
completed step: 2, loss: 13.280, mtp_loss: 1.221, main_model_loss: 12.059
completed step: 3, loss: 13.219, mtp_loss: 1.219, main_model_loss: 12.000
completed step: 4, loss: 13.193, mtp_loss: 1.218, main_model_loss: 11.975
```

两组 loss 曲线完全一致（mtp_loss 差异在浮点精度内），说明 CP 分片下的 `_shift_left_one_cp_aware` 产生了与单 chip 等价的结果。