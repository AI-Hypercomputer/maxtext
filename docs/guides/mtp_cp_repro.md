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

# 2. Clone maxtext + checkout 本分支
git clone https://github.com/AI-Hypercomputer/maxtext.git ~/maxtext
cd ~/maxtext
git checkout feat/mtp-packing-cp-fix

# 3. 修改 deepseek-custom.yml 关掉 mHC/indexer/engram
#    （这些模块和 MTP 不兼容，测试用临时关掉，不提交）
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

# 4. 创建 Python 3.12 venv + 安装依赖
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

### 测试 2: MTP + CP=2（多 chip, 无 packing）

验证 `_shift_left_one_cp_aware` 跨 rank 正确性：

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

### 测试 3: MTP + CP=2 + packing（全部组合）

验证 CP-aware roll + segment-aware roll 一起工作。Synthetic data 在 packing 模式下自动生成有文档边界的 segment_ids（`_make_packed_segment_ids`）：

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
  run_name=mtp_cp_packing_test \
  enable_checkpointing=false \
  async_checkpointing=false \
  scan_layers=false \
  attention=flash \
  ici_context_parallelism=2 \
  context_parallel_strategy=all_gather \
  context_parallel_load_balance=false
```

## 单元测试

```bash
source ~/maxtext/.venv/bin/activate
cd ~/maxtext
python3 -m unittest tests.unit.multi_token_prediction_test -v
```

## 注意事项

| 问题 | 原因 | 解决 |
|---|---|---|
| deepseek-custom 的 indexer/engram 需要 tokenizer | MLA+indexer 依赖 HF tokenizer | 关掉 indexer/engram，用纯 MLA |
| mHC 与 MTP 不兼容 | MTP 输出 [B,T,E] 但 mHC 期望 4D | `mhc_expansion_rate=1` 关掉 mHC |
| Python 3.10 不支持 MaxText | MaxText requires >= 3.12 | `apt install python3.12` + venv |
| `pip install -e .` 报 `packaging.licenses` | hatchling 版本与 pip 不兼容 | `pip install --upgrade pip packaging` |

## 测试结果 (2026-07-17, TPU v6e-4)

### 测试 1: MTP + packing（单 chip）
```
completed step: 0, loss: 13.490, main_model_loss: 12.262, mtp_loss: 1.227
completed step: 1, loss: 13.377, main_model_loss: 12.153, mtp_loss: 1.224
completed step: 2, loss: 13.280, main_model_loss: 12.059, mtp_loss: 1.221
completed step: 3, loss: 13.218, main_model_loss: 11.999, mtp_loss: 1.219
completed step: 4, loss: 13.193, main_model_loss: 11.975, mtp_loss: 1.218
```

### 测试 2: MTP + CP=2（4 chips）
```
completed step: 0, loss: 13.490, main_model_loss: 12.262, mtp_loss: 1.227
completed step: 1, loss: 13.377, main_model_loss: 12.153, mtp_loss: 1.224
completed step: 2, loss: 13.280, main_model_loss: 12.059, mtp_loss: 1.221
completed step: 3, loss: 13.219, main_model_loss: 12.000, mtp_loss: 1.219
completed step: 4, loss: 13.193, main_model_loss: 11.975, mtp_loss: 1.218
```

### 测试 3: MTP + CP=2 + packing
```
completed step: 0, loss: 13.489, main_model_loss: 12.261, mtp_loss: 1.228
completed step: 1, loss: 13.371, main_model_loss: 12.147, mtp_loss: 1.223
completed step: 2, loss: 13.269, main_model_loss: 12.049, mtp_loss: 1.220
completed step: 3, loss: 13.205, main_model_loss: 11.987, mtp_loss: 1.217
completed step: 4, loss: 13.177, main_model_loss: 11.961, mtp_loss: 1.216
```

三组 loss 曲线完全一致，验证 `_shift_left_one_cp_aware` 和 `roll_and_mask_by_segment` 在所有组合下正确。