# 使用本地模型运行 HiDream-E1

本指南说明如何将模型下载到本地并修改代码使用本地路径。

## 📦 需要下载的模型

总共需要 3 个模型，总大小约 **40-50 GB**：

1. **meta-llama/Llama-3.1-8B-Instruct** (~16 GB)
2. **HiDream-ai/HiDream-I1-Full** (~15 GB)
3. **HiDream-ai/HiDream-E1-1** (~15 GB)

---

## 📥 第一步：下载模型到本地

### 方法 A: 使用 huggingface-cli（推荐）

在**有网络的机器**上（如你的 Mac）：

```bash
# 1. 安装 huggingface-cli
pip install -U huggingface_hub

# 2. 登录（需要先在网页同意 Llama 协议）
huggingface-cli login

# 3. 创建模型存储目录
mkdir -p ~/Downloads/hidream_models

# 4. 下载 Llama-3.1-8B
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir ~/Downloads/hidream_models/Llama-3.1-8B-Instruct

# 5. 下载 HiDream-I1-Full
huggingface-cli download HiDream-ai/HiDream-I1-Full \
    --local-dir ~/Downloads/hidream_models/HiDream-I1-Full

# 6. 下载 HiDream-E1.1
huggingface-cli download HiDream-ai/HiDream-E1-1 \
    --local-dir ~/Downloads/hidream_models/HiDream-E1-1
```

### 方法 B: 使用 Python 脚本

创建 `download_models.py`：

```python
from huggingface_hub import snapshot_download
import os

# 模型保存目录
save_dir = os.path.expanduser("~/Downloads/hidream_models")
os.makedirs(save_dir, exist_ok=True)

models = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "HiDream-ai/HiDream-I1-Full",
    "HiDream-ai/HiDream-E1-1",
]

for model_id in models:
    print(f"\n{'='*60}")
    print(f"下载: {model_id}")
    print(f"{'='*60}")

    model_name = model_id.split("/")[-1]
    local_path = os.path.join(save_dir, model_name)

    snapshot_download(
        repo_id=model_id,
        local_dir=local_path,
        local_dir_use_symlinks=False,
        resume_download=True,  # 支持断点续传
    )

    print(f"✓ {model_id} 已下载到: {local_path}")

print(f"\n✓ 所有模型已下载到: {save_dir}")
```

运行：
```bash
python download_models.py
```

---

## 📤 第二步：上传到服务器

```bash
# 压缩模型（可选，节省传输时间）
cd ~/Downloads
tar -czf hidream_models.tar.gz hidream_models/

# 上传到服务器
scp hidream_models.tar.gz user@server:/home/ma-user/work/jiangtianyi/

# SSH 登录服务器解压
ssh user@server
cd /home/ma-user/work/jiangtianyi/
tar -xzf hidream_models.tar.gz
```

或者不压缩直接传（时间较长）：
```bash
rsync -avz --progress \
    ~/Downloads/hidream_models/ \
    user@server:/home/ma-user/work/jiangtianyi/hidream_models/
```

---

## 🔧 第三步：修改代码

### 方法 A：创建配置文件（推荐）

在服务器上创建 `local_config.py`：

```python
"""
本地模型路径配置
"""
import os

# ========== 配置你的模型路径 ==========
MODEL_BASE_DIR = "/home/ma-user/work/jiangtianyi/hidream_models"

# 模型路径
LLAMA_PATH = os.path.join(MODEL_BASE_DIR, "Llama-3.1-8B-Instruct")
HIDREAM_I1_PATH = os.path.join(MODEL_BASE_DIR, "HiDream-I1-Full")
HIDREAM_E1_PATH = os.path.join(MODEL_BASE_DIR, "HiDream-E1-1")

# 验证路径是否存在
def verify_models():
    """检查所有模型是否存在"""
    models = {
        "Llama-3.1-8B": LLAMA_PATH,
        "HiDream-I1-Full": HIDREAM_I1_PATH,
        "HiDream-E1-1": HIDREAM_E1_PATH,
    }

    all_exist = True
    for name, path in models.items():
        if os.path.exists(path):
            print(f"✓ {name:20s} -> {path}")
        else:
            print(f"✗ {name:20s} -> {path} [不存在]")
            all_exist = False

    return all_exist

if __name__ == "__main__":
    print("检查本地模型:")
    print("="*60)
    if verify_models():
        print("="*60)
        print("✓ 所有模型路径正确")
    else:
        print("="*60)
        print("✗ 部分模型未找到，请检查路径")
```

验证配置：
```bash
python local_config.py
```

### 方法 B：直接修改推理脚本

编辑 `inference_e1_1_npu.py`，找到第 20-23 行：

```python
# 原代码：
# LLAMA_PATH = "meta-llama/Llama-3.1-8B-Instruct"
# HIDREAM_I1_PATH = "HiDream-ai/HiDream-I1-Full"
# HIDREAM_E1_PATH = "HiDream-ai/HiDream-E1-1"

# 改为：
LLAMA_PATH = "/home/ma-user/work/jiangtianyi/hidream_models/Llama-3.1-8B-Instruct"
HIDREAM_I1_PATH = "/home/ma-user/work/jiangtianyi/hidream_models/HiDream-I1-Full"
HIDREAM_E1_PATH = "/home/ma-user/work/jiangtianyi/hidream_models/HiDream-E1-1"
```

---

## ✅ 第四步：创建使用本地模型的脚本

我帮你创建一个新版本 `inference_e1_1_local.py`：

```python
import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from pipeline_hidream_image_editing import HiDreamImageEditingPipeline
from PIL import Image
from diffusers import HiDreamImageTransformer2DModel
import json
import os
from collections import defaultdict
from safetensors.torch import safe_open
import math
import logging

# ============ NPU 支持 ============
from device_utils import DeviceManager
# ==================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# ============ 使用本地模型配置 ============
try:
    from local_config import LLAMA_PATH, HIDREAM_I1_PATH, HIDREAM_E1_PATH, verify_models
    logging.info("使用本地模型配置")
    if not verify_models():
        raise FileNotFoundError("本地模型路径检查失败")
except ImportError:
    # 如果没有 local_config.py，使用默认路径
    logging.warning("未找到 local_config.py，使用默认模型路径")
    LLAMA_PATH = "meta-llama/Llama-3.1-8B-Instruct"
    HIDREAM_I1_PATH = "HiDream-ai/HiDream-I1-Full"
    HIDREAM_E1_PATH = "HiDream-ai/HiDream-E1-1"
# ==========================================

# ============ 初始化设备管理器 ============
preferred_device = os.environ.get('PREFERRED_DEVICE', None)
device_manager = DeviceManager(preferred_device=preferred_device)
DEVICE = device_manager.device
DTYPE = device_manager.dtype
# =========================================

# Flash Attention 处理
os.environ.setdefault("DIFFUSERS_ATTENTION_TYPE", "vanilla")
logging.info(f"Attention type: {os.environ.get('DIFFUSERS_ATTENTION_TYPE', 'default')}")

# ... (其余代码与 inference_e1_1_npu.py 完全相同)
```

---

## 🚀 第五步：运行

```bash
cd /home/ma-user/work/jiangtianyi/HiDream-E1

# 1. 验证模型路径
python local_config.py

# 2. 运行推理（使用本地模型）
python inference_e1_1_local.py
```

---

## 📁 最终目录结构

```
/home/ma-user/work/jiangtianyi/
├── HiDream-E1/                    # 项目代码
│   ├── local_config.py           # 本地模型配置 (新建)
│   ├── inference_e1_1_local.py   # 使用本地模型的推理脚本 (新建)
│   ├── device_utils.py
│   ├── test_npu_setup.py
│   └── ...
└── hidream_models/                # 模型文件
    ├── Llama-3.1-8B-Instruct/    # ~16 GB
    │   ├── config.json
    │   ├── tokenizer.json
    │   ├── model-*.safetensors
    │   └── ...
    ├── HiDream-I1-Full/           # ~15 GB
    │   ├── transformer/
    │   ├── vae/
    │   └── ...
    └── HiDream-E1-1/              # ~15 GB
        ├── transformer/
        └── ...
```

---

## 🔍 验证模型文件完整性

```bash
# 检查每个模型目录
ls -lh /home/ma-user/work/jiangtianyi/hidream_models/Llama-3.1-8B-Instruct/
ls -lh /home/ma-user/work/jiangtianyi/hidream_models/HiDream-I1-Full/
ls -lh /home/ma-user/work/jiangtianyi/hidream_models/HiDream-E1-1/

# 应该看到 config.json, model 文件等
```

---

## ⚠️ 常见问题

### Q1: 模型下载中断怎么办？

使用 `resume_download=True` 参数，可以断点续传：

```python
snapshot_download(
    repo_id=model_id,
    local_dir=local_path,
    resume_download=True,  # 断点续传
)
```

### Q2: 空间不够怎么办？

```bash
# 检查磁盘空间
df -h

# 如果 home 目录空间不足，使用其他目录
# 修改 local_config.py 中的 MODEL_BASE_DIR
MODEL_BASE_DIR = "/data/models"  # 改为空间充足的目录
```

### Q3: 上传太慢怎么办？

```bash
# 使用压缩可以减少体积约 20-30%
tar -czf hidream_models.tar.gz hidream_models/

# 或者分批上传
rsync -avz --progress hidream_models/Llama-3.1-8B-Instruct/ server:~/models/llama/
rsync -avz --progress hidream_models/HiDream-I1-Full/ server:~/models/hidream-i1/
rsync -avz --progress hidream_models/HiDream-E1-1/ server:~/models/hidream-e1/
```

---

## ✅ 完整流程检查清单

- [ ] 在本地下载所有模型（约 40-50 GB）
- [ ] 上传到服务器
- [ ] 创建 `local_config.py` 配置文件
- [ ] 验证模型路径：`python local_config.py`
- [ ] 创建或修改推理脚本使用本地路径
- [ ] 运行测试：`python inference_e1_1_local.py`

完成后，你就不再依赖网络下载了！
