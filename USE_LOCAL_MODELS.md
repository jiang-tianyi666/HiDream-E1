# 📦 使用本地模型快速指南

本文档说明如何配置和使用本地下载的模型。

## 🎯 你需要做什么

### 第一步：下载模型到本地（在有网络的机器上）

下载所需的 3 个模型，总大小约 40-50 GB：

```bash
# 安装工具
pip install -U huggingface_hub

# 登录 HuggingFace（需要先在网页同意 Llama 协议）
huggingface-cli login

# 下载到本地
mkdir -p ~/hidream_models

# 下载 Llama-3.1-8B（约 16 GB）
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir ~/hidream_models/Llama-3.1-8B-Instruct

# 下载 HiDream-I1-Full（约 15 GB）
huggingface-cli download HiDream-ai/HiDream-I1-Full \
    --local-dir ~/hidream_models/HiDream-I1-Full

# 下载 HiDream-E1.1（约 15 GB）
huggingface-cli download HiDream-ai/HiDream-E1-1 \
    --local-dir ~/hidream_models/HiDream-E1-1
```

### 第二步：上传到服务器

```bash
# 压缩（可选，节省传输时间）
cd ~
tar -czf hidream_models.tar.gz hidream_models/

# 上传
scp hidream_models.tar.gz user@server:/home/ma-user/work/jiangtianyi/

# 在服务器上解压
ssh user@server
cd /home/ma-user/work/jiangtianyi/
tar -xzf hidream_models.tar.gz
```

### 第三步：修改配置文件

在服务器上编辑 `local_config.py`：

```bash
cd /home/ma-user/work/jiangtianyi/HiDream-E1
vim local_config.py
```

修改第 13 行的路径：

```python
# 改为你实际的模型路径
MODEL_BASE_DIR = "/home/ma-user/work/jiangtianyi/hidream_models"
```

### 第四步：验证配置

```bash
python local_config.py
```

**如果看到：**
```
============================================================
检查本地模型路径
============================================================
基础目录: /home/ma-user/work/jiangtianyi/hidream_models
----------------------------------------------------------------------
✓ Llama-3.1-8B-Instruct  -> /home/ma-user/work/jiangtianyi/hidream_models/Llama-3.1-8B-Instruct
  └─ config.json ✓  tokenizer.json ✓
✓ HiDream-I1-Full        -> /home/ma-user/work/jiangtianyi/hidream_models/HiDream-I1-Full
  └─ transformer/ ✓
✓ HiDream-E1-1           -> /home/ma-user/work/jiangtianyi/hidream_models/HiDream-E1-1
  └─ transformer/ ✓
============================================================
✓ 所有模型路径正确且文件完整
============================================================
```

说明配置成功！

### 第五步：运行推理

```bash
# 直接运行，会自动使用本地模型
python inference_e1_1_npu.py
```

## 🔍 代码自动识别逻辑

修改后的脚本会**自动判断**是否使用本地模型：

```python
# 1. 尝试导入本地配置
try:
    from local_config import LLAMA_PATH, HIDREAM_I1_PATH, HIDREAM_E1_PATH
    # 找到了，使用本地路径
    logging.info("📦 使用本地模型配置")
except ImportError:
    # 没找到 local_config.py，使用在线下载
    logging.info("将从 HuggingFace 下载模型（需要网络）")
    LLAMA_PATH = "meta-llama/Llama-3.1-8B-Instruct"
    # ...
```

## 📁 目录结构示例

```
/home/ma-user/work/jiangtianyi/
├── HiDream-E1/                           # 项目代码
│   ├── local_config.py                  # 本地模型配置 ← 你需要修改这个
│   ├── inference_e1_1_npu.py            # 推理脚本（已修改）
│   ├── inference_npu.py                 # 推理脚本（已修改）
│   └── ...
└── hidream_models/                       # 模型文件
    ├── Llama-3.1-8B-Instruct/           # ← 下载的模型
    │   ├── config.json
    │   ├── tokenizer.json
    │   ├── model-00001-of-00004.safetensors
    │   ├── model-00002-of-00004.safetensors
    │   ├── model-00003-of-00004.safetensors
    │   ├── model-00004-of-00004.safetensors
    │   └── ...
    ├── HiDream-I1-Full/                 # ← 下载的模型
    │   ├── model_index.json
    │   ├── transformer/
    │   │   ├── config.json
    │   │   └── diffusion_pytorch_model.safetensors
    │   ├── vae/
    │   └── ...
    └── HiDream-E1-1/                    # ← 下载的模型
        ├── transformer/
        └── ...
```

## ✅ 完整验证步骤

在服务器上执行：

```bash
# 1. 检查模型文件
ls -lh /home/ma-user/work/jiangtianyi/hidream_models/

# 2. 验证配置
cd /home/ma-user/work/jiangtianyi/HiDream-E1
python local_config.py

# 3. 测试推理（会显示使用本地模型）
python inference_e1_1_npu.py
```

## 🎉 成功标志

运行推理时，看到这个日志：

```
📦 使用本地模型配置
✓ 本地模型路径验证成功
```

说明成功使用本地模型，**不再需要网络下载**！

## ❓ 常见问题

### Q: 我的模型路径不在这个位置怎么办？

**A:** 编辑 `local_config.py`，修改第 13 行的 `MODEL_BASE_DIR`

### Q: 如何知道模型是否下载完整？

**A:** 运行 `python local_config.py`，会检查关键文件

### Q: 可以删除 local_config.py 吗？

**A:** 可以！删除后脚本会自动从 HuggingFace 下载（需要网络）

### Q: 如何在本地和在线之间切换？

**A:**
- 使用本地：确保 `local_config.py` 存在且配置正确
- 使用在线：删除或重命名 `local_config.py`

## 📚 详细文档

更多信息请查看：
- [LOCAL_MODELS_GUIDE.md](LOCAL_MODELS_GUIDE.md) - 完整下载和配置指南
- [HUAWEI_NPU_SETUP_GUIDE.md](HUAWEI_NPU_SETUP_GUIDE.md) - 服务器部署指南
