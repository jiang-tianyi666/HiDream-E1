# 华为昇腾服务器部署完整指南

本指南提供在华为昇腾 NPU 服务器上部署和运行 HiDream-E1 的详细步骤。

---

## 📋 目录

1. [环境检查](#1-环境检查)
2. [安装 CANN](#2-安装-cann)
3. [安装 PyTorch 和 torch_npu](#3-安装-pytorch-和-torch_npu)
4. [部署项目代码](#4-部署项目代码)
5. [安装项目依赖](#5-安装项目依赖)
6. [配置 HuggingFace](#6-配置-huggingface)
7. [运行测试](#7-运行测试)
8. [执行推理](#8-执行推理)
9. [常见问题](#9-常见问题)
10. [性能优化](#10-性能优化)

---

## 1. 环境检查

### 1.1 检查服务器信息

```bash
# SSH 登录到华为服务器
ssh user@your-huawei-server-ip

# 检查操作系统
cat /etc/os-release
# 推荐: Ubuntu 20.04/22.04 或 CentOS 7/8

# 检查 NPU 设备
npu-smi info
# 应该能看到 NPU 设备列表，例如 Ascend 910 或 310P
```

**预期输出示例：**
```
+-----------------------------------------------------------------------------+
| npu-smi 22.0.0                   Version: 22.0.0                            |
+-----------------------------------------------------------------------------+
| NPU     Name                   | Health | Power(W)    Temp(C)               |
| Chip                           | Bus-Id | AICore(%)   Memory-Usage(MB)      |
+=============================================================================+
| 0       Ascend910              | OK     | 120.0       45                    |
|         0                      | 0000:C1:00.0 | 0          0    / 32768        |
+=============================================================================+
```

### 1.2 检查 CANN 是否已安装

```bash
# 检查 CANN 版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.info

# 或者
ls /usr/local/Ascend/
```

**如果看到类似输出，说明 CANN 已安装：**
```
/usr/local/Ascend/
├── ascend-toolkit/
├── driver/
└── nnae/
```

---

## 2. 安装 CANN

如果 CANN 未安装，需要先安装。

### 2.1 下载 CANN

访问华为官网下载 CANN：
- **下载地址**: https://www.hiascend.com/software/cann/community

选择版本：
- **推荐**: CANN 8.0.RC1 或更高版本
- **架构**: 根据你的 NPU 型号选择（Ascend 910/310P 等）
- **操作系统**: 匹配你的服务器系统

### 2.2 安装 CANN Toolkit

```bash
# 假设下载的文件名为 Ascend-cann-toolkit_8.0.RC1_linux-*.run
chmod +x Ascend-cann-toolkit_8.0.RC1_linux-*.run

# 安装（需要 root 权限或 sudo）
sudo ./Ascend-cann-toolkit_8.0.RC1_linux-*.run --install

# 按提示选择安装路径，默认: /usr/local/Ascend
```

### 2.3 安装 CANN Kernels

```bash
# 下载并安装 kernels（必需）
chmod +x Ascend-cann-kernels-910_8.0.RC1_linux.run
sudo ./Ascend-cann-kernels-910_8.0.RC1_linux.run --install
```

### 2.4 配置环境变量

```bash
# 编辑 ~/.bashrc
vim ~/.bashrc

# 添加以下内容（根据实际安装路径调整）
# ==================== CANN 环境变量 ====================
export ASCEND_HOME=/usr/local/Ascend
export ASCEND_TOOLKIT_HOME=${ASCEND_HOME}/ascend-toolkit/latest

# CANN 路径
source ${ASCEND_TOOLKIT_HOME}/set_env.sh

# NPU 设备可见性（类似 CUDA_VISIBLE_DEVICES）
export NPU_VISIBLE_DEVICES=0  # 使用第一个 NPU

# 日志级别（可选，调试时使用）
export ASCEND_GLOBAL_LOG_LEVEL=3  # 3=INFO, 0=DEBUG
export ASCEND_SLOG_PRINT_TO_STDOUT=1  # 打印日志到终端
# ======================================================

# 保存并生效
source ~/.bashrc
```

### 2.5 验证 CANN 安装

```bash
# 检查环境变量
echo $ASCEND_TOOLKIT_HOME

# 测试 npu-smi
npu-smi info

# 检查库文件
ls $ASCEND_TOOLKIT_HOME/lib64/
# 应该看到 libascendcl.so, libge_runner.so 等
```

---

## 3. 安装 PyTorch 和 torch_npu

### 3.1 创建 Python 虚拟环境（推荐）

```bash
# 使用 conda（推荐）
conda create -n hidream python=3.10
conda activate hidream

# 或使用 venv
python3 -m venv ~/hidream_env
source ~/hidream_env/bin/activate
```

### 3.2 确定 PyTorch 和 torch_npu 版本

**关键**: PyTorch 版本必须与 CANN 版本匹配！

| CANN 版本 | PyTorch 版本 | torch_npu 版本 |
|-----------|--------------|----------------|
| 8.0.RC1   | 2.1.0        | 对应版本       |
| 7.0.0     | 2.0.1        | 对应版本       |
| 6.3.0     | 1.11.0       | 对应版本       |

查看官方兼容性文档：
https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/configandinstg/instg/instg_0001.html

### 3.3 安装 PyTorch

```bash
# 示例：安装 PyTorch 2.1.0（根据你的 CANN 版本调整）
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# 验证安装
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### 3.4 安装 torch_npu

**方法 A: 使用 pip（推荐）**

```bash
# 查看可用版本
# https://gitee.com/ascend/pytorch/releases

# 安装（根据 PyTorch 版本选择对应的 torch_npu）
pip install torch_npu

# 或指定版本
# pip install torch-npu==2.1.0.post3
```

**方法 B: 从源码安装**

```bash
# 克隆仓库
git clone https://gitee.com/ascend/pytorch.git
cd pytorch

# 切换到对应分支
git checkout v2.1.0-5.0.0  # 根据实际版本

# 编译安装
bash ci/build.sh --python=3.10
pip install dist/torch_npu-*.whl
```

### 3.5 验证 torch_npu 安装

```bash
# 测试脚本
python << 'EOF'
import torch
import torch_npu

print(f"PyTorch version: {torch.__version__}")
print(f"torch_npu version: {torch_npu.__version__ if hasattr(torch_npu, '__version__') else 'unknown'}")

# 检查 NPU 可用性
if torch.npu.is_available():
    print(f"✓ NPU is available!")
    print(f"  NPU count: {torch.npu.device_count()}")
    print(f"  Current NPU: {torch.npu.current_device()}")
    print(f"  NPU name: {torch.npu.get_device_name(0)}")

    # 测试基本运算
    x = torch.randn(3, 3).to("npu:0")
    y = x @ x.T
    print(f"  ✓ Basic tensor operations work")
else:
    print("✗ NPU is not available!")
    print("  Check CANN installation and environment variables")
EOF
```

**预期输出：**
```
PyTorch version: 2.1.0
torch_npu version: 2.1.0.post3
✓ NPU is available!
  NPU count: 1
  Current NPU: 0
  NPU name: Ascend910
  ✓ Basic tensor operations work
```

---

## 4. 部署项目代码

### 4.1 上传代码到服务器

**方法 A: 使用 git**

```bash
# 在服务器上
cd ~
git clone https://github.com/your-repo/HiDream-E1.git
cd HiDream-E1
```

**方法 B: 使用 scp**

```bash
# 在本地机器上
scp -r /Users/jty/File/MLLM/HiDream-E1 user@server-ip:~/
```

**方法 C: 使用 rsync（推荐，支持增量同步）**

```bash
# 在本地机器上
rsync -avz --progress \
  /Users/jty/File/MLLM/HiDream-E1/ \
  user@server-ip:~/HiDream-E1/
```

### 4.2 检查文件完整性

```bash
cd ~/HiDream-E1
ls -lh

# 确保有以下 NPU 相关文件
ls -1 | grep -E "(npu|device_utils)"
```

**应该看到：**
```
device_utils.py
gradio_demo_npu.py
inference_e1_1_npu.py
inference_npu.py
npu_attention_optimizer.py
requirements_npu.txt
README_NPU.md
NPU_MIGRATION_GUIDE.md
FLASH_ATTENTION_SOLUTIONS.md
test_npu_setup.py
run_npu.sh
```

---

## 5. 安装项目依赖

### 5.1 安装基础依赖

```bash
cd ~/HiDream-E1

# 确保虚拟环境已激活
conda activate hidream  # 或 source ~/hidream_env/bin/activate

# 安装 NPU 版本依赖
pip install -r requirements_npu.txt

# 这个过程可能需要 5-10 分钟
```

### 5.2 处理常见安装问题

**问题 1: Diffusers 安装失败**
```bash
# 如果从 git 安装失败，尝试使用 PyPI 版本
pip install diffusers>=0.27.0
```

**问题 2: Transformers 版本冲突**
```bash
pip install transformers==4.47.1 --force-reinstall
```

**问题 3: 缺少系统库**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential python3-dev

# CentOS/RHEL
sudo yum install -y gcc gcc-c++ python3-devel
```

### 5.3 验证依赖安装

```bash
# 运行测试脚本
python test_npu_setup.py
```

**预期看到：**
```
==============================================================
 NPU Environment Test Suite
==============================================================

1. Testing PyTorch Installation
==============================================================
PyTorch version: 2.1.0
CUDA available: False

2. Testing NPU Installation
==============================================================
✓ torch_npu imported successfully
  NPU available: True
  NPU count: 1
  Current NPU: 0
  ✓ Basic NPU operations work

3. Testing Device Manager
==============================================================
✓ DeviceManager initialized
  Device type: npu
  Device: npu:0
  Data type: torch.bfloat16
  ...
```

---

## 6. 配置 HuggingFace

### 6.1 登录 HuggingFace

```bash
# 安装 CLI（如果未安装）
pip install -U huggingface_hub

# 登录
huggingface-cli login
```

**按提示输入你的 HuggingFace Token：**
1. 访问 https://huggingface.co/settings/tokens
2. 创建或复制 Access Token
3. 粘贴到终端

### 6.2 同意 Llama 模型协议

访问并同意协议：
https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

### 6.3 验证登录

```bash
huggingface-cli whoami
# 应该显示你的用户名

# 测试下载（可选，会占用空间）
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
print('✓ HuggingFace works')
"
```

### 6.4 配置缓存目录（可选）

```bash
# 如果主目录空间不足，设置缓存到其他位置
export HF_HOME=/data/huggingface  # 改为你的大容量磁盘路径
mkdir -p $HF_HOME

# 添加到 ~/.bashrc
echo "export HF_HOME=/data/huggingface" >> ~/.bashrc
```

---

## 7. 运行测试

### 7.1 快速环境测试

```bash
cd ~/HiDream-E1

# 运行完整测试套件
python test_npu_setup.py
```

### 7.2 测试设备管理器

```bash
python device_utils.py
```

**预期输出：**
```
==============================================================
Device Manager Test
==============================================================
🚀 Device Manager initialized
   Device type: npu
   Device: npu:0
   Data type: torch.bfloat16

DeviceManager(device=npu:0, dtype=torch.bfloat16)

📝 Testing generator...
✓ Generator created: <class 'torch.Generator'>

📊 Memory statistics:
📊 NPU Memory Stats:
   Allocated: 0.00 GB
   Reserved: 0.00 GB
   Peak: 0.00 GB
```

### 7.3 测试 Attention 优化器

```bash
python npu_attention_optimizer.py
```

---

## 8. 执行推理

### 8.1 准备测试图像

```bash
# 检查是否有测试图像
ls assets/test_1.png

# 如果没有，上传一张测试图片
# scp /path/to/test_image.png user@server-ip:~/HiDream-E1/assets/test_1.png
```

### 8.2 运行推理（E1.1 版本 - 推荐）

```bash
cd ~/HiDream-E1

# 方法 1: 直接运行 Python 脚本
python inference_e1_1_npu.py
```

**首次运行会：**
1. 下载模型（约 30-40 GB，需要时间）
2. 加载到 NPU（需要 2-3 分钟）
3. 执行推理

**预期输出：**
```
============================================================
Loading models...
============================================================
Loading Llama tokenizer and text encoder from meta-llama/Llama-3.1-8B-Instruct...
✓ Llama model loaded
Loading transformer from HiDream-ai/HiDream-I1-Full...
✓ Transformer loaded
...
============================================================
✓ Models loaded successfully!
  Device: npu:0
  Data type: torch.bfloat16
📊 NPU Memory Stats:
   Allocated: 15.23 GB
   Reserved: 16.00 GB
============================================================
Original size: (1024, 768)
Processed size: (1024, 768)
Instruction: Convert the image into a Ghibli style.
Starting image generation...
...
============================================================
✓ Image editing completed successfully!
  Output saved to: results/test_1_npu.jpg
  Metadata saved to: results/test_1_npu.json
  Inference time: 14.32 seconds
============================================================
```

### 8.3 使用快速启动脚本

```bash
# 给脚本执行权限
chmod +x run_npu.sh

# 运行
./run_npu.sh

# 根据提示选择：
# 1) inference_e1_1_npu.py  (E1.1 - 推荐)
# 2) inference_npu.py       (E1-Full)
# 3) gradio_demo_npu.py     (Web 界面)
# 4) test_npu_setup.py      (测试)
```

### 8.4 运行 Gradio Web 界面

```bash
# 启动 Web 界面
python gradio_demo_npu.py

# 服务会在 http://0.0.0.0:7860 启动
```

**访问界面：**
```bash
# 如果服务器有公网 IP
http://your-server-ip:7860

# 如果需要 SSH 端口转发
# 在本地机器上执行：
ssh -L 7860:localhost:7860 user@server-ip

# 然后访问
http://localhost:7860
```

---

## 9. 常见问题

### 问题 1: NPU 不可用

```bash
# 检查 npu-smi
npu-smi info

# 检查驱动
ls /usr/local/Ascend/driver/

# 重新加载环境变量
source ~/.bashrc
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

### 问题 2: 内存不足

```bash
# 启用 CPU Offload
# 修改脚本，在模型加载后添加：
pipe.enable_model_cpu_offload()

# 或减少批处理大小、降低分辨率
```

### 问题 3: 算子不支持

```bash
# 启用回退模式
export NPU_FALLBACK_MODE=1
python inference_e1_1_npu.py
```

### 问题 4: 模型下载慢或失败

```bash
# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载后指定本地路径
# 修改脚本中的模型路径
```

### 问题 5: bfloat16 不支持

脚本会自动降级到 float16 或 float32，查看日志确认使用的精度。

---

## 10. 性能优化

### 10.1 切换到 SDPA Attention

```bash
# 编辑 inference_e1_1_npu.py
vim inference_e1_1_npu.py

# 找到第 32 行，修改：
# os.environ.setdefault("DIFFUSERS_ATTENTION_TYPE", "vanilla")
# 改为：
os.environ.setdefault("DIFFUSERS_ATTENTION_TYPE", "sdpa")

# 保存后测试
python inference_e1_1_npu.py
```

### 10.2 减少推理步数

```python
# 在 edit_image() 调用时
steps=20  # 从默认的 28 改为 20
```

### 10.3 调整分辨率

```python
# 修改 resize_image 函数
def resize_image(pil_image, image_size=768):  # 从 1024 降到 768
```

### 10.4 启用算子优化

```bash
# 添加环境变量
export ACL_OP_COMPILER_CACHE_MODE=1
export ACL_OP_SELECT_IMPL_MODE=1
```

---

## 📊 性能基准参考

| 配置 | 推理时间 (28 steps) | 内存占用 |
|------|---------------------|----------|
| Ascend 910 + bfloat16 + vanilla | ~14-16秒 | ~16GB |
| Ascend 910 + bfloat16 + sdpa | ~11-13秒 | ~15GB |
| Ascend 910 + float16 | ~13-15秒 | ~14GB |

---

## ✅ 部署检查清单

- [ ] NPU 设备可见 (`npu-smi info`)
- [ ] CANN 已安装并配置环境变量
- [ ] PyTorch 和 torch_npu 版本匹配
- [ ] `torch.npu.is_available()` 返回 True
- [ ] 所有依赖已安装 (`pip list`)
- [ ] HuggingFace 已登录
- [ ] 测试脚本通过 (`python test_npu_setup.py`)
- [ ] 设备管理器正常工作
- [ ] 能成功运行推理

---

## 📞 获取帮助

**遇到问题时：**

1. 查看日志
```bash
python inference_e1_1_npu.py 2>&1 | tee debug.log
```

2. 运行诊断
```bash
python test_npu_setup.py > diagnostic.txt 2>&1
```

3. 检查文档
- README_NPU.md
- NPU_MIGRATION_GUIDE.md
- FLASH_ATTENTION_SOLUTIONS.md

4. 联系支持
- 昇腾社区：https://www.hiascend.com/forum
- GitHub Issues：https://gitee.com/ascend/pytorch

---

祝你部署顺利！🚀
