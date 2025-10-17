# 🚀 NPU 快速开始指南（5分钟版）

如果你的华为服务器**已经配置好 CANN 和 torch_npu**，按照这个快速指南操作。

详细指南见：[HUAWEI_NPU_SETUP_GUIDE.md](HUAWEI_NPU_SETUP_GUIDE.md)

---

## ⚡ 超快速部署（假设环境已配置）

```bash
# 1. 上传代码到服务器
scp -r /Users/jty/File/MLLM/HiDream-E1 user@server:/home/user/

# 2. SSH 登录服务器
ssh user@server

# 3. 进入项目目录
cd ~/HiDream-E1

# 4. 激活环境（如果使用虚拟环境）
conda activate your_env  # 或 source venv/bin/activate

# 5. 安装依赖
pip install -r requirements_npu.txt

# 6. 登录 HuggingFace
huggingface-cli login

# 7. 测试环境
python test_npu_setup.py

# 8. 运行推理
python inference_e1_1_npu.py
```

**完成！** 结果保存在 `results/test_1_npu.jpg`

---

## 📝 必须检查的事项

### 1️⃣ NPU 可用性
```bash
python -c "import torch; import torch_npu; print('NPU:', torch.npu.is_available())"
```
**必须输出**: `NPU: True`

### 2️⃣ CANN 环境变量
```bash
echo $ASCEND_TOOLKIT_HOME
```
**应该输出**: `/usr/local/Ascend/ascend-toolkit/latest` 或类似路径

### 3️⃣ NPU 设备状态
```bash
npu-smi info
```
**应该看到**: NPU 设备列表和状态

---

## 🎯 核心文件说明

| 文件 | 用途 | 何时使用 |
|------|------|----------|
| **inference_e1_1_npu.py** | E1.1 推理脚本 | 推荐，支持动态分辨率 |
| **inference_npu.py** | E1-Full 推理脚本 | 需要指令优化功能时 |
| **gradio_demo_npu.py** | Web 交互界面 | 需要可视化界面时 |
| **test_npu_setup.py** | 环境测试 | 首次部署或遇到问题时 |
| **device_utils.py** | 设备管理工具 | 被其他脚本自动调用 |
| **run_npu.sh** | 一键启动脚本 | 懒人模式 😄 |

---

## 🔧 三个常用命令

### 1. 测试环境
```bash
python test_npu_setup.py
```

### 2. 运行推理
```bash
python inference_e1_1_npu.py
```

### 3. 启动 Web 界面
```bash
python gradio_demo_npu.py
# 然后访问 http://server-ip:7860
```

---

## ⚙️ 环境变量速查

```bash
# 在 ~/.bashrc 中添加（如果还没有）
export ASCEND_HOME=/usr/local/Ascend
export ASCEND_TOOLKIT_HOME=${ASCEND_HOME}/ascend-toolkit/latest
source ${ASCEND_TOOLKIT_HOME}/set_env.sh

# 可选：指定 NPU 设备
export NPU_VISIBLE_DEVICES=0

# 可选：HuggingFace 缓存
export HF_HOME=/data/huggingface
```

**修改后记得**:
```bash
source ~/.bashrc
```

---

## 🐛 快速排错

### 问题：NPU 不可用
```bash
# 检查驱动
npu-smi info

# 重新加载环境
source ~/.bashrc
```

### 问题：torch_npu 导入失败
```bash
# 重新安装
pip uninstall torch_npu
pip install torch_npu

# 检查版本匹配
python -c "import torch; print(torch.__version__)"
```

### 问题：内存不足
```bash
# 方法1：减少分辨率
# 编辑脚本，修改 image_size=768 (从1024降到768)

# 方法2：减少步数
# 修改 steps=20 (从28降到20)

# 方法3：启用 CPU offload
# 在脚本中添加: pipe.enable_model_cpu_offload()
```

### 问题：模型下载慢
```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 📊 性能调优速查

### 提升速度
```python
# 修改 inference_e1_1_npu.py 第32行
os.environ["DIFFUSERS_ATTENTION_TYPE"] = "sdpa"  # 从 "vanilla" 改为 "sdpa"

# 减少推理步数
steps=20  # 从 28 改为 20
```

### 节省内存
```python
# 降低分辨率
image_size=768  # 从 1024 改为 768

# 使用 float16
# 会自动检测，如果 bfloat16 不支持会降级
```

---

## 📖 完整文档索引

| 文档 | 用途 |
|------|------|
| **QUICK_START_NPU.md** (本文档) | 5分钟快速开始 |
| **HUAWEI_NPU_SETUP_GUIDE.md** | 完整部署指南（从零开始）|
| **README_NPU.md** | 使用说明和注意事项 |
| **NPU_MIGRATION_GUIDE.md** | 技术迁移细节 |
| **FLASH_ATTENTION_SOLUTIONS.md** | Flash Attention 问题解决 |
| **NPU_MODIFICATIONS_SUMMARY.md** | 代码修改总结 |

---

## ✅ 最小化验证流程

**只需 3 个命令确认环境 OK：**

```bash
# 1. NPU 可用
python -c "import torch,torch_npu; assert torch.npu.is_available()"

# 2. 设备管理器正常
python -c "from device_utils import DeviceManager; dm=DeviceManager(); print(dm)"

# 3. 测试张量运算
python -c "import torch; x=torch.randn(10,10).to('npu:0'); print('✓ OK')"
```

**全部通过？立即运行推理！**

---

## 🎉 成功标志

运行推理后，看到这些说明成功：

```
✓ Models loaded successfully!
  Device: npu:0
  Data type: torch.bfloat16
...
✓ Image editing completed successfully!
  Output saved to: results/test_1_npu.jpg
  Inference time: 14.32 seconds
```

---

需要帮助？
- 📖 查看完整指南：HUAWEI_NPU_SETUP_GUIDE.md
- 🔍 运行诊断：`python test_npu_setup.py`
- 💬 提交 Issue 或咨询华为技术支持
