# HiDream-E1 NPU 版本使用指南

本文档说明如何在华为昇腾 NPU 上运行 HiDream-E1 图像编辑模型。

## 🚀 快速开始

### 环境准备

1. **安装 CANN (必需)**
   ```bash
   # 参考华为官方文档安装 CANN toolkit 和 kernels
   # https://www.hiascend.com/document
   ```

2. **安装 PyTorch 和 torch_npu**
   ```bash
   # 根据你的 CANN 版本安装对应的 torch
   pip install torch==2.1.0  # 版本号需要与 CANN 匹配
   pip install torch_npu      # 华为 NPU 插件

   # 验证安装
   python -c "import torch; import torch_npu; print(torch.npu.is_available())"
   ```

3. **安装项目依赖**
   ```bash
   pip install -r requirements_npu.txt
   ```

4. **登录 HuggingFace**
   ```bash
   huggingface-cli login
   # 需要同意 Llama-3.1-8B-Instruct 的使用协议
   ```

### 运行推理

#### 方法 1：使用 NPU 版本脚本（推荐）

```bash
# HiDream-E1.1 版本（推荐）
python inference_e1_1_npu.py

# HiDream-E1-Full 版本（带 LoRA）
python inference_npu.py
```

#### 方法 2：使用 Gradio 交互界面

```bash
python gradio_demo_npu.py
```

然后在浏览器打开 `http://localhost:7860`

### 指定设备

默认情况下会自动检测设备（NPU > CUDA > CPU），也可以手动指定：

```bash
# 强制使用 NPU
export PREFERRED_DEVICE=npu
python inference_e1_1_npu.py

# 强制使用 CUDA（如果需要对比）
export PREFERRED_DEVICE=cuda
python inference_e1_1_npu.py

# 强制使用 CPU
export PREFERRED_DEVICE=cpu
python inference_e1_1_npu.py
```

## 📝 代码修改说明

### 新增文件

1. **device_utils.py** - 设备管理工具
   - 自动检测 NPU/CUDA/CPU
   - 自动选择合适的数据类型（bfloat16/float16/float32）
   - 统一的内存管理接口

2. **inference_e1_1_npu.py** - NPU 版本推理脚本（E1.1）
3. **inference_npu.py** - NPU 版本推理脚本（E1-Full）
4. **gradio_demo_npu.py** - NPU 版本交互界面

### 关键修改点

1. **设备指定**
   ```python
   # 原代码
   pipe.to("cuda", torch.bfloat16)

   # NPU 版本
   from device_utils import DeviceManager
   dm = DeviceManager()
   pipe.to(dm.device, dm.dtype)
   ```

2. **随机生成器**
   ```python
   # 原代码
   generator = torch.Generator("cuda").manual_seed(seed)

   # NPU 版本
   generator = dm.create_generator(seed)
   ```

3. **内存监控**
   ```python
   # 原代码
   torch.cuda.memory_summary(device='cuda')

   # NPU 版本
   dm.memory_stats()  # 自动适配 NPU/CUDA
   ```

4. **Flash Attention 禁用**
   ```python
   # NPU 不支持 CUDA Flash Attention
   os.environ["DIFFUSERS_ATTENTION_TYPE"] = "vanilla"
   ```

## ⚠️ 注意事项

### 1. 性能预期

- **首次运行**：可能较慢（模型下载、编译）
- **没有 Flash Attention**：推理速度可能比 CUDA 版本慢 30-50%
- **内存占用**：类似 CUDA 版本，约需 16GB+ 显存

### 2. 精度支持

设备管理器会自动检测并选择最佳精度：
- 优先使用 `bfloat16`（如果支持）
- 降级到 `float16`（如果 bfloat16 不支持）
- 最后降级到 `float32`（如果前两者都不支持）

### 3. 已知问题

#### 问题 1：算子不支持
**现象**：`RuntimeError: operator XXX is not implemented for NPU`

**解决**：
```bash
# 启用回退模式（不支持的算子自动回退到 CPU）
export NPU_FALLBACK_MODE=1
python inference_e1_1_npu.py
```

#### 问题 2：内存溢出
**解决**：
```python
# 在脚本中启用 CPU offload
pipe.enable_model_cpu_offload()
```

#### 问题 3：Llama 模型加载失败
**解决**：
```bash
# 确保已登录 HuggingFace 并同意协议
huggingface-cli login
```

## 🧪 测试和验证

### 基础测试

```bash
# 1. 测试设备检测
python device_utils.py

# 2. 测试推理（小步数）
python -c "
from device_utils import DeviceManager
dm = DeviceManager()
print(f'Device: {dm.device}')
print(f'Data type: {dm.dtype}')
"
```

### 性能对比

```bash
# 在 NPU 上运行
time python inference_e1_1_npu.py

# 在 CUDA 上运行（如果有）
export PREFERRED_DEVICE=cuda
time python inference_e1_1_npu.py
```

## 📊 性能优化建议

### 1. 使用混合精度

如果遇到精度问题，可以让 VAE 使用更高精度：

```python
# 在 init_models() 后添加
pipe.vae = pipe.vae.to(torch.float32)
```

### 2. 减少推理步数

```python
# 在 edit_image() 中调整
steps=20  # 从默认的 28 减少到 20
```

### 3. 降低分辨率

```python
# 修改 resize_image 函数
def resize_image(pil_image, image_size=768):  # 从 1024 降到 768
    ...
```

### 4. 批处理优化

如果处理多张图片：

```python
# 使用批处理
prompts = ["instruction1", "instruction2", "instruction3"]
images = [img1, img2, img3]
results = pipe(prompt=prompts, image=images, ...)
```

## 🔍 调试技巧

### 1. 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. 检查模型精度

```python
print(f"Transformer dtype: {pipe.transformer.dtype}")
print(f"VAE dtype: {pipe.vae.dtype}")
print(f"Text encoder dtype: {pipe.text_encoder_4.dtype}")
```

### 3. 逐组件测试

```python
# 测试 VAE
dummy_img = torch.randn(1, 3, 512, 512).to(dm.device, dm.dtype)
latent = pipe.vae.encode(dummy_img).latent_dist.sample()
print("✓ VAE works")

# 测试 Text Encoder
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
inputs = tokenizer("test", return_tensors="pt").to(dm.device)
outputs = pipe.text_encoder_4(**inputs)
print("✓ Text encoder works")
```

## 📞 获取帮助

- **华为昇腾社区**：https://www.hiascend.com/forum
- **CANN 文档**：https://www.hiascend.com/document
- **torch_npu GitHub**：https://gitee.com/ascend/pytorch
- **项目 Issues**：提交到原项目或联系开发者

## 📄 文件对照表

| 原始文件 | NPU 版本 | 说明 |
|---------|---------|------|
| inference_e1_1.py | inference_e1_1_npu.py | E1.1 推理脚本 |
| inference.py | inference_npu.py | E1-Full 推理脚本 |
| gradio_demo_1_1.py | gradio_demo_npu.py | 交互界面 |
| requirements.txt | requirements_npu.txt | 依赖列表 |
| - | device_utils.py | 设备管理工具（新增） |
| - | README_NPU.md | NPU 使用指南（新增） |

## ✅ 迁移检查清单

- [ ] 已安装 CANN toolkit 和 kernels
- [ ] 已安装 torch 和 torch_npu
- [ ] `torch.npu.is_available()` 返回 `True`
- [ ] 已安装项目依赖 `pip install -r requirements_npu.txt`
- [ ] 已登录 HuggingFace `huggingface-cli login`
- [ ] 已测试设备检测 `python device_utils.py`
- [ ] 已运行基础推理测试
- [ ] 输出图像质量可接受
- [ ] 性能满足需求

祝使用顺利！如有问题，请参考 NPU_MIGRATION_GUIDE.md 获取更详细的技术说明。
