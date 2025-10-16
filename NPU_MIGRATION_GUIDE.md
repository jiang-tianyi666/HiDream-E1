# 华为昇腾 NPU 迁移指南

本指南说明如何将 HiDream-E1 迁移到华为昇腾 NPU 上运行。

## 🔍 关键问题概览

将此模型迁移到昇腾 NPU 主要涉及以下方面：

### 1. **设备指定问题** (高优先级)
### 2. **Flash Attention 适配** (关键)
### 3. **算子兼容性** (需要测试)
### 4. **精度支持** (bfloat16)
### 5. **依赖库适配** (PyTorch、Diffusers)

---

## 📋 详细迁移清单

### 一、环境准备

#### 1.1 安装昇腾 CANN 和 torch_npu

```bash
# 1. 安装 CANN (假设使用 CANN 8.0+)
# 参考华为官方文档安装 CANN toolkit 和 kernels

# 2. 安装 torch 和 torch_npu
pip install torch==2.1.0  # 或与你的 CANN 版本匹配的 torch
pip install torch_npu  # 华为提供的 NPU 插件

# 3. 验证安装
python -c "import torch; import torch_npu; print(torch.npu.is_available())"
```

#### 1.2 检查 NPU 设备

```python
import torch
import torch_npu

# 检查 NPU 数量
print(f"NPU count: {torch.npu.device_count()}")

# 设置默认 NPU
torch.npu.set_device(0)
```

---

### 二、代码修改要点

#### 2.1 设备字符串替换 (必须)

**问题文件**：
- [inference_e1_1.py:107](inference_e1_1.py#L107)
- [inference.py:64](inference.py#L64)
- [gradio_demo_1_1.py:107](gradio_demo_1_1.py#L107)
- [gradio_demo.py:62](gradio_demo.py#L62)

**修改示例**：

```python
# 原代码
pipe = pipe.to("cuda", torch.bfloat16)

# 修改为
device = "npu:0"  # 或使用环境变量控制
pipe = pipe.to(device, torch.bfloat16)
```

**建议使用配置方式**：

```python
# 在文件开头添加
import os

# 自动检测设备
if torch.npu.is_available():
    DEVICE = f"npu:{torch.npu.current_device()}"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# 后续使用
pipe = pipe.to(DEVICE, torch.bfloat16)
```

#### 2.2 Generator 设备修改 (必须)

**问题位置**：
- [inference_e1_1.py:153](inference_e1_1.py#L153)
- [inference.py:88](inference.py#L88)
- [gradio_demo.py:79](gradio_demo.py#L79)
- [gradio_demo_1_1.py:125](gradio_demo_1_1.py#L125)

```python
# 原代码
generator = torch.Generator("cuda").manual_seed(seed)

# 修改为
generator = torch.Generator(DEVICE).manual_seed(seed)
```

#### 2.3 内存监控适配

**问题位置**：
- [inference_e1_1.py:110](inference_e1_1.py#L110)
- [gradio_demo_1_1.py:109](gradio_demo_1_1.py#L109)

```python
# 原代码
torch.cuda.memory_summary(device='cuda', abbreviated=True)

# 修改为
if DEVICE.startswith('npu'):
    # NPU 内存查询
    memory_allocated = torch.npu.memory_allocated() / 1024**3  # GB
    memory_reserved = torch.npu.memory_reserved() / 1024**3
    print(f"NPU Memory: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
else:
    torch.cuda.memory_summary(device=DEVICE, abbreviated=True)
```

#### 2.4 权重加载设备

**问题位置**：
- [inference.py:38](inference.py#L38)
- [gradio_demo.py:34](gradio_demo.py#L34)

```python
# 原代码
lora_ckpt = load_file(lora_ckpt_path, device="cuda")

# 修改为
lora_ckpt = load_file(lora_ckpt_path, device=DEVICE)

# 或先加载到 CPU 再转移
lora_ckpt = load_file(lora_ckpt_path, device="cpu")
# 后续会自动转移到正确设备
```

---

### 三、Flash Attention 适配 (关键)

#### 3.1 问题说明

原代码依赖 CUDA Flash Attention：
```bash
pip install -U flash-attn --no-build-isolation
```

**昇腾 NPU 不支持 CUDA Flash Attention**，需要替代方案。

#### 3.2 解决方案

**方案 A：使用标准 Attention（简单但慢）**

修改 Diffusers 库或在初始化时禁用 Flash Attention：

```python
# 在加载模型前设置
import os
os.environ["DIFFUSERS_ATTENTION_TYPE"] = "vanilla"

# 或在 transformer 配置中
transformer = HiDreamImageTransformer2DModel.from_pretrained(
    ...,
    use_flash_attention=False  # 如果模型支持此参数
)
```

**方案 B：使用昇腾优化的 Attention**

华为可能提供了优化的 attention 实现（需要查看 CANN 文档）：

```python
# 可能需要修改 diffusers 源码中的 attention 层
# 路径类似：site-packages/diffusers/models/attention_processor.py

from torch_npu.contrib.module import MultiHeadAttention  # 假设存在

# 替换 attention 实现
```

**方案 C：编译安装昇腾版 Flash Attention（如果可用）**

检查华为是否提供了 NPU 版本的 Flash Attention：
```bash
# 查找昇腾社区是否有实现
# 或联系华为技术支持
```

#### 3.3 性能影响

- 标准 Attention：速度可能下降 30-50%
- 昇腾优化版：接近 CUDA Flash Attention 性能

---

### 四、Diffusers 库适配

#### 4.1 检查兼容性

```python
# 测试 Diffusers 在 NPU 上的基本功能
import torch
import torch_npu
from diffusers import AutoencoderKL

device = "npu:0"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
vae = vae.to(device, torch.bfloat16)

# 测试前向传播
dummy_input = torch.randn(1, 3, 512, 512).to(device, torch.bfloat16)
try:
    latent = vae.encode(dummy_input).latent_dist.sample()
    print("VAE encoding test passed!")
except Exception as e:
    print(f"VAE test failed: {e}")
```

#### 4.2 可能需要的补丁

如果遇到算子不支持，可能需要：

```python
# 在模型加载前
import torch_npu
torch_npu.npu.set_compile_mode(jit_compile=False)  # 禁用 JIT 以调试

# 或设置算子回退
os.environ['NPU_FALLBACK_MODE'] = '1'  # 不支持的算子回退到 CPU
```

---

### 五、精度问题

#### 5.1 bfloat16 支持检查

```python
# 检查 NPU 是否支持 bfloat16
device = "npu:0"
try:
    test_tensor = torch.randn(10, 10, dtype=torch.bfloat16).to(device)
    print("bfloat16 is supported on NPU")
except Exception as e:
    print(f"bfloat16 not supported: {e}")
    print("Consider using float16 or float32")
```

#### 5.2 精度降级方案

如果 bfloat16 不支持：

```python
# 使用 float16
DTYPE = torch.float16 if not torch.npu.is_bf16_supported() else torch.bfloat16

pipe = pipe.to(DEVICE, DTYPE)
```

---

### 六、完整迁移示例代码

创建一个通用的设备抽象层：

```python
# device_utils.py
import torch
import os

class DeviceManager:
    def __init__(self):
        self.device_type = self._detect_device()
        self.device = self._get_device()
        self.dtype = self._get_dtype()

    def _detect_device(self):
        """自动检测可用设备"""
        try:
            import torch_npu
            if torch.npu.is_available():
                return 'npu'
        except ImportError:
            pass

        if torch.cuda.is_available():
            return 'cuda'

        return 'cpu'

    def _get_device(self):
        """获取设备字符串"""
        if self.device_type == 'npu':
            return f"npu:{torch.npu.current_device()}"
        elif self.device_type == 'cuda':
            return f"cuda:{torch.cuda.current_device()}"
        else:
            return "cpu"

    def _get_dtype(self):
        """获取支持的数据类型"""
        if self.device_type == 'npu':
            # 检查 bfloat16 支持
            try:
                test = torch.tensor([1.0], dtype=torch.bfloat16).to(self.device)
                return torch.bfloat16
            except:
                print("Warning: bfloat16 not supported on NPU, using float16")
                return torch.float16
        else:
            return torch.bfloat16

    def create_generator(self, seed):
        """创建随机生成器"""
        return torch.Generator(self.device).manual_seed(seed)

    def memory_stats(self):
        """打印内存统计"""
        if self.device_type == 'npu':
            allocated = torch.npu.memory_allocated() / 1024**3
            reserved = torch.npu.memory_reserved() / 1024**3
            print(f"NPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        elif self.device_type == 'cuda':
            print(torch.cuda.memory_summary(abbreviated=True))
        else:
            print("CPU device - no GPU memory tracking")

# 使用示例
dm = DeviceManager()
print(f"Using device: {dm.device} with dtype: {dm.dtype}")

# 在模型加载时使用
pipe = pipe.to(dm.device, dm.dtype)
generator = dm.create_generator(seed=42)
```

---

### 七、测试和验证

#### 7.1 逐步测试

```bash
# 1. 测试基础环境
python -c "import torch; import torch_npu; print(torch.npu.is_available())"

# 2. 测试文本编码器
python test_text_encoders_npu.py

# 3. 测试 VAE
python test_vae_npu.py

# 4. 测试完整推理
python inference_e1_1_npu.py
```

#### 7.2 性能对比

```python
import time

# 记录推理时间
start = time.time()
image = pipe(
    prompt=instruction,
    image=test_image,
    num_inference_steps=28,
    ...
).images[0]
end = time.time()

print(f"Inference time: {end - start:.2f}s")
dm.memory_stats()
```

---

### 八、常见问题和解决方案

#### 8.1 算子不支持

**错误**: `RuntimeError: operator XXX is not implemented for NPU`

**解决**:
```python
# 方案1: 回退到CPU
os.environ['NPU_FALLBACK_MODE'] = '1'

# 方案2: 查找替代算子
# 联系华为技术支持获取算子支持列表
```

#### 8.2 精度问题

**现象**: 输出图像质量明显下降

**检查**:
```python
# 确认使用的精度
print(f"Model dtype: {pipe.transformer.dtype}")
print(f"VAE dtype: {pipe.vae.dtype}")

# 尝试混合精度
pipe.vae = pipe.vae.to(torch.float32)  # VAE 用 fp32
```

#### 8.3 内存溢出

**解决**:
```python
# 启用 CPU offload
pipe.enable_model_cpu_offload()

# 或启用序列化 CPU offload
pipe.enable_sequential_cpu_offload()

# 减少 batch size 或分辨率
```

#### 8.4 Transformers 库版本兼容

```bash
# 可能需要特定版本的 transformers
pip install transformers==4.36.0  # 测试不同版本

# 检查 Llama 模型加载
python -c "from transformers import LlamaForCausalLM; print('OK')"
```

---

### 九、性能优化建议

#### 9.1 使用图编译（如果支持）

```python
# 尝试使用昇腾的图编译功能
import torch_npu
from torch_npu.contrib import transfer_to_npu

# 可能需要对模型进行追踪编译
# 参考 CANN 文档的图编译部分
```

#### 9.2 混合精度训练/推理

```python
from torch.cuda.amp import autocast  # 或 torch_npu 的等效实现

with autocast('npu'):  # 可能需要使用 torch_npu 的 API
    output = pipe(...)
```

#### 9.3 批处理优化

```python
# 如果处理多张图，使用批处理
images = [img1, img2, img3]
prompts = ["edit1", "edit2", "edit3"]

# 批量推理
outputs = pipe(
    prompt=prompts,
    image=images,
    ...
)
```

---

### 十、推荐迁移步骤

1. **环境准备** (1-2天)
   - 安装 CANN、torch_npu
   - 验证基础功能

2. **代码基础适配** (1天)
   - 替换所有 "cuda" 为设备变量
   - 使用 DeviceManager 统一管理

3. **依赖库测试** (2-3天)
   - 测试 Diffusers 兼容性
   - 测试 Transformers Llama 模型
   - 解决 Flash Attention 问题

4. **功能验证** (2天)
   - 端到端推理测试
   - 对比输出质量
   - 性能基准测试

5. **优化调优** (按需)
   - 性能优化
   - 内存优化
   - 批处理优化

---

### 十一、联系支持

- **华为昇腾社区**: https://www.hiascend.com/forum
- **CANN 文档**: https://www.hiascend.com/document
- **PyTorch NPU 插件**: https://gitee.com/ascend/pytorch

---

## 📝 检查清单

迁移前请确认：

- [ ] 已安装 CANN toolkit 和 kernels
- [ ] 已安装匹配版本的 torch 和 torch_npu
- [ ] NPU 设备可正常识别 (`torch.npu.is_available()`)
- [ ] 确认 bfloat16/float16 支持情况
- [ ] 准备好测试图片和对比基准
- [ ] 了解 Flash Attention 的替代方案
- [ ] 备份原始 CUDA 代码

迁移后请验证：

- [ ] 模型能成功加载到 NPU
- [ ] 推理能正常完成（至少使用 float32）
- [ ] 输出质量与 CUDA 版本对比可接受
- [ ] 内存使用在可控范围
- [ ] 推理速度满足需求

---

## ⚠️ 重要提示

1. **不要期望零修改运行** - NPU 和 CUDA 有本质区别
2. **从简单到复杂** - 先测试单个组件再测试完整流程
3. **保留 CUDA 版本** - 作为功能和性能对比基准
4. **关注算子覆盖率** - 某些高级算子可能不支持
5. **性能可能不同** - 首次迁移重点是功能，性能需要后续优化

祝迁移顺利！
