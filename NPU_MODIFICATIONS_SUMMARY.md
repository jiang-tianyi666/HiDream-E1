# NPU 适配修改总结

本文档总结了为支持华为昇腾 NPU 所做的所有代码修改。

## 📦 新增文件

### 1. 核心工具
- **device_utils.py** - 设备管理工具类
  - 自动检测 NPU/CUDA/CPU
  - 自动选择最佳数据类型（bfloat16/float16/float32）
  - 统一的内存管理和监控接口
  - 随机生成器创建
  - SafeTensors 加载辅助

### 2. NPU 版本脚本
- **inference_e1_1_npu.py** - E1.1 版本 NPU 推理脚本
  - 支持自动设备检测
  - 集成设备管理器
  - 禁用 Flash Attention
  - 添加性能统计

- **inference_npu.py** - E1-Full 版本 NPU 推理脚本
  - 支持 LoRA 模型
  - 指令优化功能
  - NPU 兼容的权重加载

- **gradio_demo_npu.py** - NPU 版本交互界面
  - Web 界面支持
  - 实时设备状态显示
  - 批量图像处理

### 3. 配置和文档
- **requirements_npu.txt** - NPU 专用依赖列表
- **README_NPU.md** - NPU 使用指南
- **NPU_MIGRATION_GUIDE.md** - 详细的迁移技术文档
- **test_npu_setup.py** - 环境测试脚本

## 🔧 核心修改点

### 1. 设备字符串替换

**原代码模式：**
```python
pipe.to("cuda", torch.bfloat16)
generator = torch.Generator("cuda").manual_seed(seed)
```

**修改后：**
```python
from device_utils import DeviceManager
dm = DeviceManager()

pipe.to(dm.device, dm.dtype)
generator = dm.create_generator(seed)
```

**修改位置：**
- inference_e1_1.py → inference_e1_1_npu.py
- inference.py → inference_npu.py
- gradio_demo_1_1.py → gradio_demo_npu.py

### 2. 内存监控适配

**原代码：**
```python
torch.cuda.memory_summary(device='cuda', abbreviated=True)
```

**修改后：**
```python
dm.memory_stats()  # 自动适配 NPU/CUDA/CPU
```

### 3. Flash Attention 处理

**添加的代码：**
```python
import os
os.environ.setdefault("DIFFUSERS_ATTENTION_TYPE", "vanilla")
```

**原因：** NPU 不支持 CUDA Flash Attention，需要回退到标准 attention

### 4. 权重加载优化

**原代码：**
```python
lora_ckpt = load_file(lora_ckpt_path, device="cuda")
```

**修改后：**
```python
lora_ckpt = dm.load_safetensors(lora_ckpt_path)
# 先加载到 CPU，后续自动转移到目标设备
```

### 5. 数据类型自动检测

**原代码：**
```python
text_encoder = LlamaForCausalLM.from_pretrained(
    ...,
    torch_dtype=torch.bfloat16
)
```

**修改后：**
```python
text_encoder = LlamaForCausalLM.from_pretrained(
    ...,
    torch_dtype=DTYPE  # 自动检测的数据类型
)
```

## 📊 DeviceManager 类功能

```python
class DeviceManager:
    def __init__(self, preferred_device=None):
        """初始化，可指定设备或自动检测"""

    def create_generator(self, seed):
        """创建设备兼容的随机生成器"""

    def memory_stats(self):
        """显示内存统计（NPU/CUDA/CPU 自适应）"""

    def empty_cache(self):
        """清空设备缓存"""

    def load_safetensors(self, path):
        """加载 safetensors 文件到正确设备"""

    def set_memory_efficient_mode(self):
        """启用内存优化模式"""
```

## 🎯 使用方式对比

### 原始版本（仅支持 CUDA）
```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python inference_e1_1.py
```

### NPU 版本（支持 NPU/CUDA/CPU）
```bash
# 1. 安装 CANN 和 torch_npu（仅 NPU 需要）
# 2. 安装依赖
pip install -r requirements_npu.txt

# 3. 测试环境
python test_npu_setup.py

# 4. 运行推理（自动检测设备）
python inference_e1_1_npu.py

# 或指定设备
export PREFERRED_DEVICE=npu
python inference_e1_1_npu.py
```

## 🔍 关键设计决策

### 1. 为什么创建新文件而不是修改原文件？

**优点：**
- ✅ 保留原始 CUDA 版本作为参考
- ✅ 方便对比和调试
- ✅ 不影响原有用户
- ✅ 可以并存两个版本

**缺点：**
- ❌ 代码有一定重复
- ❌ 需要维护两套代码

### 2. 为什么使用 DeviceManager 类？

**优点：**
- ✅ 统一的设备管理接口
- ✅ 自动检测和回退机制
- ✅ 易于扩展（未来支持更多设备）
- ✅ 代码复用性高

### 3. 为什么禁用 Flash Attention？

- NPU 不支持 CUDA Flash Attention
- 标准 attention 更兼容
- 性能损失可接受（30-50%）
- 未来可能有 NPU 优化版本

## 📝 完整修改清单

### A. 设备相关修改（15+ 处）

| 文件 | 行号 | 原代码 | 修改后 |
|-----|------|--------|--------|
| inference_e1_1.py | 107 | `.to("cuda", torch.bfloat16)` | `.to(DEVICE, DTYPE)` |
| inference_e1_1.py | 153 | `torch.Generator("cuda")` | `dm.create_generator(seed)` |
| inference_e1_1.py | 110 | `torch.cuda.memory_summary(...)` | `dm.memory_stats()` |
| inference.py | 38 | `load_file(..., device="cuda")` | `dm.load_safetensors(...)` |
| inference.py | 64 | `.to("cuda", torch.bfloat16)` | `.to(DEVICE, DTYPE)` |
| inference.py | 88 | `torch.Generator("cuda")` | `dm.create_generator(seed)` |
| gradio_demo_1_1.py | 107 | `.to("cuda", torch.bfloat16)` | `.to(DEVICE, DTYPE)` |
| gradio_demo_1_1.py | 125 | `torch.Generator("cuda")` | `dm.create_generator(seed)` |
| ... | ... | ... | ... |

### B. 新增代码（所有 NPU 文件）

| 文件 | 新增内容 |
|-----|---------|
| 所有 NPU 脚本 | `from device_utils import DeviceManager` |
| 所有 NPU 脚本 | `os.environ.setdefault("DIFFUSERS_ATTENTION_TYPE", "vanilla")` |
| 所有 NPU 脚本 | 设备管理器初始化 |
| 所有 NPU 脚本 | 性能统计和日志 |

### C. 配置修改

- requirements_npu.txt：移除 flash-attn 依赖
- 环境变量支持：PREFERRED_DEVICE

## 🧪 测试覆盖

### test_npu_setup.py 测试项

1. ✅ PyTorch 安装验证
2. ✅ NPU 可用性检测
3. ✅ DeviceManager 功能测试
4. ✅ 依赖库安装检查
5. ✅ 精度支持测试（float32/float16/bfloat16）
6. ✅ 模型下载测试

## 📈 性能影响

### 预期性能变化

| 组件 | CUDA | NPU（无优化） | 说明 |
|------|------|---------------|------|
| Attention | Flash Attention | 标准 Attention | -30-50% 速度 |
| Transformer | 优化算子 | 可能需要回退 | 取决于算子支持 |
| VAE | CUDA 优化 | NPU 原生 | 性能相近 |
| 总体推理 | 基准 | 70-85% | 首次迁移预期 |

### 优化潜力

- 🔧 使用昇腾优化的 Attention（如果可用）
- 🔧 图编译和算子融合
- 🔧 混合精度优化
- 🔧 批处理优化

## 🚀 后续改进方向

### 短期（1-2周）
- [ ] 寻找 NPU 优化的 Attention 实现
- [ ] 性能基准测试和对比
- [ ] 添加更多测试用例
- [ ] 优化内存使用

### 中期（1个月）
- [ ] 探索图编译优化
- [ ] 算子级性能分析
- [ ] 支持更多昇腾设备型号
- [ ] 文档完善

### 长期（2-3个月）
- [ ] 原地修改原始文件（可选）
- [ ] 性能优化到接近 CUDA 水平
- [ ] 支持分布式推理
- [ ] 社区反馈整合

## 📚 参考资源

- **华为昇腾文档**: https://www.hiascend.com/document
- **torch_npu 仓库**: https://gitee.com/ascend/pytorch
- **CANN 开发指南**: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition
- **昇腾社区论坛**: https://www.hiascend.com/forum

## ✅ 验收标准

代码修改成功的标准：

1. ✅ 在 NPU 环境下能成功加载模型
2. ✅ 能完成端到端推理
3. ✅ 输出图像质量与 CUDA 版本相当
4. ✅ 无严重内存泄漏
5. ✅ 错误处理健壮（算子不支持时能回退）
6. ✅ 日志清晰，便于调试
7. ✅ 文档完整，易于使用

---

## 联系方式

如有问题或建议，请：
1. 查看 README_NPU.md 使用指南
2. 查看 NPU_MIGRATION_GUIDE.md 技术文档
3. 运行 test_npu_setup.py 诊断环境
4. 提交 Issue 或联系开发团队

最后更新：2025-10-16
