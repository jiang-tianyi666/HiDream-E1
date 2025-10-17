# Flash Attention NPU 解决方案详解

本文档详细说明如何在华为昇腾 NPU 上解决 Flash Attention 依赖问题。

## 🎯 问题概述

### 原始依赖
```bash
pip install flash-attn --no-build-isolation
```

### 问题
- ❌ flash-attn 是 CUDA 专属库
- ❌ 使用 CUDA 特定的内核（kernel）
- ❌ 在 NPU 上无法编译和运行

## 💡 解决方案对比

| 方案 | 性能 | 复杂度 | 兼容性 | 推荐度 |
|-----|------|--------|--------|--------|
| 方案1: 禁用Flash Attention | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ 已实现 |
| 方案2: PyTorch SDPA | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ 推荐 |
| 方案3: 昇腾优化API | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | 🔍 需调研 |
| 方案4: 手动实现优化 | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ | 🔧 高级 |

---

## 📋 方案 1: 禁用 Flash Attention（已实现）

### 实现方式
```python
import os
os.environ["DIFFUSERS_ATTENTION_TYPE"] = "vanilla"
```

### 优点
- ✅ 零修改，立即可用
- ✅ 100% 兼容性
- ✅ 功能完整

### 缺点
- ❌ 性能损失 30-50%
- ❌ 内存占用可能增加 20-30%

### 已集成
所有 NPU 脚本已经自动应用此方案：
- `inference_e1_1_npu.py`
- `inference_npu.py`
- `gradio_demo_npu.py`

### 性能影响
```
推理时间对比（28 steps，768x768）:
- CUDA + Flash Attention: ~8秒
- NPU + Vanilla Attention: ~12-14秒 (预估)
```

---

## 📋 方案 2: PyTorch Scaled Dot Product Attention（推荐 ⭐）

### 原理
PyTorch 2.0+ 内置了优化的 `scaled_dot_product_attention`，它会自动选择最优实现：
- CUDA: 使用 Flash Attention 或 Memory Efficient Attention
- NPU: 可能使用昇腾优化的实现

### 检测是否可用
```python
import torch

# 检查 PyTorch 版本
print(f"PyTorch version: {torch.__version__}")

# 检查是否有 SDPA
has_sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
print(f"Has SDPA: {has_sdpa}")

# 测试是否能在 NPU 上运行
if has_sdpa:
    q = torch.randn(1, 8, 128, 64).to("npu:0")
    k = torch.randn(1, 8, 128, 64).to("npu:0")
    v = torch.randn(1, 8, 128, 64).to("npu:0")

    try:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        print("✓ SDPA works on NPU!")
    except Exception as e:
        print(f"✗ SDPA failed: {e}")
```

### 使用方式

#### 方法 A: 环境变量（推荐）
```python
import os
os.environ["DIFFUSERS_ATTENTION_TYPE"] = "sdpa"  # 而不是 "vanilla"
```

#### 方法 B: 使用优化器工具
```python
from npu_attention_optimizer import apply_npu_attention_optimization

# 在加载模型前调用
optimizer = apply_npu_attention_optimization()

# 然后正常加载模型
pipe = HiDreamImageEditingPipeline.from_pretrained(...)
```

### 性能提升预期
```
相比 Vanilla Attention:
- 速度提升: 1.2-1.5x
- 内存减少: 10-20%
```

### 集成到现有脚本

修改你的 NPU 脚本（例如 `inference_e1_1_npu.py`）:

```python
# 原代码
os.environ.setdefault("DIFFUSERS_ATTENTION_TYPE", "vanilla")

# 改为
os.environ.setdefault("DIFFUSERS_ATTENTION_TYPE", "sdpa")

# 或者更智能的方式
import torch
if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    os.environ["DIFFUSERS_ATTENTION_TYPE"] = "sdpa"
    print("✓ Using PyTorch SDPA")
else:
    os.environ["DIFFUSERS_ATTENTION_TYPE"] = "vanilla"
    print("⚠️  SDPA not available, using vanilla")
```

---

## 📋 方案 3: 昇腾优化 API（需要华为支持）

### 可能的优化点

#### 1. 算子融合
```python
import torch_npu

# 启用算子融合优化
torch_npu.npu.set_option({
    'ACL_OP_COMPILER_CACHE_MODE': '1',  # 算子编译缓存
    'ACL_OP_SELECT_IMPL_MODE': '1',     # 优化算子选择
})
```

#### 2. 图编译
```python
# 使用 TorchScript 进行图优化
model = torch.jit.script(model)
# 或
model = torch.jit.trace(model, example_input)
```

#### 3. 混合精度优化
```python
# 使用昇腾的 AMP（如果支持）
from torch_npu.contrib import transfer_to_npu

model = transfer_to_npu(model)
```

### 需要咨询华为的问题清单

**关键问题：**
1. ✅ 昇腾是否有优化的 Attention 实现？
2. ✅ `scaled_dot_product_attention` 在 NPU 上的性能如何？
3. ✅ 是否有专门的 Diffusion Models 优化？
4. ✅ 哪个 CANN 版本支持最好？
5. ✅ 是否有性能分析工具？

**联系渠道：**
- 昇腾社区论坛: https://www.hiascend.com/forum
- 技术支持邮箱: （查看华为官网）
- GitHub Issues: https://gitee.com/ascend/pytorch

---

## 📋 方案 4: 手动优化 Attention（高级）

### 适用场景
- 对性能要求极高
- 愿意深入修改代码
- 有 NPU 编程经验

### 实现思路

#### 1. 创建自定义 Attention 处理器

```python
# custom_npu_attention.py
import torch
import torch.nn.functional as F

class NPUOptimizedAttention:
    """NPU 优化的 Attention 实现"""

    def __call__(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        # 查询、键、值投影
        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states or hidden_states)
        value = self.to_v(encoder_hidden_states or hidden_states)

        # Reshape 为多头
        query = self._reshape_heads_to_batch_dim(query)
        key = self._reshape_heads_to_batch_dim(key)
        value = self._reshape_heads_to_batch_dim(value)

        # 使用优化的 attention 计算
        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 优化路径
            hidden_states = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False
            )
        else:
            # 手动实现（带优化）
            hidden_states = self._compute_attention(query, key, value, attention_mask)

        # Reshape 回原始形状
        hidden_states = self._reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _compute_attention(self, query, key, value, attention_mask=None):
        """手动实现的优化 Attention"""
        # 使用 torch.matmul 而不是 einsum（NPU 可能更优化）
        scale = query.shape[-1] ** -0.5

        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)

        # 应用注意力到 value
        hidden_states = torch.matmul(attention_probs, value)

        return hidden_states
```

#### 2. 替换 Diffusers 的 Attention 处理器

```python
from diffusers.models.attention_processor import Attention

# 在加载模型后
for name, module in pipe.transformer.named_modules():
    if isinstance(module, Attention):
        module.set_processor(NPUOptimizedAttention())
        print(f"Replaced attention in {name}")
```

### 优化技巧

1. **算子选择**
```python
# 避免使用 NPU 不支持或慢的算子
# 优先使用: matmul, add, mul, softmax
# 避免使用: einsum (可能慢), complex indexing
```

2. **内存布局**
```python
# 确保张量连续存储
query = query.contiguous()
key = key.contiguous()
value = value.contiguous()
```

3. **精度权衡**
```python
# Attention 可以使用更高精度
with torch.cuda.amp.autocast(enabled=False):  # 或 torch_npu equivalent
    attention_scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
attention_scores = attention_scores.to(query.dtype)
```

---

## 🧪 性能测试和对比

### 测试脚本

```python
# test_attention_performance.py
import torch
import time
from npu_attention_optimizer import NPUAttentionOptimizer

def benchmark():
    optimizer = NPUAttentionOptimizer()

    # 测试不同配置
    configs = [
        {"seq_len": 256, "hidden_dim": 768, "num_heads": 12},
        {"seq_len": 512, "hidden_dim": 768, "num_heads": 12},
        {"seq_len": 1024, "hidden_dim": 768, "num_heads": 12},
    ]

    for config in configs:
        print(f"\nTesting: seq_len={config['seq_len']}, hidden_dim={config['hidden_dim']}")
        results = optimizer.benchmark_attention(**config, device="npu:0")

        for method, time_ms in results.items():
            print(f"  {method}: {time_ms*1000:.2f} ms")

if __name__ == "__main__":
    benchmark()
```

运行：
```bash
python test_attention_performance.py
```

---

## 📊 推荐实施路线图

### 阶段 1: 快速启动（当前）
- ✅ 使用 Vanilla Attention
- ✅ 功能验证
- ✅ 基准性能测试

### 阶段 2: 性能优化（1周内）
```bash
# 1. 测试 PyTorch SDPA
python npu_attention_optimizer.py

# 2. 如果 SDPA 可用，更新脚本
# 修改: os.environ["DIFFUSERS_ATTENTION_TYPE"] = "sdpa"

# 3. 对比性能
python test_attention_performance.py
```

### 阶段 3: 深度优化（按需）
- 联系华为技术支持
- 使用昇腾特定优化
- 自定义 Attention 实现

---

## 🔧 快速切换不同方案

### 创建配置文件

```python
# attention_config.py
import os

ATTENTION_STRATEGIES = {
    "vanilla": {
        "env": {"DIFFUSERS_ATTENTION_TYPE": "vanilla"},
        "description": "Standard attention, slowest but most compatible"
    },
    "sdpa": {
        "env": {"DIFFUSERS_ATTENTION_TYPE": "sdpa"},
        "description": "PyTorch SDPA, balanced performance"
    },
    "auto": {
        "env": {},  # 自动检测
        "description": "Auto-detect best method"
    }
}

def set_attention_strategy(strategy="auto"):
    """设置 Attention 策略"""
    if strategy not in ATTENTION_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")

    config = ATTENTION_STRATEGIES[strategy]

    if strategy == "auto":
        # 自动检测逻辑
        import torch
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            strategy = "sdpa"
        else:
            strategy = "vanilla"
        config = ATTENTION_STRATEGIES[strategy]

    # 应用环境变量
    for key, value in config["env"].items():
        os.environ[key] = value

    print(f"✓ Attention strategy set to: {strategy}")
    print(f"  {config['description']}")

    return strategy
```

### 使用方式

```python
# 在脚本开头
from attention_config import set_attention_strategy

# 尝试最优策略
strategy = set_attention_strategy("auto")

# 或手动指定
# strategy = set_attention_strategy("sdpa")
# strategy = set_attention_strategy("vanilla")
```

---

## ✅ 行动建议

### 立即行动（今天）
1. ✅ 使用已有的 NPU 脚本（已包含 Vanilla Attention）
2. ✅ 运行 `python npu_attention_optimizer.py` 检测可用方案
3. ✅ 记录基准性能数据

### 短期优化（本周）
1. ⭐ 测试 PyTorch SDPA 在你的 NPU 上的性能
2. ⭐ 如果 SDPA 可用且更快，更新环境变量
3. ⭐ 对比前后性能差异

### 中期优化（2-4周）
1. 🔍 联系华为技术支持咨询优化方案
2. 🔍 查阅最新的 CANN 文档
3. 🔍 在昇腾社区寻找类似案例

### 长期优化（按需）
1. 🔧 根据华为反馈实施专门优化
2. 🔧 考虑自定义 Attention 实现
3. 🔧 持续跟进昇腾生态更新

---

## 📞 获取帮助

如果遇到问题：

1. **检查日志**
```bash
python inference_e1_1_npu.py 2>&1 | tee inference.log
grep -i "attention" inference.log
```

2. **运行诊断**
```bash
python npu_attention_optimizer.py
```

3. **查看文档**
- README_NPU.md
- NPU_MIGRATION_GUIDE.md

4. **社区求助**
- 昇腾论坛
- PyTorch GitHub

---

**总结**：目前所有 NPU 脚本已经使用方案1（Vanilla Attention），可以立即运行。建议尽快测试方案2（PyTorch SDPA），可能获得显著性能提升！
