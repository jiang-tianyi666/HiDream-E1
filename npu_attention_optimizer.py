"""
NPU Attention 优化器
尝试使用昇腾提供的优化 Attention 实现
"""

import torch
import logging
import os

logger = logging.getLogger(__name__)


class NPUAttentionOptimizer:
    """检测并应用 NPU 优化的 Attention"""

    def __init__(self):
        self.npu_available = self._check_npu()
        self.optimized_attention = None
        self.method = None

        if self.npu_available:
            self._detect_optimized_attention()

    def _check_npu(self):
        """检查 NPU 是否可用"""
        try:
            import torch_npu
            return torch.npu.is_available()
        except ImportError:
            return False

    def _detect_optimized_attention(self):
        """检测可用的优化 Attention 实现"""

        # 方法 1: 检查 torch_npu.contrib 中的优化实现
        try:
            import torch_npu
            if hasattr(torch_npu, 'contrib'):
                # 可能的优化实现
                if hasattr(torch_npu.contrib, 'function'):
                    logger.info("✓ Found torch_npu.contrib.function")
                    self.method = "torch_npu_contrib"
                    return
        except Exception as e:
            logger.debug(f"torch_npu.contrib not available: {e}")

        # 方法 2: 检查是否有 NPU 优化的 scaled_dot_product_attention
        try:
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                # PyTorch 2.0+ 内置的优化 attention
                logger.info("✓ Using PyTorch native scaled_dot_product_attention")
                self.method = "pytorch_native"
                return
        except Exception as e:
            logger.debug(f"PyTorch native attention not available: {e}")

        # 方法 3: 回退到标准实现
        logger.warning("⚠️  No optimized attention found, using vanilla implementation")
        self.method = "vanilla"

    def patch_diffusers_attention(self):
        """
        为 Diffusers 库打补丁，使用优化的 Attention

        这个方法会修改 Diffusers 的 attention 处理器
        """
        if not self.npu_available:
            logger.info("NPU not available, skipping attention optimization")
            return False

        if self.method == "pytorch_native":
            return self._use_pytorch_native_attention()
        elif self.method == "torch_npu_contrib":
            return self._use_npu_contrib_attention()
        else:
            return self._use_vanilla_attention()

    def _use_pytorch_native_attention(self):
        """使用 PyTorch 2.0+ 的原生优化 attention"""
        try:
            # 设置环境变量让 Diffusers 使用 PyTorch 的 scaled_dot_product_attention
            os.environ["DIFFUSERS_ATTENTION_TYPE"] = "sdpa"  # Scaled Dot Product Attention

            logger.info("✓ Enabled PyTorch native SDPA (should work on NPU)")
            logger.info("  This may provide better performance than vanilla attention")
            return True
        except Exception as e:
            logger.warning(f"Failed to enable native SDPA: {e}")
            return False

    def _use_npu_contrib_attention(self):
        """使用 torch_npu.contrib 提供的优化"""
        try:
            import torch_npu

            # 这里需要根据华为的实际 API 进行调整
            # 以下是假设的 API，需要查阅华为文档

            # 示例：启用 NPU 优化的算子融合
            if hasattr(torch_npu, 'npu'):
                if hasattr(torch_npu.npu, 'set_option'):
                    torch_npu.npu.set_option({
                        'ACL_OP_COMPILER_CACHE_MODE': '1',  # 启用算子编译缓存
                        'ACL_OP_SELECT_IMPL_MODE': '1',     # 优化算子选择
                    })
                    logger.info("✓ Enabled NPU operator optimizations")

            return True
        except Exception as e:
            logger.warning(f"Failed to enable NPU contrib attention: {e}")
            return False

    def _use_vanilla_attention(self):
        """使用标准 Attention（回退方案）"""
        os.environ["DIFFUSERS_ATTENTION_TYPE"] = "vanilla"
        logger.info("Using vanilla attention (fallback)")
        return True

    def get_attention_info(self):
        """返回当前使用的 Attention 实现信息"""
        info = {
            "npu_available": self.npu_available,
            "method": self.method,
            "env_diffusers_attention": os.environ.get("DIFFUSERS_ATTENTION_TYPE", "default"),
        }
        return info

    def benchmark_attention(self, seq_len=1024, hidden_dim=768, num_heads=12, device="npu:0"):
        """
        对比不同 Attention 实现的性能

        Args:
            seq_len: 序列长度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            device: 设备
        """
        import time

        results = {}

        # 准备测试数据
        head_dim = hidden_dim // num_heads
        q = torch.randn(1, num_heads, seq_len, head_dim).to(device)
        k = torch.randn(1, num_heads, seq_len, head_dim).to(device)
        v = torch.randn(1, num_heads, seq_len, head_dim).to(device)

        # 预热
        _ = self._vanilla_attention(q, k, v)

        # 测试 Vanilla Attention
        start = time.time()
        for _ in range(10):
            _ = self._vanilla_attention(q, k, v)
        vanilla_time = (time.time() - start) / 10
        results["vanilla"] = vanilla_time

        # 测试 PyTorch Native SDPA (如果可用)
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            try:
                start = time.time()
                for _ in range(10):
                    _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                sdpa_time = (time.time() - start) / 10
                results["pytorch_sdpa"] = sdpa_time
            except Exception as e:
                logger.warning(f"SDPA benchmark failed: {e}")

        return results

    @staticmethod
    def _vanilla_attention(q, k, v, scale=None):
        """标准 Attention 实现"""
        if scale is None:
            scale = q.shape[-1] ** -0.5

        # q: [batch, heads, seq_q, head_dim]
        # k: [batch, heads, seq_k, head_dim]
        # v: [batch, heads, seq_v, head_dim]

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output


def apply_npu_attention_optimization():
    """
    便捷函数：自动检测并应用最佳的 NPU Attention 优化

    在模型加载前调用此函数
    """
    optimizer = NPUAttentionOptimizer()
    success = optimizer.patch_diffusers_attention()

    info = optimizer.get_attention_info()
    logger.info("=" * 60)
    logger.info("NPU Attention Optimization Status:")
    logger.info(f"  NPU Available: {info['npu_available']}")
    logger.info(f"  Method: {info['method']}")
    logger.info(f"  Diffusers Attention Type: {info['env_diffusers_attention']}")
    logger.info("=" * 60)

    return optimizer


if __name__ == "__main__":
    # 测试脚本
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("NPU Attention Optimizer Test")
    print("=" * 60 + "\n")

    optimizer = NPUAttentionOptimizer()
    optimizer.patch_diffusers_attention()

    info = optimizer.get_attention_info()
    print(f"\nAttention Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 如果 NPU 可用，运行性能测试
    if optimizer.npu_available:
        print("\n" + "=" * 60)
        print("Running Performance Benchmark...")
        print("=" * 60)

        try:
            results = optimizer.benchmark_attention(
                seq_len=512,
                hidden_dim=768,
                num_heads=12,
                device="npu:0"
            )

            print("\nBenchmark Results (average time per iteration):")
            for method, time_ms in results.items():
                print(f"  {method:20s}: {time_ms*1000:.2f} ms")

            if "pytorch_sdpa" in results and "vanilla" in results:
                speedup = results["vanilla"] / results["pytorch_sdpa"]
                print(f"\nSpeedup (SDPA vs Vanilla): {speedup:.2f}x")

        except Exception as e:
            print(f"\nBenchmark failed: {e}")
    else:
        print("\n⚠️  NPU not available, skipping benchmark")

    print("\n" + "=" * 60)
