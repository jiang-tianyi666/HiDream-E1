"""
设备管理工具 - 支持 NPU/CUDA/CPU 自动切换
Device Management Utility - Auto-detection for NPU/CUDA/CPU
"""

import torch
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeviceManager:
    """统一的设备管理器，自动检测并适配 NPU/CUDA/CPU"""

    def __init__(self, preferred_device=None):
        """
        初始化设备管理器

        Args:
            preferred_device: 可选，指定设备类型 'npu', 'cuda', 'cpu'
                            如果为 None，则自动检测
        """
        self.device_type = self._detect_device(preferred_device)
        self.device = self._get_device()
        self.dtype = self._get_dtype()

        logger.info(f"🚀 Device Manager initialized")
        logger.info(f"   Device type: {self.device_type}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Data type: {self.dtype}")

    def _detect_device(self, preferred=None):
        """自动检测可用设备"""
        if preferred:
            preferred = preferred.lower()
            if preferred == 'npu':
                try:
                    import torch_npu
                    if torch.npu.is_available():
                        return 'npu'
                    else:
                        logger.warning("NPU requested but not available, falling back...")
                except ImportError:
                    logger.warning("torch_npu not installed, falling back...")
            elif preferred == 'cuda':
                if torch.cuda.is_available():
                    return 'cuda'
                else:
                    logger.warning("CUDA requested but not available, falling back...")
            elif preferred == 'cpu':
                return 'cpu'

        # 自动检测顺序：NPU > CUDA > CPU
        try:
            import torch_npu
            if torch.npu.is_available():
                logger.info("✓ NPU detected and available")
                return 'npu'
        except ImportError:
            pass

        if torch.cuda.is_available():
            logger.info("✓ CUDA detected and available")
            return 'cuda'

        logger.info("Using CPU (no GPU detected)")
        return 'cpu'

    def _get_device(self):
        """获取设备字符串"""
        if self.device_type == 'npu':
            try:
                import torch_npu
                device_id = int(os.environ.get('NPU_DEVICE_ID', torch.npu.current_device()))
                return f"npu:{device_id}"
            except:
                return "npu:0"
        elif self.device_type == 'cuda':
            device_id = int(os.environ.get('CUDA_VISIBLE_DEVICES', torch.cuda.current_device()))
            return f"cuda:{device_id}"
        else:
            return "cpu"

    def _get_dtype(self):
        """获取支持的数据类型，优先使用 bfloat16"""
        if self.device_type == 'cpu':
            # CPU 通常不支持 bfloat16 高效运算
            logger.info("Using float32 for CPU")
            return torch.float32

        # 测试 bfloat16 支持
        try:
            test_tensor = torch.tensor([1.0], dtype=torch.bfloat16).to(self.device)
            _ = test_tensor * 2  # 简单运算测试
            logger.info("✓ bfloat16 is supported")
            return torch.bfloat16
        except Exception as e:
            logger.warning(f"bfloat16 not supported ({e}), trying float16...")

            # 尝试 float16
            try:
                test_tensor = torch.tensor([1.0], dtype=torch.float16).to(self.device)
                _ = test_tensor * 2
                logger.info("✓ Using float16 instead")
                return torch.float16
            except Exception as e:
                logger.warning(f"float16 not supported ({e}), using float32")
                return torch.float32

    def create_generator(self, seed):
        """
        创建随机数生成器

        Args:
            seed: 随机种子

        Returns:
            torch.Generator
        """
        try:
            generator = torch.Generator(self.device).manual_seed(int(seed))
            return generator
        except Exception as e:
            logger.warning(f"Failed to create generator on {self.device}: {e}")
            logger.warning("Falling back to CPU generator")
            return torch.Generator("cpu").manual_seed(int(seed))

    def memory_stats(self):
        """打印内存统计信息"""
        if self.device_type == 'npu':
            try:
                import torch_npu
                allocated = torch.npu.memory_allocated() / 1024**3
                reserved = torch.npu.memory_reserved() / 1024**3
                max_allocated = torch.npu.max_memory_allocated() / 1024**3
                logger.info(f"📊 NPU Memory Stats:")
                logger.info(f"   Allocated: {allocated:.2f} GB")
                logger.info(f"   Reserved: {reserved:.2f} GB")
                logger.info(f"   Peak: {max_allocated:.2f} GB")
            except Exception as e:
                logger.warning(f"Failed to get NPU memory stats: {e}")

        elif self.device_type == 'cuda':
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                max_allocated = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(f"📊 CUDA Memory Stats:")
                logger.info(f"   Allocated: {allocated:.2f} GB")
                logger.info(f"   Reserved: {reserved:.2f} GB")
                logger.info(f"   Peak: {max_allocated:.2f} GB")
            except Exception as e:
                logger.warning(f"Failed to get CUDA memory stats: {e}")
        else:
            logger.info("📊 CPU device - no GPU memory tracking")

    def empty_cache(self):
        """清空设备缓存"""
        if self.device_type == 'npu':
            try:
                import torch_npu
                torch.npu.empty_cache()
                logger.info("✓ NPU cache cleared")
            except:
                pass
        elif self.device_type == 'cuda':
            torch.cuda.empty_cache()
            logger.info("✓ CUDA cache cleared")

    def set_memory_efficient_mode(self):
        """设置内存高效模式"""
        if self.device_type == 'npu':
            try:
                import torch_npu
                # NPU 特定优化
                torch_npu.npu.set_compile_mode(jit_compile=False)
                logger.info("✓ NPU memory efficient mode enabled")
            except Exception as e:
                logger.warning(f"Failed to set NPU memory mode: {e}")
        elif self.device_type == 'cuda':
            # CUDA 特定优化
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("✓ CUDA TF32 mode enabled for better performance")

    def load_safetensors(self, path):
        """
        加载 safetensors 文件到正确的设备

        Args:
            path: safetensors 文件路径

        Returns:
            state_dict
        """
        from safetensors.torch import load_file

        # 先加载到 CPU，再转移到目标设备（更安全）
        try:
            state_dict = load_file(path, device="cpu")
            logger.info(f"✓ Loaded {path} to CPU, will transfer to {self.device}")
            return state_dict
        except Exception as e:
            logger.error(f"Failed to load safetensors: {e}")
            raise

    def __repr__(self):
        return f"DeviceManager(device={self.device}, dtype={self.dtype})"


def get_device_manager(preferred_device=None):
    """
    获取全局设备管理器实例（单例模式）

    Args:
        preferred_device: 可选，指定设备类型

    Returns:
        DeviceManager instance
    """
    if not hasattr(get_device_manager, '_instance'):
        get_device_manager._instance = DeviceManager(preferred_device)
    return get_device_manager._instance


# 便捷函数
def auto_device():
    """返回自动检测的设备字符串"""
    dm = get_device_manager()
    return dm.device


def auto_dtype():
    """返回自动检测的数据类型"""
    dm = get_device_manager()
    return dm.dtype


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("Device Manager Test")
    print("=" * 60)

    dm = DeviceManager()
    print(f"\n{dm}")

    # 测试生成器
    print("\n📝 Testing generator...")
    gen = dm.create_generator(42)
    print(f"✓ Generator created: {gen}")

    # 测试内存统计
    print("\n📊 Memory statistics:")
    dm.memory_stats()

    # 测试张量创建
    print("\n🧪 Testing tensor operations...")
    try:
        test_tensor = torch.randn(100, 100).to(dm.device, dm.dtype)
        result = torch.matmul(test_tensor, test_tensor.T)
        print(f"✓ Tensor operation successful")
        print(f"  Shape: {result.shape}, dtype: {result.dtype}, device: {result.device}")
    except Exception as e:
        print(f"✗ Tensor operation failed: {e}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
