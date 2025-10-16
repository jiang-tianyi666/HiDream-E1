#!/usr/bin/env python
"""
NPU 环境测试脚本
用于验证 NPU 环境是否正确配置
"""

import sys
import torch

def test_pytorch():
    """测试 PyTorch 安装"""
    print("=" * 60)
    print("1. Testing PyTorch Installation")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
    print()

def test_npu():
    """测试 NPU 安装和可用性"""
    print("=" * 60)
    print("2. Testing NPU Installation")
    print("=" * 60)
    try:
        import torch_npu
        print("✓ torch_npu imported successfully")
        print(f"  torch_npu version: {torch_npu.__version__ if hasattr(torch_npu, '__version__') else 'unknown'}")

        npu_available = torch.npu.is_available()
        print(f"  NPU available: {npu_available}")

        if npu_available:
            print(f"  NPU count: {torch.npu.device_count()}")
            print(f"  Current NPU: {torch.npu.current_device()}")

            # 测试 NPU 基本操作
            try:
                test_tensor = torch.randn(10, 10).to("npu:0")
                result = test_tensor @ test_tensor.T
                print("  ✓ Basic NPU operations work")
            except Exception as e:
                print(f"  ✗ NPU operations failed: {e}")
        else:
            print("  ⚠️  NPU is not available!")

    except ImportError:
        print("✗ torch_npu not installed")
        print("  Please install: pip install torch_npu")
    print()

def test_device_manager():
    """测试设备管理器"""
    print("=" * 60)
    print("3. Testing Device Manager")
    print("=" * 60)
    try:
        from device_utils import DeviceManager

        dm = DeviceManager()
        print(f"✓ DeviceManager initialized")
        print(f"  Device type: {dm.device_type}")
        print(f"  Device: {dm.device}")
        print(f"  Data type: {dm.dtype}")

        # 测试生成器
        gen = dm.create_generator(42)
        print(f"  ✓ Generator created: {type(gen).__name__}")

        # 测试张量操作
        test_tensor = torch.randn(100, 100).to(dm.device, dm.dtype)
        result = test_tensor @ test_tensor.T
        print(f"  ✓ Tensor operations work (shape: {result.shape})")

        # 内存统计
        print("\n  Memory Stats:")
        dm.memory_stats()

    except ImportError as e:
        print(f"✗ Failed to import device_utils: {e}")
    except Exception as e:
        print(f"✗ DeviceManager test failed: {e}")
        import traceback
        traceback.print_exc()
    print()

def test_dependencies():
    """测试关键依赖"""
    print("=" * 60)
    print("4. Testing Dependencies")
    print("=" * 60)

    dependencies = [
        ("transformers", "Transformers"),
        ("diffusers", "Diffusers"),
        ("PIL", "Pillow"),
        ("safetensors", "SafeTensors"),
        ("peft", "PEFT"),
        ("einops", "Einops"),
        ("accelerate", "Accelerate"),
    ]

    for module_name, display_name in dependencies:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"✓ {display_name:15s} {version}")
        except ImportError:
            print(f"✗ {display_name:15s} NOT INSTALLED")
    print()

def test_precision_support():
    """测试精度支持"""
    print("=" * 60)
    print("5. Testing Precision Support")
    print("=" * 60)

    try:
        from device_utils import DeviceManager
        dm = DeviceManager()
        device = dm.device

        precisions = [
            (torch.float32, "float32"),
            (torch.float16, "float16"),
            (torch.bfloat16, "bfloat16"),
        ]

        for dtype, name in precisions:
            try:
                test = torch.tensor([1.0, 2.0, 3.0], dtype=dtype).to(device)
                result = test * 2
                print(f"✓ {name:10s} supported")
            except Exception as e:
                print(f"✗ {name:10s} not supported: {e}")

    except Exception as e:
        print(f"✗ Precision test failed: {e}")
    print()

def test_model_loading():
    """测试模型加载（简化测试）"""
    print("=" * 60)
    print("6. Testing Model Loading (Basic)")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer

        print("Testing tokenizer download...")
        # 使用一个小模型测试
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        print("✓ Model download and loading works")

    except Exception as e:
        print(f"⚠️  Model loading test failed: {e}")
        print("   This may be due to network issues or missing HuggingFace login")
    print()

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print(" NPU Environment Test Suite")
    print("=" * 60 + "\n")

    test_pytorch()
    test_npu()
    test_device_manager()
    test_dependencies()
    test_precision_support()
    test_model_loading()

    print("=" * 60)
    print(" Test Summary")
    print("=" * 60)

    # 判断是否可以使用 NPU
    try:
        import torch_npu
        if torch.npu.is_available():
            print("✅ NPU is ready to use!")
            print("   You can run: python inference_e1_1_npu.py")
        else:
            print("⚠️  NPU is installed but not available")
            print("   Check your CANN installation and NPU drivers")
    except ImportError:
        print("⚠️  torch_npu not installed")
        print("   Install it from: https://gitee.com/ascend/pytorch")

    # 判断是否可以使用 CUDA 作为替代
    if torch.cuda.is_available():
        print("\n💡 CUDA is available as fallback")
        print("   Set: export PREFERRED_DEVICE=cuda")

    print("\n" + "=" * 60)
    print(" For detailed NPU migration guide, see:")
    print("   - NPU_MIGRATION_GUIDE.md")
    print("   - README_NPU.md")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
