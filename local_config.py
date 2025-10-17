"""
本地模型路径配置文件
Local Model Paths Configuration

使用说明:
1. 修改 MODEL_BASE_DIR 为你的模型存储目录
2. 确保目录结构正确
3. 运行 python local_config.py 验证路径
"""
import os

# ========== 配置你的模型基础目录 ==========
# 默认路径，根据实际情况修改
MODEL_BASE_DIR = "/home/ma-user/work/jiangtianyi/HiDream-E1/hidream_models"

# 如果模型在其他位置，修改这里，例如：
# MODEL_BASE_DIR = "/data/models/hidream"
# MODEL_BASE_DIR = "/mnt/shared/models"
# ==========================================

# ========== 模型子目录路径 ==========
LLAMA_PATH = os.path.join(MODEL_BASE_DIR, "Llama-3.1-8B-Instruct")
HIDREAM_I1_PATH = os.path.join(MODEL_BASE_DIR, "HiDream-I1-Full")
HIDREAM_E1_PATH = os.path.join(MODEL_BASE_DIR, "HiDream-E1-1")
# ====================================

def verify_models(verbose=True):
    """
    验证所有模型路径是否存在

    Returns:
        bool: 所有模型都存在返回 True，否则返回 False
    """
    models = {
        "Llama-3.1-8B-Instruct": LLAMA_PATH,
        "HiDream-I1-Full": HIDREAM_I1_PATH,
        "HiDream-E1-1": HIDREAM_E1_PATH,
    }

    all_exist = True

    if verbose:
        print("\n" + "="*70)
        print("检查本地模型路径")
        print("="*70)
        print(f"基础目录: {MODEL_BASE_DIR}")
        print("-"*70)

    for name, path in models.items():
        exists = os.path.exists(path)

        if verbose:
            status = "✓" if exists else "✗"
            print(f"{status} {name:25s} -> {path}")

            # 检查关键文件
            if exists:
                if name == "Llama-3.1-8B-Instruct":
                    config_file = os.path.join(path, "config.json")
                    tokenizer_file = os.path.join(path, "tokenizer.json")
                    if os.path.exists(config_file) and os.path.exists(tokenizer_file):
                        print(f"  └─ config.json ✓  tokenizer.json ✓")
                    else:
                        print(f"  └─ ⚠️  缺少必要文件")
                        all_exist = False
                else:
                    transformer_dir = os.path.join(path, "transformer")
                    if os.path.exists(transformer_dir):
                        print(f"  └─ transformer/ ✓")
                    else:
                        print(f"  └─ ⚠️  缺少 transformer 目录")
                        all_exist = False

        if not exists:
            all_exist = False

    if verbose:
        print("="*70)
        if all_exist:
            print("✓ 所有模型路径正确且文件完整")
        else:
            print("✗ 部分模型未找到或文件不完整")
            print("\n建议:")
            print("1. 检查 MODEL_BASE_DIR 路径是否正确")
            print("2. 确认模型已下载并上传到服务器")
            print("3. 查看 LOCAL_MODELS_GUIDE.md 获取下载指南")
        print("="*70 + "\n")

    return all_exist


def get_model_paths():
    """
    返回模型路径字典

    Returns:
        dict: 包含所有模型路径的字典
    """
    return {
        "llama": LLAMA_PATH,
        "hidream_i1": HIDREAM_I1_PATH,
        "hidream_e1": HIDREAM_E1_PATH,
    }


def print_usage():
    """打印使用说明"""
    print("\n" + "="*70)
    print("使用本地模型的方法")
    print("="*70)
    print("\n方法 1: 使用配置文件（推荐）")
    print("-"*70)
    print("在你的脚本中:")
    print("  from local_config import LLAMA_PATH, HIDREAM_I1_PATH, HIDREAM_E1_PATH")
    print("  # 然后直接使用这些路径变量")
    print()
    print("方法 2: 直接修改脚本")
    print("-"*70)
    print("编辑 inference_e1_1_npu.py，修改路径定义:")
    print(f'  LLAMA_PATH = "{LLAMA_PATH}"')
    print(f'  HIDREAM_I1_PATH = "{HIDREAM_I1_PATH}"')
    print(f'  HIDREAM_E1_PATH = "{HIDREAM_E1_PATH}"')
    print("="*70 + "\n")


if __name__ == "__main__":
    # 运行验证
    success = verify_models(verbose=True)

    # 打印使用说明
    if success:
        print_usage()

        # 显示路径信息供复制
        print("可以复制的路径定义:")
        print("-"*70)
        print(f'LLAMA_PATH = "{LLAMA_PATH}"')
        print(f'HIDREAM_I1_PATH = "{HIDREAM_I1_PATH}"')
        print(f'HIDREAM_E1_PATH = "{HIDREAM_E1_PATH}"')
        print("-"*70 + "\n")
    else:
        import sys
        sys.exit(1)
