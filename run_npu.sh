#!/bin/bash
# HiDream-E1 NPU 快速启动脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "HiDream-E1 NPU Edition - Quick Start"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查 Python
echo -e "${YELLOW}1. Checking Python...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Python not found${NC}"
    exit 1
fi
PYTHON_VERSION=$(python --version)
echo -e "${GREEN}✓ $PYTHON_VERSION${NC}"
echo ""

# 检查 NPU
echo -e "${YELLOW}2. Checking NPU availability...${NC}"
python -c "
import sys
try:
    import torch
    import torch_npu
    if torch.npu.is_available():
        print('✓ NPU is available')
        print(f'  NPU count: {torch.npu.device_count()}')
        sys.exit(0)
    else:
        print('⚠️  torch_npu installed but NPU not available')
        sys.exit(1)
except ImportError as e:
    print(f'⚠️  torch_npu not installed: {e}')
    print('  Will try to use CUDA or CPU instead')
    sys.exit(0)
"
NPU_STATUS=$?
echo ""

# 检查依赖
echo -e "${YELLOW}3. Checking dependencies...${NC}"
python -c "
import sys
required = ['torch', 'transformers', 'diffusers', 'PIL', 'safetensors', 'peft']
missing = []
for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f'✗ Missing packages: {missing}')
    print('  Run: pip install -r requirements_npu.txt')
    sys.exit(1)
else:
    print('✓ All dependencies installed')
    sys.exit(0)
"
if [ $? -ne 0 ]; then
    echo -e "${RED}Please install dependencies first${NC}"
    exit 1
fi
echo ""

# 检查 HuggingFace 登录
echo -e "${YELLOW}4. Checking HuggingFace login...${NC}"
if ! huggingface-cli whoami &> /dev/null; then
    echo -e "${YELLOW}⚠️  Not logged into HuggingFace${NC}"
    echo -e "   Run: ${GREEN}huggingface-cli login${NC}"
    echo -e "   (Required for Llama-3.1-8B-Instruct model)"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    HF_USER=$(huggingface-cli whoami)
    echo -e "${GREEN}✓ Logged in as: $HF_USER${NC}"
fi
echo ""

# 选择要运行的脚本
echo -e "${YELLOW}5. Select script to run:${NC}"
echo "  1) inference_e1_1_npu.py  (E1.1 - Recommended)"
echo "  2) inference_npu.py       (E1-Full with LoRA)"
echo "  3) gradio_demo_npu.py     (Web Interface)"
echo "  4) test_npu_setup.py      (Environment Test)"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        SCRIPT="inference_e1_1_npu.py"
        ;;
    2)
        SCRIPT="inference_npu.py"
        ;;
    3)
        SCRIPT="gradio_demo_npu.py"
        ;;
    4)
        SCRIPT="test_npu_setup.py"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${YELLOW}6. Running $SCRIPT...${NC}"
echo "=========================================="
echo ""

# 设置环境变量（可选）
# export PREFERRED_DEVICE=npu  # 强制使用 NPU
# export NPU_FALLBACK_MODE=1   # 启用算子回退

# 运行脚本
python "$SCRIPT"

# 检查退出状态
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo -e "${GREEN}✓ Execution completed successfully!${NC}"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo -e "${RED}✗ Execution failed${NC}"
    echo "=========================================="
    echo ""
    echo "Troubleshooting:"
    echo "  1. Run: python test_npu_setup.py"
    echo "  2. Check: README_NPU.md"
    echo "  3. Check: NPU_MIGRATION_GUIDE.md"
    exit 1
fi
