import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from pipeline_hidream_image_editing import HiDreamImageEditingPipeline
from PIL import Image
from peft import LoraConfig
from huggingface_hub import hf_hub_download
from diffusers import HiDreamImageTransformer2DModel
from instruction_refinement import refine_instruction
from safetensors.torch import load_file
import os
import logging

# ============ NPU 支持 ============
from device_utils import DeviceManager
# ==================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# ============ 初始化设备管理器 ============
preferred_device = os.environ.get('PREFERRED_DEVICE', None)
device_manager = DeviceManager(preferred_device=preferred_device)
DEVICE = device_manager.device
DTYPE = device_manager.dtype
# =========================================

# ============ 模型路径配置 ============
try:
    from local_config import LLAMA_PATH, verify_models
    if verify_models(verbose=False):
        LLAMA_MODEL_PATH = LLAMA_PATH
        logging.info("📦 使用本地 Llama 模型")
    else:
        raise FileNotFoundError("本地模型路径验证失败")
except (ImportError, FileNotFoundError):
    LLAMA_MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
    logging.info("将从 HuggingFace 下载 Llama 模型")
# ======================================

# Flash Attention 处理
os.environ.setdefault("DIFFUSERS_ATTENTION_TYPE", "vanilla")
logging.info(f"Attention type: {os.environ.get('DIFFUSERS_ATTENTION_TYPE', 'default')}")

# Set to True to enable instruction refinement and transformer model
ENABLE_REFINE = True

# Load models
logging.info("=" * 60)
logging.info("Loading models...")
logging.info("=" * 60)

tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(LLAMA_MODEL_PATH)
text_encoder_4 = LlamaForCausalLM.from_pretrained(
    LLAMA_MODEL_PATH,
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=DTYPE,  # 使用自动检测的数据类型
)
logging.info("✓ Llama model loaded")

# Configure transformer model if refinement is enabled
transformer = None
reload_keys = None
if ENABLE_REFINE:
    logging.info("Loading transformer with LoRA for refinement...")
    transformer = HiDreamImageTransformer2DModel.from_pretrained("HiDream-ai/HiDream-I1-Full", subfolder="transformer")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["to_k", "to_q", "to_v", "to_out", "to_k_t", "to_q_t", "to_v_t", "to_out_t", "w1", "w2", "w3", "final_layer.linear"],
        init_lora_weights="gaussian",
    )
    transformer.add_adapter(lora_config)
    transformer.max_seq = 4608

    logging.info("Downloading LoRA checkpoint...")
    lora_ckpt_path = hf_hub_download(repo_id="HiDream-ai/HiDream-E1-Full", filename="HiDream-E1-Full.safetensors")

    # 使用设备管理器加载权重（先加载到 CPU）
    lora_ckpt = device_manager.load_safetensors(lora_ckpt_path)

    src_state_dict = transformer.state_dict()
    reload_keys_list = [k for k in lora_ckpt if "lora" not in k]
    reload_keys = {
        "editing": {k: v for k, v in lora_ckpt.items() if k in reload_keys_list},
        "refine": {k: v for k, v in src_state_dict.items() if k in reload_keys_list},
    }
    info = transformer.load_state_dict(lora_ckpt, strict=False)
    assert len(info.unexpected_keys) == 0
    logging.info("✓ LoRA weights loaded")

# Initialize pipeline
logging.info("Creating pipeline...")
if ENABLE_REFINE:
    pipe = HiDreamImageEditingPipeline.from_pretrained(
        "HiDream-ai/HiDream-I1-Full",
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=DTYPE,
        transformer=transformer,
    )
else:
    pipe = HiDreamImageEditingPipeline.from_pretrained(
        "HiDream-ai/HiDream-E1-Full",
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=DTYPE,
    )

pipe = pipe.to(DEVICE, DTYPE)  # 使用设备管理器的设备和数据类型

logging.info("=" * 60)
logging.info("✓ Models loaded successfully!")
logging.info(f"  Device: {DEVICE}")
logging.info(f"  Data type: {DTYPE}")
device_manager.memory_stats()
logging.info("=" * 60)

# Load and preprocess test image
test_image = Image.open("assets/test_1.png")
original_width, original_height = test_image.size
test_image = test_image.resize((768, 768))

# Define instruction
instruction = 'Convert the image into a Ghibli style.'

# Refine instruction if enabled
if ENABLE_REFINE:
    logging.info("Refining instruction...")
    try:
        refined_instruction = refine_instruction(src_image=test_image, src_instruction=instruction)
        logging.info(f"Original instruction: {instruction}")
        logging.info(f"Refined instruction: {refined_instruction}")
    except Exception as e:
        logging.warning(f"Instruction refinement failed: {e}")
        logging.warning("Using formatted instruction without refinement")
        refined_instruction = f"Editing Instruction: {instruction}. Target Image Description: The image converted to Ghibli style."
else:
    refined_instruction = instruction

# Generate image
logging.info("Starting image generation...")
import time
start_time = time.time()

# 使用设备管理器创建生成器
generator = device_manager.create_generator(3)

image = pipe(
    prompt=refined_instruction,
    negative_prompt="low resolution, blur",
    image=test_image,
    guidance_scale=5.0,
    image_guidance_scale=4.0,
    num_inference_steps=28,
    generator=generator,  # 使用设备管理器创建的生成器
    refine_strength=0.3 if ENABLE_REFINE else 0.0,
    reload_keys=reload_keys,
).images[0]

end_time = time.time()
inference_time = end_time - start_time

# Resize back to original dimensions and save
image = image.resize((original_width, original_height))
output_path = "output_npu.jpg"
image.save(output_path)

logging.info("=" * 60)
logging.info(f"✓ Image saved to {output_path}")
logging.info(f"  Inference time: {inference_time:.2f} seconds")
device_manager.memory_stats()
logging.info("=" * 60)
