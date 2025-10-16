import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from pipeline_hidream_image_editing import HiDreamImageEditingPipeline
from PIL import Image, ImageOps
from diffusers import HiDreamImageTransformer2DModel
import json
import os
from collections import defaultdict
from safetensors.torch import safe_open
import math
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
# 可以通过环境变量 PREFERRED_DEVICE 指定设备: 'npu', 'cuda', 'cpu'
# 或者直接修改下面的参数
preferred_device = os.environ.get('PREFERRED_DEVICE', None)  # None 表示自动检测
device_manager = DeviceManager(preferred_device=preferred_device)
DEVICE = device_manager.device
DTYPE = device_manager.dtype
# =========================================

# Paths and configuration
LLAMA_PATH = "meta-llama/Llama-3.1-8B-Instruct"
HIDREAM_I1_PATH = "HiDream-ai/HiDream-I1-Full"
HIDREAM_E1_PATH = "HiDream-ai/HiDream-E1-1"

# Flash Attention 处理
# NPU 不支持 CUDA Flash Attention，这里禁用它
os.environ.setdefault("DIFFUSERS_ATTENTION_TYPE", "vanilla")
logging.info(f"Attention type: {os.environ.get('DIFFUSERS_ATTENTION_TYPE', 'default')}")

def resize_image(pil_image, image_size = 1024):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    m = 16
    width, height = pil_image.width, pil_image.height
    S_max = image_size * image_size
    scale = S_max / (width * height)
    scale = math.sqrt(scale)

    new_sizes = [
        (round(width * scale) // m * m, round(height * scale) // m * m),
        (round(width * scale) // m * m, math.floor(height * scale) // m * m),
        (math.floor(width * scale) // m * m, round(height * scale) // m * m),
        (math.floor(width * scale) // m * m, math.floor(height * scale) // m * m),
    ]
    new_sizes = sorted(new_sizes, key=lambda x: x[0] * x[1], reverse=True)

    for new_size in new_sizes:
        if new_size[0] * new_size[1] <= S_max:
            break

    s1 = width / new_size[0]
    s2 = height / new_size[1]
    if s1 < s2:
        pil_image = pil_image.resize([new_size[0], round(height / s1)], resample=Image.BICUBIC)
        top = (round(height / s1) - new_size[1]) // 2
        pil_image = pil_image.crop((0, top, new_size[0], top + new_size[1]))
    else:
        pil_image = pil_image.resize([round(width / s2), new_size[1]], resample=Image.BICUBIC)
        left = (round(width / s2) - new_size[0]) // 2
        pil_image = pil_image.crop((left, 0, left + new_size[0], new_size[1]))

    return pil_image

def load_safetensors(directory):
    """Load sharded safetensors from directory"""
    with open(f"{directory}/diffusion_pytorch_model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    shards = defaultdict(list)
    for name, file in weight_map.items():
        shards[file].append(name)

    state_dict = {}
    for file, names in shards.items():
        # 始终先加载到 CPU，然后再转移到目标设备（更兼容）
        with safe_open(f"{directory}/{file}", framework="pt", device="cpu") as f:
            state_dict.update({name: f.get_tensor(name) for name in names})
    return state_dict

def init_models():
    """Initialize and load all required models"""
    global pipe, reload_keys
    logging.info("=" * 60)
    logging.info("Loading models...")
    logging.info("=" * 60)

    # Load tokenizer and text encoder
    logging.info(f"Loading Llama tokenizer and text encoder from {LLAMA_PATH}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(LLAMA_PATH)
    text_encoder = LlamaForCausalLM.from_pretrained(
        LLAMA_PATH,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=DTYPE  # 使用自动检测的数据类型
    )
    logging.info("✓ Llama model loaded")

    # Load transformer
    logging.info(f"Loading transformer from {HIDREAM_I1_PATH}...")
    transformer = HiDreamImageTransformer2DModel.from_pretrained(HIDREAM_I1_PATH, subfolder="transformer")
    transformer.max_seq = 8192
    logging.info("✓ Transformer loaded")

    # Load editing weights
    logging.info(f"Loading editing weights from {HIDREAM_E1_PATH}...")
    src_dict = transformer.state_dict()
    edit_dict = load_safetensors(HIDREAM_E1_PATH + "/transformer")
    reload_keys = {"editing": src_dict, "refine": edit_dict}
    transformer.load_state_dict(edit_dict, strict=True)
    logging.info("✓ Editing weights loaded")

    # Create pipeline
    logging.info("Creating pipeline...")
    pipe = HiDreamImageEditingPipeline.from_pretrained(
        HIDREAM_I1_PATH,
        tokenizer_4=tokenizer,
        text_encoder_4=text_encoder,
        torch_dtype=DTYPE,  # 使用自动检测的数据类型
        transformer=transformer
    ).to(DEVICE, DTYPE)  # 使用设备管理器的设备和数据类型

    logging.info("=" * 60)
    logging.info("✓ Models loaded successfully!")
    logging.info(f"  Device: {DEVICE}")
    logging.info(f"  Data type: {DTYPE}")

    # 显示内存统计
    device_manager.memory_stats()
    logging.info("=" * 60)

    return pipe, reload_keys

def edit_image(image_path, instruction, negative_instruction="low quality, blurry, distorted",
               guidance_scale=3.0, img_guidance_scale=1.5, steps=28, refine_strength=0.3,
               clip_cfg_norm=True, seed=3):
    """
    Edit an image using the HiDream pipeline

    Args:
        image_path (str): Path to input image
        instruction (str): Editing instruction
        negative_instruction (str): Negative prompt
        guidance_scale (float): Guidance scale for text conditioning
        img_guidance_scale (float): Guidance scale for image conditioning
        steps (int): Number of inference steps
        refine_strength (float): Strength of refinement
        clip_cfg_norm (bool): Whether to use CLIP CFG normalization
        seed (int): Random seed
        output_path (str): Path to save output image

    Returns:
        PIL.Image: Edited image
        dict: Metadata
    """
    try:
        # Load and preprocess image
        img = Image.open(image_path)
        original_size = img.size
        processed_img = resize_image(img)

        logging.info(f"Original size: {original_size}")
        logging.info(f"Processed size: {processed_img.size}")
        logging.info(f"Instruction: {instruction}")

        # 使用设备管理器创建生成器
        generator = device_manager.create_generator(seed)

        # Generate edited image
        logging.info("Starting image generation...")
        result = pipe(
            prompt=instruction,
            negative_prompt=negative_instruction,
            image=processed_img,
            guidance_scale=guidance_scale,
            image_guidance_scale=img_guidance_scale,
            num_inference_steps=int(steps),
            generator=generator,  # 使用设备管理器创建的生成器
            refine_strength=refine_strength,
            reload_keys=reload_keys,
            clip_cfg_norm=clip_cfg_norm,
        )

        output_image = result.images[0].resize(original_size)
        metadata = {
            "instruction": instruction,
            "negative_instruction": negative_instruction,
            "guidance_scale": guidance_scale,
            "img_guidance_scale": img_guidance_scale,
            "steps": steps,
            "refine_strength": refine_strength,
            "clip_cfg_norm": clip_cfg_norm,
            "seed": seed,
            "input_image": image_path,
            "original_size": original_size,
            "processed_size": processed_img.size,
            "device": DEVICE,  # 记录使用的设备
            "dtype": str(DTYPE)  # 记录使用的数据类型
        }

        return output_image, metadata

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        raise e

def main():
    """Main function to run the inference"""
    # Initialize models
    init_models()

    # Configuration
    input_image_path = "assets/test_1.png"
    instruction = "Convert the image into a Ghibli style."
    output_path = "results/test_1_npu.jpg"

    # Check if input image exists
    if not os.path.exists(input_image_path):
        logging.error(f"Input image not found: {input_image_path}")
        return

    # Edit the image
    try:
        import time
        start_time = time.time()

        edited_image, metadata = edit_image(
            image_path=input_image_path,
            instruction=instruction,
            negative_instruction="low quality, blurry, distorted",
            guidance_scale=3.0,
            img_guidance_scale=1.5,
            steps=28,
            refine_strength=0.3,
            clip_cfg_norm=True,
            seed=3,
        )

        end_time = time.time()
        inference_time = end_time - start_time

        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        meta_path = f"{output_path.rsplit('.', 1)[0]}.json"
        edited_image.save(output_path)

        # Add performance info to metadata
        metadata["inference_time_seconds"] = inference_time

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        logging.info("=" * 60)
        logging.info("✓ Image editing completed successfully!")
        logging.info(f"  Output saved to: {output_path}")
        logging.info(f"  Metadata saved to: {meta_path}")
        logging.info(f"  Inference time: {inference_time:.2f} seconds")

        # Final memory stats
        device_manager.memory_stats()
        logging.info("=" * 60)

    except Exception as e:
        logging.error(f"Failed to edit image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
