import torch
import gradio as gr
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from pipeline_hidream_image_editing import HiDreamImageEditingPipeline
from PIL import Image
from diffusers import HiDreamImageTransformer2DModel
import json, os
from collections import defaultdict
from safetensors.torch import safe_open
import uuid
import math
import logging
from diffusers.utils import logging as diffusers_logging

# ============ NPU 支持 ============
from device_utils import DeviceManager
# ==================================

# Set diffusers logging to INFO level
diffusers_logging.set_verbosity_info()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger.info("Logging configuration completed")

# ============ 初始化设备管理器 ============
preferred_device = os.environ.get('PREFERRED_DEVICE', None)
device_manager = DeviceManager(preferred_device=preferred_device)
DEVICE = device_manager.device
DTYPE = device_manager.dtype
# =========================================

# Flash Attention 处理
os.environ.setdefault("DIFFUSERS_ATTENTION_TYPE", "vanilla")
logger.info(f"Attention type: {os.environ.get('DIFFUSERS_ATTENTION_TYPE', 'default')}")

# Paths and globals
LLAMA_PATH = "meta-llama/Llama-3.1-8B-Instruct"
HIDREAM_I1_PATH = "HiDream-ai/HiDream-I1-Full"
HIDREAM_E1_PATH = "HiDream-ai/HiDream-E1-1"
pipe = None
reload_keys = None

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
    with open(f"{directory}/diffusion_pytorch_model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    shards = defaultdict(list)
    for name, file in weight_map.items():
        shards[file].append(name)

    state_dict = {}
    for file, names in shards.items():
        with safe_open(f"{directory}/{file}", framework="pt", device="cpu") as f:
            state_dict.update({name: f.get_tensor(name) for name in names})
    return state_dict

def init_models():
    global pipe, reload_keys
    logger.info("=" * 60)
    logger.info("Loading models...")
    logger.info("=" * 60)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(LLAMA_PATH)
    text_encoder = LlamaForCausalLM.from_pretrained(
        LLAMA_PATH,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=DTYPE  # 使用自动检测的数据类型
    )
    logger.info("✓ Llama model loaded")

    transformer = HiDreamImageTransformer2DModel.from_pretrained(HIDREAM_I1_PATH, subfolder="transformer")
    transformer.max_seq = 8192
    logger.info("✓ Transformer loaded")

    src_dict = transformer.state_dict()
    edit_dict = load_safetensors(HIDREAM_E1_PATH + "/transformer")
    reload_keys = {"editing": src_dict, "refine": edit_dict}
    transformer.load_state_dict(edit_dict, strict=True)
    logger.info("✓ Editing weights loaded")

    pipe = HiDreamImageEditingPipeline.from_pretrained(
        HIDREAM_I1_PATH,
        tokenizer_4=tokenizer,
        text_encoder_4=text_encoder,
        torch_dtype=DTYPE,
        transformer=transformer
    ).to(DEVICE, DTYPE)  # 使用设备管理器的设备和数据类型

    logger.info("=" * 60)
    logger.info("✓ Models loaded successfully!")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Data type: {DTYPE}")
    device_manager.memory_stats()
    logger.info("=" * 60)

def edit_image_gradio(image, instruction, negative_instruction, guidance_scale,
                      img_guidance_scale, steps, refine_strength, clip_cfg_norm, seed):
    try:
        if image is None:
            return None, "❌ Please upload an image first!"

        if not instruction.strip():
            return None, "❌ Please provide an editing instruction!"

        logger.info(f"Processing image with instruction: {instruction}")

        # Ensure image is PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        original_size = image.size
        processed_img = resize_image(image)

        logger.info(f"Original size: {original_size}, Processed size: {processed_img.size}")

        # 使用设备管理器创建生成器
        generator = device_manager.create_generator(int(seed))

        # Generate
        import time
        start_time = time.time()

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

        end_time = time.time()
        inference_time = end_time - start_time

        output_image = result.images[0].resize(original_size)

        # Save to outputs directory
        os.makedirs("outputs", exist_ok=True)
        output_filename = f"outputs/edited_{uuid.uuid4().hex[:8]}.jpg"
        output_image.save(output_filename)

        # Memory stats
        device_manager.memory_stats()

        status_msg = f"✅ Success! Time: {inference_time:.2f}s | Device: {DEVICE} | Saved to: {output_filename}"
        logger.info(status_msg)

        return output_image, status_msg

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg

# Initialize models on startup
init_models()

# Create Gradio interface
with gr.Blocks(title="HiDream-E1.1 NPU Edition") as demo:
    gr.Markdown("""
    # 🎨 HiDream-E1.1 Image Editing Demo (NPU Compatible)

    Upload an image and provide editing instructions. This version supports NPU/CUDA/CPU auto-detection.

    **Current Device**: `{}`
    **Data Type**: `{}`
    """.format(DEVICE, DTYPE))

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="pil")
            instruction = gr.Textbox(
                label="Editing Instruction",
                placeholder="e.g., Convert the image into a Ghibli style",
                value="Convert the image into a Ghibli style"
            )
            negative_instruction = gr.Textbox(
                label="Negative Prompt",
                value="low quality, blurry, distorted"
            )

            with gr.Accordion("Advanced Settings", open=False):
                guidance_scale = gr.Slider(0.0, 10.0, value=3.0, step=0.1, label="Text Guidance Scale")
                img_guidance_scale = gr.Slider(0.0, 10.0, value=1.5, step=0.1, label="Image Guidance Scale")
                steps = gr.Slider(10, 50, value=28, step=1, label="Inference Steps")
                refine_strength = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Refine Strength")
                clip_cfg_norm = gr.Checkbox(value=True, label="CLIP CFG Normalization")
                seed = gr.Number(value=3, label="Random Seed", precision=0)

            submit_btn = gr.Button("🚀 Generate", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Edited Image")
            status_text = gr.Textbox(label="Status", interactive=False)

    gr.Examples(
        examples=[
            ["Convert the image into a Ghibli style"],
            ["Make it look like a watercolor painting"],
            ["Transform into a cyberpunk style"],
            ["Add a sunset background"],
        ],
        inputs=[instruction],
    )

    submit_btn.click(
        fn=edit_image_gradio,
        inputs=[
            input_image, instruction, negative_instruction,
            guidance_scale, img_guidance_scale, steps,
            refine_strength, clip_cfg_norm, seed
        ],
        outputs=[output_image, status_text]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
