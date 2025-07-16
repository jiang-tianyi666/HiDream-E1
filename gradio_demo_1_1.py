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
# Configure logging to show INFO messages
import logging
from diffusers.utils import logging as diffusers_logging
import math

# Set diffusers logging to INFO level to see the logger.info messages
diffusers_logging.set_verbosity_info()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Ensure output goes to console
    ]
)
logger.info("Logging configuration completed")

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
    logger.info("Loading models...")
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(LLAMA_PATH)
    text_encoder = LlamaForCausalLM.from_pretrained(
        LLAMA_PATH, output_hidden_states=True, output_attentions=True, torch_dtype=torch.bfloat16
    )
    
    transformer = HiDreamImageTransformer2DModel.from_pretrained(HIDREAM_I1_PATH, subfolder="transformer")
    transformer.max_seq = 8192

    src_dict = transformer.state_dict()
    edit_dict = load_safetensors(HIDREAM_E1_PATH + "/transformer")
    reload_keys = {"editing": src_dict, "refine": edit_dict}
    transformer.load_state_dict(edit_dict, strict=True)

    pipe = HiDreamImageEditingPipeline.from_pretrained(
        HIDREAM_I1_PATH, tokenizer_4=tokenizer, text_encoder_4=text_encoder,
        torch_dtype=torch.bfloat16, transformer=transformer
    ).to("cuda", torch.bfloat16)
    logger.info("Models loaded!")
    logger.info(f"Current CUDA memory: {torch.cuda.memory_summary(device='cuda', abbreviated=True)}")

def edit_image(img, instruction, negative_instruction, guidance_scale, img_guidance_scale, steps, refine_strength, clip_cfg_norm, seed):
    if not img: return None, "Upload an image first."
    if not instruction.strip(): return None, "Provide an instruction."
    
    try:
        original_size = img.size
        processed_img = resize_image(img)
        logger.info(f"Original size: {original_size}")
        logger.info(f"Processed size: {processed_img.size}")
        
        # Use provided negative instruction or default
        result = pipe(
            prompt=instruction, negative_prompt=negative_instruction.strip(), image=processed_img,
            guidance_scale=guidance_scale, image_guidance_scale=img_guidance_scale,
            num_inference_steps=int(steps), generator=torch.Generator("cuda").manual_seed(int(seed)),
            refine_strength=refine_strength, reload_keys=reload_keys, clip_cfg_norm=clip_cfg_norm
        )
        save_key = str(uuid.uuid4())
        src_image_path = f"results/{save_key}_src.jpg"
        tgt_image_path = f"results/{save_key}_tgt.jpg"
        meta_path = f"results/{save_key}.json"
        os.makedirs("results", exist_ok=True)
        processed_img.save(src_image_path)
        result.images[0].resize(original_size).save(tgt_image_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "instruction": instruction,
                "negative_instruction": negative_instruction,
                "guidance_scale": guidance_scale,
                "img_guidance_scale": img_guidance_scale,
                "steps": steps,
                "refine_strength": refine_strength,
                "clip_cfg_norm": clip_cfg_norm,
                "seed": seed,
                "src_image_path": src_image_path,
                "tgt_image_path": tgt_image_path,
                "meta_path": meta_path
            }, f, indent=4)
        logger.info(f"Saved results to {src_image_path}, {tgt_image_path}, {meta_path}")
        return result.images[0].resize(original_size), f"Success: '{instruction}'"
    except Exception as e:
        return None, f"Error: {e}"

init_models()

with gr.Blocks(title="HiDream Image Editor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 HiDream Image Editor\nTransform images with AI-powered editing instructions.")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="Upload Image", type="pil", height=400)
            instruction = gr.Textbox(label="Instruction", placeholder="e.g., Convert to Ghibli style", lines=2)
            negative_instruction = gr.Textbox(label="Negative Instruction", placeholder="e.g., low quality, blurry, distorted", lines=1, value="")
            
            with gr.Accordion("Advanced", open=False):
                guidance = gr.Slider(1, 20, 3, 0.1, label="Guidance Scale")
                img_guidance = gr.Slider(1, 10, 1.5, 0.1, label="Image Guidance")
                steps = gr.Slider(10, 50, 28, 1, label="Steps")
                refine = gr.Slider(0, 1, 0.3, 0.05, label="Refine Strength")
                clip_norm = gr.Checkbox(label="CLIP CFG Norm", value=True)
                seed = gr.Number(label="Seed", value=3, precision=0)
            
            btn = gr.Button("🎨 Edit", variant="primary", size="lg")
        
        with gr.Column():
            img_output = gr.Image(label="Result", height=400)
            status = gr.Textbox(label="Status", interactive=False, lines=2)
    
    btn.click(edit_image, [img_input, instruction, negative_instruction, guidance, img_guidance, steps, refine, clip_norm, seed], [img_output, status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False) 