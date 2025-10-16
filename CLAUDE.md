# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

HiDream-E1 is an instruction-based image editing model built on HiDream-I1. The repository contains two model versions:
- **HiDream-E1-Full**: Original version requiring formatted instructions (768x768 resolution)
- **HiDream-E1.1**: Updated version supporting dynamic resolution (up to 1M pixels) and direct instructions

## Setup and Installation

### Prerequisites
- CUDA 12.4 recommended
- HuggingFace account with access to Llama-3.1-8B-Instruct model
- Login required: `huggingface-cli login`

### Installation
```bash
pip install -r requirements.txt
pip install -U flash-attn --no-build-isolation
pip install -U git+https://github.com/huggingface/diffusers.git
```

## Running Inference

### HiDream-E1.1 (Recommended)
```bash
python inference_e1_1.py
```
- Supports dynamic resolution (up to 1M pixels)
- Accepts direct instructions: "convert the image into a Ghibli style"
- No instruction refinement needed

### HiDream-E1-Full (Legacy)
```bash
python inference.py
```
- Fixed 768x768 resolution
- Requires formatted instructions:
  ```
  Editing Instruction: {instruction}. Target Image Description: {description}
  ```

### Instruction Refinement (E1-Full only)
```bash
python instruction_refinement.py --src_image ./test.jpeg --src_instruction "convert the image into a Ghibli style"
```
Requires VLM API (OpenAI or local vllm) - set `OPENAI_API_KEY` environment variable.

### Gradio Demo
```bash
# For HiDream-E1.1
python gradio_demo_1_1.py

# For HiDream-E1-Full
python gradio_demo.py
```

## Architecture

### Core Components

1. **Pipeline**: `pipeline_hidream_image_editing.py`
   - Custom `HiDreamImageEditingPipeline` extending Diffusers `DiffusionPipeline`
   - Multi-encoder architecture:
     - 4 text encoders: CLIP (2x), T5, Llama-3.1-8B
     - VAE for latent encoding/decoding
     - HiDreamImageTransformer2DModel as the core diffusion model
   - Supports two-stage processing: editing + refinement

2. **Model Loading Patterns**:
   - E1-Full: Uses LoRA adapters on top of HiDream-I1-Full transformer
   - E1.1: Loads sharded safetensors directly with custom loader
   - Both versions use `reload_keys` mechanism to switch between editing and refinement stages

3. **Image Processing**:
   - E1-Full: Resizes to fixed 768x768
   - E1.1: Dynamic resize maintaining aspect ratio with 1M pixel cap (16px grid alignment)
   - Post-generation resize back to original dimensions

### Key Parameters

- `refine_strength` (0.0-1.0): Controls balance between editing and refinement stages
  - During first (1 - refine_strength) steps: performs main editing
  - Remaining steps: uses HiDream-I1-Full for img2img refinement
  - Set to 0.0 to disable refinement
- `guidance_scale`: Text conditioning strength (typical: 3.0-5.0)
- `image_guidance_scale`: Image conditioning strength (typical: 1.5-4.0)
- `clip_cfg_norm`: CLIP CFG normalization (E1.1 only)
- `num_inference_steps`: Default 28 steps

### Instruction Refinement System (E1-Full)

The `instruction_refinement.py` module uses GPT-4o to:
1. Analyze source image
2. Refine editing instruction for specificity (handles ambiguity in multi-object scenes)
3. Generate target image description
4. Format as: "Editing Instruction: {X}. Target Image Description: {Y}"

## Model Files

- HuggingFace repos:
  - `HiDream-ai/HiDream-E1-Full`
  - `HiDream-ai/HiDream-E1-1`
  - `HiDream-ai/HiDream-I1-Full` (base model for refinement)
  - `meta-llama/Llama-3.1-8B-Instruct` (text encoder)

## Development Notes

- All inference scripts use `torch.bfloat16` precision on CUDA
- Models are loaded to GPU by default
- The pipeline modifies transformer `max_seq` parameter (4608 for E1-Full, 8192 for E1.1)
- E1-Full uses PEFT LoRA with r=16, targeting attention and MLP layers
- E1.1 uses sharded checkpoint loading with JSON index mapping
