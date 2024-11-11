import os
import random
from os import path
from contextlib import nullcontext
import time
from sys import platform
import torch
import logging
from diffusers import AutoPipelineForImage2Image, StableDiffusionPipeline

cache_path = path.join(path.dirname(path.abspath(__file__)), "models")

# Setup cache directories and logging
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path
logging.basicConfig(level=logging.INFO)
is_mac = platform == "darwin"

def should_use_fp16():
    if is_mac:
        return True
    gpu_props = torch.cuda.get_device_properties("cuda")
    if gpu_props.major < 6:
        return False
    nvidia_16_series = ["1660", "1650", "1630"]
    for x in nvidia_16_series:
        if x in gpu_props.name:
            return False
    return True

class timer:
    def __init__(self, method_name="timed process"):
        self.method = method_name

    def __enter__(self):
        self.start = time.time()
        print(f"{self.method} starts")

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        print(f"{self.method} took {str(round(end - self.start, 2))}s")

def load_models(model_id="stabilityai/stable-diffusion-2-1", enable_rl=False):
    # Load model pipeline with dynamic model selection
    if not is_mac:
        torch.backends.cuda.matmul.allow_tf32 = True
    use_fp16 = should_use_fp16()

    if enable_rl:
        model_id = "model_with_reinforcement"  # Placeholder for RL-enhanced model

    if use_fp16:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            cache_dir=cache_path,
            torch_dtype=torch.float16,
            safety_checker=None
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            cache_dir=cache_path,
            safety_checker=None
        )

    device = "mps" if is_mac else "cuda"
    pipe.to(device=device)
    return pipe, device

def infer(pipe, device, prompt, image, steps=4, scale=1, strength=0.9, seed=None, storyline=None):
    generator = torch.Generator()
    seed = random.randint(0, 2**63) if seed is None else seed
    generator.manual_seed(seed)

    with torch.inference_mode():
        with torch.autocast(device) if device == "cuda" else nullcontext():
            with timer("inference"):
                result = pipe(
                    prompt=prompt,
                    image=image,
                    generator=generator,
                    num_inference_steps=steps,
                    guidance_scale=scale,
                    strength=strength
                )
                logging.info(f"Storyline: {storyline}")
                return result.images[0]

def main():
    pass  # Placeholder for main execution
