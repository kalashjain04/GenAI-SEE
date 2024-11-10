import os
import random
from os import path
import time
from sys import platform
import torch

cache_path = path.join(path.dirname(path.abspath(__file__)), "models")

os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path
is_mac = platform == "darwin"

def should_use_fp16():
    if is_mac:
        return True

    try:
        gpu_props = torch.cuda.get_device_properties("cuda")
        if gpu_props.major < 6:
            return False

        nvidia_16_series = ["1660", "1650", "1630"]
        for x in nvidia_16_series:
            if x in gpu_props.name:
                return False
    except Exception:
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

def load_models(model_id="Lykon/dreamshaper-7"):
    from diffusers import StableDiffusionImg2ImgPipeline, EulerDiscreteScheduler
    from diffusers.utils import load_image

    if not is_mac:
        torch.backends.cuda.matmul.allow_tf32 = True

    use_fp16 = should_use_fp16()

    # Model loading with advanced configuration options
    if use_fp16:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None
        )
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            safety_checker=None
        )

    # Advanced scheduler for better control over the generation
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    # Move model to GPU or MPS
    pipe.to("cuda" if torch.cuda.is_available() else "mps")

    # Seed generator for reproducibility
    generator = torch.Generator()

    def infer(
            prompt,
            image,
            num_inference_steps=50,
            guidance_scale=7.5,
            strength=0.75,
            seed=random.randrange(0, 2**63)
    ):
        with torch.inference_mode():
            with torch.autocast("cuda" if torch.cuda.is_available() else "mps"):
                with timer("inference"):
                    return pipe(
                        prompt=prompt,
                        image=load_image(image),
                        generator=generator.manual_seed(seed),
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=strength
                    ).images[0]

    return infer

