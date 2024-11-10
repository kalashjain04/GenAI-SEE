# main.py
import os
import random
import torch
from diffusers import StableDiffusionPipeline
from contextlib import contextmanager
import time

cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Set cache environment
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path

# Timer context manager
@contextmanager
def timer(name):
    start = time.time()
    yield
    print(f"{name} took {round(time.time() - start, 2)}s")

# Load the model with FP16 support if possible
def load_model(model_id="stabilityai/stable-diffusion-xl-base-1.5", use_fp16=True):
    if use_fp16 and torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

# Run inference
def infer(pipe, prompt, num_steps=50, guidance_scale=7.5, seed=None):
    generator = torch.manual_seed(seed or random.randint(0, 2**32 - 1))
    with torch.inference_mode():
        with timer("Inference"):
            result = pipe(prompt, num_inference_steps=num_steps, guidance_scale=guidance_scale, generator=generator)
    return result.images[0]
