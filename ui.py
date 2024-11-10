import os
import gradio as gr
from PIL import Image
from main import load_models

canvas_size = 512
cache_path = "/content/models"

# Ensure cache directory exists
if not os.path.exists(cache_path):
    os.makedirs(cache_path, exist_ok=True)

infer = load_models()

def process_image(p, im, steps, cfg, image_strength, seed):
    if not im:
        return Image.new("RGB", (canvas_size, canvas_size))
    return infer(
        prompt=p,
        image=im,
        num_inference_steps=steps,
        guidance_scale=cfg,
        strength=image_strength,
        seed=int(seed)
    )

# Setting up Gradio Interface
with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            with gr.Column():
                s = gr.Slider(label="Steps", minimum=4, maximum=8, step=1, value=4)
                c = gr.Slider(label="CFG", minimum=0.1, maximum=3, step=0.1, value=1)
                i_s = gr.Slider(label="Sketch Strength", minimum=0.1, maximum=0.9, step=0.1, value=0.9)
            with gr.Column():
                mod = gr.Text(label="Hugging Face Model ID", value="Lykon/dreamshaper-7")
                t = gr.Text(label="Prompt", value="")
                se = gr.Number(label="Seed", value=1337)

        with gr.Row(equal_height=True):
            i = gr.Image(source="canvas", tool="color-sketch", shape=(canvas_size, canvas_size), width=canvas_size, height=canvas_size, type="pil")
            o = gr.Image(width=canvas_size, height=canvas_size)

            def update_model(model_name):
                global infer
                infer = load_models(model_name)

            mod.change(fn=update_model, inputs=mod)

            def submit_fn(p, im, steps, cfg, image_strength, seed):
                return process_image(p, im, steps, cfg, image_strength, seed)

            # Make sure all controls update the output image
            reactive_controls = [t, i, s, c, i_s, se]
            for control in reactive_controls:
                control.change(fn=submit_fn, inputs=reactive_controls, outputs=o)

if __name__ == "__main__":
    demo.launch(share=True)
