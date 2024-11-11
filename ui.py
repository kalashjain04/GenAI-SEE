import os
import argparse
import gradio as gr
from main import load_models, infer, cache_path
from PIL import Image
from os import path

canvas_size = 512
if not path.exists(cache_path):
    os.makedirs(cache_path, exist_ok=True)

with gr.Blocks() as demo:
    pipe, device = load_models()
    
    with gr.Column():
        with gr.Row():
            steps_slider = gr.Slider(label="Inference Steps", minimum=4, maximum=10, step=1, value=4, interactive=True)
            scale_slider = gr.Slider(label="Guidance Scale", minimum=0.5, maximum=3, step=0.1, value=1, interactive=True)
            strength_slider = gr.Slider(label="Strength", minimum=0.1, maximum=1, step=0.1, value=0.8, interactive=True)
        
        with gr.Row():
            model_id_input = gr.Text(label="Model ID", value="stabilityai/stable-diffusion-2-1", interactive=True)
            prompt_input = gr.Text(label="Prompt", value="Describe your scene...", interactive=True)
            storyline_input = gr.Text(label="Storyline (Optional)", value="", interactive=True)
            genre_input = gr.Text(label="Genre (Optional)", value="", interactive=True)
            seed_input = gr.Number(label="Seed", value=random.randint(0, 9999), interactive=True)

        with gr.Row(equal_height=True):
            input_image = gr.Image(source="canvas", tool="color-sketch", shape=(canvas_size, canvas_size), width=canvas_size, height=canvas_size, type="pil")
            output_image = gr.Image(width=canvas_size, height=canvas_size)

            def process_image(prompt, image, steps, scale, strength, seed, storyline, genre):
                if not image:
                    return Image.new("RGB", (canvas_size, canvas_size))
                return infer(pipe, device, prompt, image, steps, scale, strength, seed, storyline)

            # Connect UI controls to inference function
            reactive_controls = [prompt_input, input_image, steps_slider, scale_slider, strength_slider, seed_input, storyline_input, genre_input]

            # Automatically update output when any control changes
            for control in reactive_controls:
                control.change(fn=process_image, inputs=reactive_controls, outputs=output_image)

            # Update model when the model ID is changed
            def update_model(model_name):
                global pipe, device
                pipe, device = load_models(model_name)
                logging.info(f"Updated model to {model_name}")

            model_id_input.change(fn=update_model, inputs=model_id_input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Deploy on Gradio for sharing", default=False)
    args = parser.parse_args()
    demo.launch(share=args.share)
