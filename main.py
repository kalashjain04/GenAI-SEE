# ui.py
import gradio as gr
from main import load_model, infer

# Load the model
pipe = load_model()

# Gradio app function
def generate_image(prompt, steps, scale, seed):
    return infer(pipe, prompt, num_steps=steps, guidance_scale=scale, seed=seed)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Stable Diffusion XL Demo")

    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="Enter image description")
        seed = gr.Number(label="Seed", value=12345)

    with gr.Row():
        steps = gr.Slider(label="Steps", minimum=10, maximum=100, step=10, value=50)
        scale = gr.Slider(label="Guidance Scale", minimum=5.0, maximum=15.0, step=0.5, value=7.5)

    output_image = gr.Image(label="Generated Image", type="pil")
    generate_btn = gr.Button("Generate Image")

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, steps, scale, seed],
        outputs=[output_image]
    )

if __name__ == "__main__":
    demo.launch()
