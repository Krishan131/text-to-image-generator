# streamlit_app.py

import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import login
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os


login("HF_TOKEN")  # Replace with your HF token

# Load the Stable Diffusion model
@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")  # Use "cuda" if you have a GPU
    return pipe

pipe = load_model()
st.title("🖼️ Text to Image Generator with Stable Diffusion")
prompt = st.text_input("Enter your prompt:", "")

if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image..."):
            image = pipe(prompt).images[0]
            st.image(image, caption=f"Prompt: {prompt}", use_container_width=True)
            
            # Add download button
            buf = BytesIO()
            image.save(buf, format="PNG")
            st.download_button(
                label="Download Image",
                data=buf.getvalue(),
                file_name="generated_image.png",
                mime="image/png"
            )
    else:
        st.warning("Please enter a prompt to generate an image.")


