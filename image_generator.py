from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from io import BytesIO

def generate_image(prompt: str):
    """
    Generates an image based on the given prompt using Stable Diffusion.
    
    Args:
        prompt (str): The text prompt to generate the image.
    
    Returns:
        Image object: Generated image as a PIL Image.
    """
    # Load the Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Generate image
    image = pipe(prompt).images[0]
    
    return image

if __name__ == "__main__":
    prompt_text = input("Enter your prompt: ")
    img = generate_image(prompt_text)
    img.show()  # Display the image


# from diffusers import StableDiffusionPipeline
# import torch

# def generate_image(prompt: str):
#     """
#     Generates an image based on the given prompt using an optimized Stable Diffusion pipeline.
    
#     Args:
#         prompt (str): The text prompt to generate the image.
    
#     Returns:
#         PIL.Image.Image: The generated image.
#     """
#     # Load the model with optimizations
#     pipe = StableDiffusionPipeline.from_pretrained(
#         "stabilityai/stable-diffusion-2-1",
#         torch_dtype=torch.float16  # Use half-precision for faster inference
#     )
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     pipe.to(device)

#     # Enable optimizations
#     pipe.enable_attention_slicing()  # Reduces memory usage
#     pipe.enable_xformers_memory_efficient_attention()  # Uses faster memory-efficient attention
#     pipe.vae.enable_tiling()  # Further optimizations for low-memory GPUs

#     # Generate image with reduced steps and optimized size
#     image = pipe(prompt, num_inference_steps=30, height=512, width=512).images[0]
    
#     return image

# if __name__ == "__main__":
#     prompt_text = input("Enter your prompt: ")
#     img = generate_image(prompt_text)
#     img.show()  # Display the image
