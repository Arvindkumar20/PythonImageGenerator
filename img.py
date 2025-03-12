import os
import re
from diffusers import StableDiffusionPipeline
import torch

# Directory to store generated images
IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)

def sanitize_filename(prompt):
    """Sanitize prompt to create a valid filename."""
    return re.sub(r'[^a-zA-Z0-9-_ ]', '', prompt).replace(" ", "_") + ".png"

def generate_image(prompt):
    """
    Checks if an image for the given prompt already exists.
    If it exists, return the existing file path.
    Otherwise, generate a new image, save it, and return the file path.
    
    :param prompt: The text description for the image.
    :return: The saved image path.
    """
    try:
        filename = sanitize_filename(prompt)
        output_path = os.path.join(IMAGE_DIR, filename)

        # Check if the image already exists
        if os.path.exists(output_path):
            print(f"Image already exists: {output_path}")
            return output_path

        # Load Stable Diffusion model
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate image
        image = pipe(prompt).images[0]

        # Save image
        image.save(output_path)
        print(f"Image generated and saved at: {output_path}")
        return output_path

    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    prompt = input("Enter image prompt: ")
    image_path = generate_image(prompt)
    print(f"Image path: {image_path}")
