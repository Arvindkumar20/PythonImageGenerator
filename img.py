import os
import re
import sys
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
            print(output_path)  # Return path to Express server
            return output_path

        # Load Stable Diffusion model
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate image
        image = pipe(prompt).images[0]

        # Save image
        image.save(output_path)
        print(output_path)  # Return path to Express server
        return output_path

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: No prompt provided", file=sys.stderr)
        sys.exit(1)

    prompt = sys.argv[1]
    generate_image(prompt)
