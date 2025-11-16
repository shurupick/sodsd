import os
from pathlib import Path
import random

from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.utils import load_image
import torch

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Load template from JSON above (you can paste directly)
generator = {
    "ground_types": [
        "rich dark black chernozem soil",
        "loamy soil with mixed clay and sand particles",
        "light sandy soil with fine grains",
        "dry steppe ground with cracked soil",
        "wet swampy soil with organic debris",
        "rocky ground with small stones",
        "forest soil with needles and leaves",
        "reddish-brown clay soil",
    ],
    "seasons": {
        "spring": "early spring surface with fresh sprouts and subtle humidity",
        "summer": "summer ground with warm sunlight and vivid natural colors",
        "autumn": "autumn ground with orange and brown fallen leaves",
        "winter": "frozen ground with thin frost crystals and cold tones",
    },
    "surface_details": [
        "detailed organic texture",
        "slightly uneven soil structure",
        "fine-grain natural texture",
        "moist reflective details",
        "dry cracked surface patterns",
        "scattered organic debris",
    ],
    "negative_prompts": [
        "ugly, distorted, low-quality, blurry, artifacts",
        "oversaturated, unnatural colors",
        "synthetic look, CGI, cartoon, illustration",
        "low contrast, washed out, overexposed",
        "repeating patterns, tiling artifacts",
        "people, animals, buildings, objects",
        "trash, footprints, tools, unnatural objects",
        "shadows or reflections of humans or equipment",
    ],
}


def generate_prompt():
    gt = random.choice(generator["ground_types"])
    season_key = random.choice(list(generator["seasons"].keys()))
    season = generator["seasons"][season_key]
    detail = random.choice(generator["surface_details"])
    negs = ", ".join(random.sample(generator["negative_prompts"], 3))

    prompt = (
        f"Top-down close-up view of {gt}, {season}, {detail}, "
        f"overhead perspective, natural diffuse daylight, high detail, photorealistic texture."
    )

    return prompt, negs


for i in range(10):
    p, n = generate_prompt()
    print("PROMPT:", p)
    print("NEG:", n)
    print("---")
    image = pipe(prompt=p, negative_prompt=n).images[0]
    save_path = Path("/home/ubuntu/sodsd/data/raw")
    image_name = Path(f"imagedopNew_{i}.png")
    image.save(os.path.join(save_path, image_name))