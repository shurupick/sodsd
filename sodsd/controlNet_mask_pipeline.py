# !pip install transformers accelerate
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
import numpy as np
import torch
import cv2
from PIL import Image


init_image = Image.open("/home/ubuntu/sodsd/data/raw/astronaut_rides_horse.png")
init_image = init_image.resize((1024, 1024))

generator = generator = torch.Generator(device="cpu").manual_seed(1)

mask_image = Image.open("/home/ubuntu/sodsd/data/maskPfm/view_rx0.0_ry0.0_rz60.0.png")
mask_image = mask_image.resize((1024, 1024))


def make_canny_condition(image):
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image


control_image = make_canny_condition(init_image)

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
)
pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
)

pipe.enable_model_cpu_offload()

# generate image
image = pipe(
    "a handsome man with ray-ban sunglasses",
    num_inference_steps=20,
    generator=generator,
    eta=1.0,
    image=init_image,
    mask_image=mask_image,
    control_image=control_image
).images[0]

image.save("/home/ubuntu/sodsd/data/pfm/1.png")