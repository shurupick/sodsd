import argparse
import json
from pathlib import Path
import random

from diffusers import StableDiffusionXLPipeline
import torch

DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

BACKGROUND_PROMPT_PARTS = {
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


def generate_prompt(rng: random.Random):
    gt = rng.choice(BACKGROUND_PROMPT_PARTS["ground_types"])
    season_key = rng.choice(list(BACKGROUND_PROMPT_PARTS["seasons"].keys()))
    season = BACKGROUND_PROMPT_PARTS["seasons"][season_key]
    detail = rng.choice(BACKGROUND_PROMPT_PARTS["surface_details"])
    negs = ", ".join(rng.sample(BACKGROUND_PROMPT_PARTS["negative_prompts"], 3))

    prompt = (
        f"Top-down close-up view of {gt}, {season}, {detail}, "
        "overhead perspective, natural diffuse daylight, high detail, photorealistic texture."
    )

    return prompt, negs, season_key


def resolve_device(device: str):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype_for_device(device: str):
    return torch.float16 if device == "cuda" else torch.float32


def build_pipeline(model_id: str, device: str):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype_for_device(device),
    )
    return pipe.to(device)


def torch_generator_for_seed(seed: int | None, device: str):
    if seed is None:
        return None

    generator_device = device if device in {"cpu", "cuda"} else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def generate_backgrounds(
    count: int,
    outdir: Path,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "auto",
    seed: int | None = None,
    prefix: str = "background",
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 30,
    save_prompts: bool = True,
):
    device = resolve_device(device)
    rng = random.Random(seed)
    pipe = build_pipeline(model_id, device)
    outdir.mkdir(parents=True, exist_ok=True)

    for index in range(count):
        prompt, negative_prompt, season = generate_prompt(rng)
        image_seed = None if seed is None else seed + index
        generator = torch_generator_for_seed(image_seed, device)

        print("PROMPT:", prompt)
        print("NEG:", negative_prompt)
        print("---")

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]

        image_path = outdir / f"{prefix}_{index:04d}.png"
        image.save(image_path)

        if save_prompts:
            metadata_path = image_path.with_suffix(".json")
            metadata_path.write_text(
                json.dumps(
                    {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "season": season,
                        "seed": image_seed,
                        "model_id": model_id,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        print(f"saved: {image_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate raw natural ground backgrounds.")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--outdir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prefix", default="background")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--no-save-prompts", action="store_true")
    args = parser.parse_args()

    generate_backgrounds(
        count=args.count,
        outdir=args.outdir,
        model_id=args.model_id,
        device=args.device,
        seed=args.seed,
        prefix=args.prefix,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        save_prompts=not args.no_save_prompts,
    )


if __name__ == "__main__":
    main()
