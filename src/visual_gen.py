import os
import torch
from diffusers import FluxPipeline

def generate_storyboard_visuals(storyboard, preset, images_dir):
    os.makedirs(images_dir, exist_ok=True)
    vis_cfg = preset["visuals"]
    model_id = vis_cfg.get("model_id", "black-forest-labs/FLUX.1-schnell")
    style_prefix = vis_cfg["style_prefix"].strip() + " "

    pending_scenes = []
    for ch in storyboard["chapters"]:
        for sc in ch["scenes"]:
            img_path = os.path.join(images_dir, f"{sc['scene_id']}.png")
            if not os.path.exists(img_path):
                pending_scenes.append((sc, img_path))

    if not pending_scenes:
        print("[*] All FLUX scene keyframes already exist in cache.")
        return

    print(f"[+] Loading {model_id} for {len(pending_scenes)} scenes...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)

    # Prevent OOM on Kaggle T4 GPUs
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    for idx, (sc, img_path) in enumerate(pending_scenes):
        full_prompt = style_prefix + sc["visual_prompt"]
        print(f"[+] Generating FLUX frame [{idx+1}/{len(pending_scenes)}]: {sc['scene_id']}")
        image = pipe(
            prompt=full_prompt,
            num_inference_steps=int(vis_cfg.get("steps", 4)),
            guidance_scale=float(vis_cfg.get("guidance_scale", 0.0)),
            width=int(vis_cfg.get("width", 1280)),
            height=int(vis_cfg.get("height", 720)),
            generator=torch.Generator("cpu").manual_seed(1000 + idx)
        ).images[0]
        image.save(img_path)

    del pipe
    torch.cuda.empty_cache()
