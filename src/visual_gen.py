import os
import textwrap
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

class UniversalVisualEngine:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None

    def load_sdxl_lightning(self):
        if self.pipe is not None:
            return
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_4step_unet.safetensors"

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(self.device, dtype)
        unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=self.device))
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base, unet=unet, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None
        ).to(self.device)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )

    def render_title_or_diagram_card(self, scene: dict, preset: dict, out_path: str):
        style = preset["channel_style"]
        if style.startswith("mr_nightmare"):
            bg_color, text_color, accent_color = (8, 8, 10), (225, 225, 225), (180, 30, 30)
        elif style.startswith("history_in_minutes"):
            bg_color, text_color, accent_color = (28, 25, 23), (231, 229, 228), (217, 119, 6)
        else:
            bg_color, text_color, accent_color = (255, 255, 255), (20, 20, 20), (210, 45, 45)

        img = Image.new("RGB", (1920, 1080), color=bg_color)
        draw = ImageDraw.Draw(img)

        try:
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            main_font = ImageFont.truetype(font_path, 76)
            sub_font = ImageFont.truetype(font_path, 52)
        except Exception:
            main_font = ImageFont.load_default()
            sub_font = ImageFont.load_default()

        card_text = (scene.get("card_text") or scene["narration_text"][:55]).upper()

        if scene.get("scene_type") == "split_comparison" and "|" in card_text:
            left_txt, right_txt = [t.strip() for t in card_text.split("|", 1)]
            for y in range(120, 960, 40):
                draw.line([(960, y), (960, y + 20)], fill=accent_color, width=6)
            left_wrapped = "\n".join(textwrap.wrap(left_txt, width=18))
            right_wrapped = "\n".join(textwrap.wrap(right_txt, width=18))
            draw.multiline_text((480, 540), left_wrapped, fill=text_color, font=sub_font, anchor="mm", align="center")
            draw.multiline_text((1440, 540), right_wrapped, fill=accent_color, font=sub_font, anchor="mm", align="center")
        else:
            wrapped = "\n".join(textwrap.wrap(card_text, width=26))
            draw.multiline_text((960, 540), wrapped, fill=text_color, font=main_font, anchor="mm", align="center")

        img.save(out_path)

    def generate_scene_image(self, scene: dict, preset: dict, out_path: str) -> str:
        if os.path.exists(out_path):
            return out_path

        s_type = scene.get("scene_type", "character_scene")
        if s_type in ["title_card", "split_comparison"]:
            self.render_title_or_diagram_card(scene, preset, out_path)
            return out_path

        self.load_sdxl_lightning()
        style_prefix = preset["visuals"]["style_prefix"]
        char_token = preset["visuals"].get("character_token", "")
        full_prompt = f"{style_prefix} {char_token}, {scene['image_prompt']}"

        image = self.pipe(
            prompt=full_prompt,
            negative_prompt=preset["visuals"]["negative_prompt"],
            num_inference_steps=4,
            guidance_scale=0.0,
            width=1344,
            height=768,
        ).images[0]

        image = image.resize((1920, 1080), Image.Resampling.LANCZOS)
        image.save(out_path)
        return out_path
