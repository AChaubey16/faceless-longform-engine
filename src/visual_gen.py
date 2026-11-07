import os
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import FluxPipeline

class UniversalVisualEngine:
    def __init__(self):
        self.pipe = None
        self.current_model_id = None
        self.scene_seed_counter = 1000

    def _load_pipeline(self, model_id: str):
        if self.pipe is not None and self.current_model_id == model_id:
            return
        print(f"[+] Loading FLUX model: {model_id} onto GPU with memory optimizations...")
        torch.cuda.empty_cache()
        
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        hf_token = os.environ.get("HF_TOKEN")
        
        self.pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=hf_token
        )
        self.pipe.enable_sequential_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.current_model_id = model_id

    def _get_font(self, size: int):
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                return ImageFont.truetype(fp, size)
        return ImageFont.load_default()

    def _render_title_card(self, card_text: str, preset: dict, out_path: str):
        render_cfg = preset.get("render", {})
        bg = tuple(render_cfg.get("bg_color", [244, 241, 234]))
        fg = tuple(render_cfg.get("text_color", [24, 20, 22]))
        accent = tuple(render_cfg.get("accent_color", [236, 110, 30]))

        img = Image.new("RGB", (1920, 1080), color=bg)
        draw = ImageDraw.Draw(img)
        font = self._get_font(76)

        text = (card_text or "CHAPTER").upper()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = (1920 - tw) // 2
        ty = (1080 - th) // 2 - 20

        draw.text((tx, ty), text, fill=fg, font=font)
        draw.rounded_rectangle(
            [tx - 20, ty + th + 28, tx + tw + 20, ty + th + 42],
            radius=6,
            fill=accent
        )
        img.save(out_path)

    def _render_split_comparison(self, card_text: str, preset: dict, out_path: str):
        render_cfg = preset.get("render", {})
        bg = tuple(render_cfg.get("bg_color", [244, 241, 234]))
        fg = tuple(render_cfg.get("text_color", [24, 20, 22]))
        accent = tuple(render_cfg.get("accent_color", [236, 110, 30]))

        parts = [p.strip() for p in (card_text or "THEN | NOW").split("|")]
        left_txt = parts[0].upper()
        right_txt = parts[1].upper() if len(parts) > 1 else "VS"

        img = Image.new("RGB", (1920, 1080), color=bg)
        draw = ImageDraw.Draw(img)
        font = self._get_font(64)

        draw.line([(960, 140), (960, 940)], fill=accent, width=10)

        lb = draw.textbbox((0, 0), left_txt, font=font)
        rb = draw.textbbox((0, 0), right_txt, font=font)
        draw.text(((960 - (lb[2] - lb[0])) // 2, 500), left_txt, fill=fg, font=font)
        draw.text((960 + (960 - (rb[2] - rb[0])) // 2, 500), right_txt, fill=fg, font=font)
        img.save(out_path)

    def generate_scene_image(self, scene: dict, preset: dict, img_path: str) -> str:
        if os.path.exists(img_path):
            return img_path

        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        vis_cfg = preset.get("visuals", {})
        scene_type = scene.get("scene_type", "character_scene")
        card_text = scene.get("card_text", "").strip()

        if scene_type == "title_card" and card_text and not preset.get("channel_style", "").startswith("mr_nightmare"):
            self._render_title_card(card_text, preset, img_path)
            return img_path
        if scene_type == "split_comparison" and card_text and not preset.get("channel_style", "").startswith("mr_nightmare"):
            self._render_split_comparison(card_text, preset, img_path)
            return img_path

        model_id = vis_cfg.get("model_id", "black-forest-labs/FLUX.1-schnell")
        style_prefix = vis_cfg.get("style_prefix", "").strip() + " "
        raw_prompt = scene.get("image_prompt") or scene.get("visual_prompt") or scene.get("narration_text", "")
        full_prompt = style_prefix + raw_prompt

        self._load_pipeline(model_id)
        self.scene_seed_counter += 1
        sid = scene.get("scene_id", self.scene_seed_counter)

        print(f"    [+] FLUX rendering scene_{sid:04d} ({scene_type})...")
        image = self.pipe(
            prompt=full_prompt,
            num_inference_steps=int(vis_cfg.get("steps", 4)),
            guidance_scale=float(vis_cfg.get("guidance_scale", 0.0)),
            width=1024,
            height=576,
            generator=torch.Generator("cpu").manual_seed(1000 + int(sid))
        ).images[0]

        image = image.resize((1280, 720), Image.Resampling.LANCZOS)
        image.save(img_path)
        torch.cuda.empty_cache()
        return img_path
