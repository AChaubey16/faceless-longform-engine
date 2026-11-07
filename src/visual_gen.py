import os
import torch
from diffusers import FluxPipeline

class UniversalVisualEngine:
    def __init__(self, preset=None, **kwargs):
        self.preset = preset or {}
        vis_cfg = self.preset.get("visuals", self.preset)
        self.model_id = vis_cfg.get("model_id", "black-forest-labs/FLUX.1-schnell")
        self.steps = int(vis_cfg.get("steps", 4))
        self.guidance_scale = float(vis_cfg.get("guidance_scale", 0.0))
        self.width = int(vis_cfg.get("width", 1280))
        self.height = int(vis_cfg.get("height", 720))
        self.style_prefix = vis_cfg.get(
            "style_prefix",
            "A clean 2D vector style illustration in the aesthetic of a modern educational explainer video. "
            "Simple cartoon character with a round gray head, wide expressive eyes, thick black marker outlines, "
            "flat color fills, minimalist whiteboard animation style, clean lines, high contrast."
        ).strip() + " "
        self.pipe = None

    def _load_pipeline(self):
        if self.pipe is not None:
            return
        print(f"[+] Loading {self.model_id} onto GPU...")
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        hf_token = os.environ.get("HF_TOKEN")
        self.pipe = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            token=hf_token
        )
        # Prevent OOM on Kaggle T4 GPUs
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

    def generate_image(self, visual_prompt, out_path, seed=1000):
        if os.path.exists(out_path):
            print(f"[*] Cached frame exists, skipping: {out_path}")
            return out_path

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self._load_pipeline()

        full_prompt = self.style_prefix + str(visual_prompt)
        image = self.pipe(
            prompt=full_prompt,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance_scale,
            width=self.width,
            height=self.height,
            generator=torch.Generator("cpu").manual_seed(seed)
        ).images[0]
        image.save(out_path)
        return out_path

    def generate_storyboard_visuals(self, storyboard, images_dir):
        os.makedirs(images_dir, exist_ok=True)
        scenes = []
        if "chapters" in storyboard:
            for ch in storyboard["chapters"]:
                scenes.extend(ch.get("scenes", []))
        elif "scenes" in storyboard:
            scenes = storyboard["scenes"]

        pending = []
        for idx, sc in enumerate(scenes):
            sid = sc.get("scene_id", f"scene_{idx:03d}")
            img_path = os.path.join(images_dir, f"{sid}.png")
            if not os.path.exists(img_path):
                pending.append((idx, sc, img_path))

        if not pending:
            print("[*] All FLUX scene keyframes already exist in cache.")
            return images_dir

        self._load_pipeline()
        for count, (idx, sc, img_path) in enumerate(pending, 1):
            prompt = sc.get("visual_prompt") or sc.get("image_prompt") or sc.get("narration", "")
            print(f"[+] Generating FLUX frame [{count}/{len(pending)}]: {os.path.basename(img_path)}")
            self.generate_image(prompt, img_path, seed=1000 + idx)

        self.unload()
        return images_dir

    # Aliases so any call signature in main.py works seamlessly
    generate_all = generate_storyboard_visuals
    generate_visuals = generate_storyboard_visuals
    generate = generate_storyboard_visuals
    render_visuals = generate_storyboard_visuals

    def unload(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()

def generate_storyboard_visuals(storyboard, preset, images_dir):
    engine = UniversalVisualEngine(preset)
    return engine.generate_storyboard_visuals(storyboard, images_dir)
EOF

!python /kaggle/working/faceless-longform-engine/main.py --preset presets/ink_explainer_12m.yaml --topic "What Did Ancient Humans Actually Do All Night?"
