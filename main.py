import argparse
import json
import os
import re
import yaml
import subprocess
from src.script_gen import generate_full_storyboard
from src.audio_tts import AudioEngine
from src.visual_gen import UniversalVisualEngine
from src.video_render import render_chapter_to_mp4

def slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')[:35]

def main():
    parser = argparse.ArgumentParser(description="Run Faceless Long-Form Video Engine (Groq Edition)")
    parser.add_argument("--topic", required=True, help="Video topic or title")
    parser.add_argument("--preset", required=True, help="Preset name: ink_explainer_12m | history_in_minutes | mr_nightmare_1hr")
    parser.add_argument("--output_dir", default="./output", help="Output directory")
    args = parser.parse_args()

    preset_path = os.path.join("presets", f"{args.preset}.yaml")
    if not os.path.exists(preset_path):
        raise FileNotFoundError(f"Preset not found: {preset_path}")

    with open(preset_path, "r") as f:
        preset = yaml.safe_load(f)

    project_slug = slugify(args.topic)
    project_dir = os.path.join(args.output_dir, f"{args.preset}_{project_slug}")
    os.makedirs(project_dir, exist_ok=True)

    print(f"[1/4] Generating Storyboard via Groq ({preset['script'].get('groq_model')}) for [{preset['channel_style']}]...")
    storyboard_path = generate_full_storyboard(args.topic, preset_path, project_dir)

    with open(storyboard_path, "r") as f:
        master_storyboard = json.load(f)

    audio_engine = AudioEngine()
    visual_engine = UniversalVisualEngine()
    chapter_mp4s = []

    print(f"[2/4 & 3/4] Processing Audio (Kokoro-82M) & Visuals (SDXL-Lightning + Pillow) Unit-by-Unit...")
    for idx, chapter in enumerate(master_storyboard["chapters"]):
        print(f" -> Processing Unit {idx+1}/{len(master_storyboard['chapters'])}: {chapter['chapter_title']}")

        # 1. Generate Voiceover WAVs & record exact durations
        chapter = audio_engine.generate_chapter_audio(chapter, preset, project_dir)

        # Save updated audio durations to storyboard
        with open(storyboard_path, "w") as sf_out:
            json.dump(master_storyboard, sf_out, indent=2)

        # 2. Generate Scene Images / Info Cards
        for scene in chapter["scenes"]:
            sid = scene["scene_id"]
            img_path = os.path.join(project_dir, f"scene_{sid:04d}.png")
            visual_engine.generate_scene_image(scene, preset, img_path)

        # 3. Render Chapter MP4 via FFmpeg
        chap_mp4 = render_chapter_to_mp4(chapter, preset, project_dir)
        chapter_mp4s.append(chap_mp4)

    print(f"[4/4] Concatenating Final Master Video...")
    master_list = os.path.join(project_dir, "master_list.txt")
    with open(master_list, "w") as ml:
        for cm in chapter_mp4s:
            ml.write(f"file '{os.path.abspath(cm)}'\n")

    final_output = os.path.join(project_dir, f"{project_slug}_final.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", master_list, "-c", "copy", final_output
    ], check=True)

    print(f"\n[SUCCESS] Final video generated at: {final_output}")

if __name__ == "__main__":
    main()
