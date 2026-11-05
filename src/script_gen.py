import os
import json
import time
import yaml
from groq import Groq

def generate_full_storyboard(topic: str, preset_path: str, project_dir: str) -> str:
    os.makedirs(project_dir, exist_ok=True)
    storyboard_path = os.path.join(project_dir, "storyboard.json")

    with open(preset_path, "r") as f:
        preset = yaml.safe_load(f)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set! Set os.environ['GROQ_API_KEY'] = 'gsk_...'")

    client = Groq(api_key=api_key)
    model_name = preset["script"].get("groq_model", "llama-3.3-70b-versatile")
    num_units = preset["script"]["unit_count"]
    words_per_unit = preset["script"]["words_per_unit"]
    words_per_scene = preset["script"]["target_words_per_scene"]

    outline_file = os.path.join(project_dir, "outline.json")
    if os.path.exists(outline_file):
        with open(outline_file, "r") as of:
            master_data = json.load(of)
    else:
        outline_prompt = f"""
        Topic: {topic}
        Create an outline of exactly {num_units} chapters/stories for a '{preset['channel_style']}' video.
        Return strict JSON matching this exact structure:
        {{
          "video_title": "Compelling Video Title",
          "units": [
            {{
              "unit_number": 1,
              "title": "Chapter or Story Title",
              "premise": "Detailed 2-sentence summary of what happens in this unit"
            }}
          ]
        }}
        """
        outline_resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": preset["script"]["system_prompt"]},
                {"role": "user", "content": outline_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        master_data = json.loads(outline_resp.choices[0].message.content)
        with open(outline_file, "w") as of:
            json.dump(master_data, of, indent=2)

    master_data["chapters"] = []
    scene_counter = 1

    for unit in master_data["units"]:
        u_num = unit["unit_number"]
        unit_file = os.path.join(project_dir, f"unit_{u_num:02d}.json")

        if os.path.exists(unit_file):
            with open(unit_file, "r") as uf:
                chapter_data = json.load(uf)
                for s in chapter_data["scenes"]:
                    s["scene_id"] = scene_counter
                    scene_counter += 1
                master_data["chapters"].append(chapter_data)
                continue

        num_scenes = max(5, words_per_unit // words_per_scene)
        unit_prompt = f"""
        Overall Video Topic: {topic}
        Write Unit #{u_num}: "{unit['title']}"
        Premise: {unit['premise']}
        Target total length for this unit: ~{words_per_unit} words split across roughly {num_scenes} scenes (~{words_per_scene} words per scene).
        Allowed scene_types: {preset['visuals']['allowed_scene_types']}

        Return strict JSON matching this exact schema:
        {{
          "chapter_number": {u_num},
          "chapter_title": "{unit['title']}",
          "scenes": [
            {{
              "scene_type": "character_scene",
              "narration_text": "Spoken voiceover text for this scene...",
              "image_prompt": "Detailed visual description of physical subject and environment (NO text inside AI images)...",
              "card_text": "Short text ONLY if scene_type is title_card or split_comparison (use 'LEFT | RIGHT' for split_comparison), else empty string",
              "camera_motion": "zoom_in"
            }}
          ]
        }}
        Note: camera_motion must be one of: zoom_in, zoom_out, pan_right, pan_left.
        """
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": preset["script"]["system_prompt"]},
                {"role": "user", "content": unit_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.72,
        )
        chapter_data = json.loads(resp.choices[0].message.content)
        for s in chapter_data["scenes"]:
            s["scene_id"] = scene_counter
            scene_counter += 1

        with open(unit_file, "w") as uf:
            json.dump(chapter_data, uf, indent=2)

        master_data["chapters"].append(chapter_data)
        print(f"[+] Generated Unit {u_num}/{num_units}: {unit['title']} ({len(chapter_data['scenes'])} scenes)")
        time.sleep(1.5)  # Respect Groq free-tier rate limits

    with open(storyboard_path, "w") as f:
        json.dump(master_data, f, indent=2)

    return storyboard_path
