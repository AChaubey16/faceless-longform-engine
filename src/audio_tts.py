import os
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro
from huggingface_hub import hf_hub_download

class AudioEngine:
    def __init__(self):
        self.kokoro = None

    def load_model(self):
        if self.kokoro is not None:
            return
        model_path = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="kokoro-v1.0.onnx")
        voices_path = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="voices.json")
        self.kokoro = Kokoro(model_path, voices_path)

    def generate_chapter_audio(self, chapter: dict, preset: dict, project_dir: str) -> dict:
        self.load_model()
        voice_id = preset["audio"]["voice_id"]
        speed = float(preset["audio"]["speed"])
        pause_ms = int(preset["audio"].get("sentence_pause_ms", 250))

        for scene in chapter["scenes"]:
            sid = scene["scene_id"]
            wav_path = os.path.join(project_dir, f"scene_{sid:04d}.wav")

            if os.path.exists(wav_path):
                info = sf.info(wav_path)
                scene["audio_duration"] = round(info.duration, 2)
                continue

            text = scene["narration_text"].strip()
            if not text:
                text = "..."

            samples, sample_rate = self.kokoro.create(text, voice=voice_id, speed=speed, lang="en-us")

            # Append natural pause padding at the end of each scene clip
            pad_samples = int(sample_rate * (pause_ms / 1000.0))
            padded_audio = np.concatenate([samples, np.zeros(pad_samples, dtype=np.float32)])

            sf.write(wav_path, padded_audio, sample_rate)
            info = sf.info(wav_path)
            scene["audio_duration"] = round(info.duration, 2)

        return chapter
