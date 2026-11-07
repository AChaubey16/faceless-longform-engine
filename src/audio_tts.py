import os
import urllib.request
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

class AudioEngine:
    def __init__(self):
        self.kokoro = None

    def load_model(self):
        if self.kokoro is not None:
            return
        
        model_filename = "kokoro-v0_19.onnx"
        voices_filename = "voices.bin"

        if not os.path.exists(model_filename):
            print("[+] Downloading kokoro-v0_19.onnx from GitHub releases...")
            urllib.request.urlretrieve(
                "[https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx)",
                model_filename
            )

        if not os.path.exists(voices_filename):
            print("[+] Downloading voices.bin from GitHub releases...")
            urllib.request.urlretrieve(
                "[https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin)",
                voices_filename
            )

        print("[+] Loading Kokoro ONNX engine...")
        self.kokoro = Kokoro(model_filename, voices_filename)

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

            text = scene.get("narration_text", "").strip()
            if not text:
                text = "..."

            samples, sample_rate = self.kokoro.create(text, voice=voice_id, speed=speed, lang="en-us")
            pad_samples = int(sample_rate * (pause_ms / 1000.0))
            padded_audio = np.concatenate([samples, np.zeros(pad_samples, dtype=np.float32)])

            sf.write(wav_path, padded_audio, sample_rate)
            info = sf.info(wav_path)
            scene["audio_duration"] = round(info.duration, 2)

        return chapter
