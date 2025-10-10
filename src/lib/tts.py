import os
import wave
import time

from lib import file_manager

class TextToSpeech:
    def __init__(self, model_path=os.path.join(file_manager.tts_model_dir,"fr_FR-upmc-medium.onnx")):
        # Stub: do not load Piper model; keep interface identical
        self.model = None

    def text_to_speech(self, txt_path, output_path=None):
        # Read text just to keep same behavior, though we ignore it in stub
        with open(txt_path, "r", encoding="utf-8") as f:
            _ = f.read()

        start_time = time.time()
        timestamp = int(time.time() * 1000)

        if output_path is None:
            output_path = os.path.join(os.path.dirname(txt_path), f"out_{timestamp}.wav")
        else:
            if os.path.isdir(output_path):
                output_path = os.path.join(output_path, f"out_{timestamp}.wav")

        output_path = str(output_path)

        # Generate a short silent WAV so downstream can play something
        sample_rate = 22050
        duration_seconds = 1
        num_frames = sample_rate * duration_seconds
        n_channels = 1
        sampwidth = 2  # 16-bit PCM

        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(n_channels)
            wav_file.setsampwidth(sampwidth)
            wav_file.setframerate(sample_rate)
            silence = b"\x00\x00" * num_frames
            wav_file.writeframes(silence)

        delta = time.time() - start_time
        print(f"TTS (stub) completed in {delta:.2f} seconds -> {output_path}")
        return output_path

myTTS = TextToSpeech()
