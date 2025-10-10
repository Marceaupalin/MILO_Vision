import piper
import os
import wave
import time

from lib import file_manager

class TextToSpeech:
    def __init__(self, model_path=os.path.join(file_manager.tts_model_dir,"fr_FR-upmc-medium.onnx")):
        self.model = piper.PiperVoice.load(model_path)

    def text_to_speech(self, txt_path, output_path=None):
        with open(txt_path, "r", encoding="utf-8") as f:
            txt = f.read()

        start_time = time.time()
        timestamp = int(time.time() * 1000)

        if output_path is None:
            output_path = os.path.join(os.path.dirname(txt_path), f"out_{timestamp}.wav")
        else:
            if os.path.isdir(output_path):
                output_path = os.path.join(output_path, f"out_{timestamp}.wav")

        output_path = str(output_path)

        with wave.open(output_path, "wb") as wav_file:
            self.model.synthesize_wav(txt, wav_file)

        delta = time.time() - start_time
        print(f"TTS completed in {delta:.2f} seconds -> {output_path}")
        return output_path

myTTS = TextToSpeech()
