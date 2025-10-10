from pathlib import Path
import time
import os

from lib import file_manager
from lib import message_queue

class Transcriber:

    def __init__(self, model_size="medium", device="cuda", compute_type="int8_float16"):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        # Stub mode: no actual model is loaded so the app can run without Whisper
        self._model = None

        self._output_dir = file_manager.transcript_dir
        self._output_dir.mkdir(exist_ok=True, parents=True)

        self._audio_dir = file_manager.wav_dir
        self._audio_dir.mkdir(exist_ok=True, parents=True)

    def setModelSize(self, model_size):
        self._model_size = model_size

    def setOutputDir(self, output_dir):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(exist_ok=True, parents=True)

    def setAudioDir(self, audio_dir):
        self._audio_dir = Path(audio_dir)
        self._audio_dir.mkdir(exist_ok=True, parents=True)

    def setup_model(self, model_size, output_dir, audio_dir):
        self.setModelSize(model_size)
        self.setOutputDir(output_dir)
        self.setAudioFile(audio_dir)

    def load_model(self):
        # Stub: do nothing, keep self._model as None
        self._model = None

    def clearTransciptDir(self):
        if not self._output_dir.exists():
            print(f"Folder {self._output_dir} don't exist.")
            return

        file_count = 0
        for file in self._output_dir.iterdir():
            if file.is_file():
                try:
                    file.unlink()
                    file_count += 1
                except Exception as e:
                    print(f"Error: {file.name} : {e}")

        print(f"{file_count} file deleted from {self._output_dir}")

    def transcribe_all_files(self):

        audio_files= list(self._audio_dir.glob("*.wav"))
        if not audio_files:
            print(f"No audio file found in dir : {self._audio_dir}")
            return

        for audio_path in audio_files:
            self.transcribe_file(audio_path)

    def transcribe_file(self, audio_path, output_dir=None):
        audio_path = Path(audio_path)

        # Si output_dir est fourni, on l'utilise, sinon on garde self._output_dir
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        else:
            output_dir = self._output_dir

        output_path = output_dir / (audio_path.stem + ".txt")

        if output_path.exists():
            print(f"{output_path.name} already exist, pass")
            return

        print(f"Begin transcript (stub) of : {audio_path.name}")
        start_time = time.time()

        # Stub output: deterministic placeholder transcript so downstream pipeline can run
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("[0.00 - 1.23] Ceci est un transcript factice genere sans modele.\n")
            f.write("[1.23 - 3.45] Le systeme fonctionne en mode demo sans Whisper.\n")

        delta = time.time() - start_time
        print(f"File saved to : {output_path}")
        # message_queue.message_queue_handler.publish("Transcriber_topic", f"{output_path}")
        print(f"Transcription (stub) completed in {delta:.2f} seconds")

        return output_path.name


myTranscrib = Transcriber()