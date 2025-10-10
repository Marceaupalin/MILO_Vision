from lib import transcriber
from lib import recorder
from lib import subsynthetizer
from lib import message_queue

from pathlib import Path

import threading
import time
import glob
import os
import signal

running = True

script_dir = Path(__file__).resolve().parent

def handler(signum, frame):
    global running
    running = False

signal.signal(signal.SIGTERM, handler)

def wait_for_user_input(recorder, transcriber, synthesizer):
    global running
    while running:
        time.sleep(1)

    print("\n\n\n\n")
    recorder.stop()
    time.sleep(1)

    audio_files = glob.glob(str(Path(__file__).resolve().parent.parent / "audio" / "recorder" / "*.wav"))
    if audio_files:

        latest_audio = Path(max(audio_files, key=os.path.getctime))

        print(f"[MAIN] Last audio file : {latest_audio}")
        print("[MAIN] Transcribe last audio...")
        transcriber.transcribe_file(latest_audio)

        transcript_file = f"synthetiser/transcripts/{Path(latest_audio).stem}.txt"
        timeout = 5
        waited = 0
        while not os.path.exists(transcript_file) and waited < timeout:
            time.sleep(0.5)
            waited += 0.5

        if os.path.exists(transcript_file):
            print(f"[MAIN] Last transcript: {transcript_file}")
            print("[MAIN] Synthesis last transcript...")
            synthesizer.generate_from_file(transcript_file)
        else:
            print("[MAIN] Failed to synthetise")
    else:
        print("[MAIN] ERROR No audio detected.")

    message_queue.message_queue_handler.stop()

    print("[MAIN] EXIT_SUCCESS")

def handle_new_audio_file(file_path, ObjTranscriber):
    ObjTranscriber.transcribe_file(Path(file_path))

def handle_new_transcript(file_path, ObjLlama):
    ObjLlama.generate_from_file(Path(file_path))

def setup_listeners(message_queue_handler=message_queue.message_queue_handler):
    message_queue_handler.subscribe("Audio_topic", "Audio_listener", callback=lambda msg: handle_new_audio_file(msg, transcriber.myTranscrib))
    message_queue_handler.subscribe("Transcriber_topic", "Transcriber_listener", callback=lambda msg: handle_new_transcript(msg, subsynthetizer.mySynthetizer))

if __name__ == "__main__":

    transcriber.myTranscrib.clearTransciptDir()
    recorder.myRecorder.clearAudioDir()

    setup_listeners(message_queue.message_queue_handler)

    subsynthetizer.mySynthetizer.clearSubSynthetizerDir()

    transcriber.myTranscrib.setAudioDir(str(script_dir / "audio" / "recorder"))
    transcriber.myTranscrib.load_model()

    recorder.myRecorder.start()

    input_thread = threading.Thread(
        target=wait_for_user_input,
        args=(recorder.myRecorder, transcriber.myTranscrib, subsynthetizer.mySynthetizer)
    )
    input_thread.start()
    input_thread.join()

    print("fin")
