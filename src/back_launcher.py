from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
import os
import threading
import time
import shutil

from lib import transcriber, subsynthetizer, file_manager, webm_to_wav_converter, tts
from lib import message_queue
from lib import vision

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
last_chunk_event = threading.Event()

# Vision thread handle
vision_thread = None


BASE_DIR = Path(__file__).resolve().parent.parent
FRONT_DIR = BASE_DIR / "front"

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_front(path):
    if path != "" and (FRONT_DIR / path).exists():
        return send_from_directory(FRONT_DIR, path)
    else:
        return send_from_directory(FRONT_DIR, "index.html")


@app.route("/start-vision", methods=["POST"])
def start_vision():
    global vision_thread
    try:
        if vision_thread is None or not vision_thread.is_alive():
            vision_thread = threading.Thread(target=vision.vision_loop, args=(socketio,), daemon=True)
            vision_thread.start()
            return {"status": "ok", "message": "Vision loop started"}, 200
        else:
            return {"status": "ok", "message": "Vision loop already running"}, 200
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500


@app.route("/upload-audio", methods=["POST"])
def upload_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(file_manager.webm_dir, filename)
    file.save(filepath)

    last_chunk = request.form.get("last_chunk", "false").lower() == "true"

    if last_chunk:
        last_chunk_event.set()

    message_queue.message_queue_handler.publish(
        "Audio_topic", {"filename": filename, "last_chunk": str(last_chunk)}
    )

    return jsonify({"status": "ok", "saved_as": filepath, "last_chunk": last_chunk})

@app.route("/get-audio/<filename>")
def get_audio(filename):
    filename = secure_filename(filename)
    return send_from_directory(file_manager.milo_webm_response_dir, filename)

@app.route("/get-response-audio/<filename>")
def get_response_audio(filename):
    filename = secure_filename(filename)
    return send_from_directory(file_manager.milo_wav_question_response_dir, filename)

@app.route("/start-recording", methods=["POST"])
def start_recording():
    try:
        file_manager.clearDirectory(file_manager.webm_dir)
        file_manager.clearDirectory(file_manager.wav_dir)
        file_manager.clearDirectory(file_manager.milo_wav_response_dir)
        file_manager.clearDirectory(file_manager.milo_webm_response_dir)
        file_manager.clearDirectory(file_manager.sub_resume_dir)
        file_manager.clearDirectory(file_manager.transcript_dir)
        file_manager.create_final_transcript()
        message_queue.clearAllStreams()
        last_chunk_event.clear()
        return {"status": "ok"}, 200
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

@app.route("/upload-question", methods=["POST"])
def upload_question():
    file_manager.clearDirectory(file_manager.milo_wav_question_dir)
    file_manager.clearDirectory(file_manager.milo_webm_question_dir)
    file_manager.clearDirectory(file_manager.milo_wav_question_response_dir)
    file_manager.clearDirectory(file_manager.milo_webm_question_response_dir)
    file_manager.clearDirectory(file_manager.question_transcript_dir)
    file_manager.clearDirectory(file_manager.milo_response_dir)
    message_queue.clearAllStreams()
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(file_manager.milo_webm_question_dir, filename)
    file.save(filepath)

    message_queue.message_queue_handler.publish(
        "Question_topic", {"filename": filename}
    )
    return {"status": "ok"}, 200

def handle_new_audio_file(msg, ObjTranscriber):
    filename = msg["filename"]
    last_chunk = msg.get("last_chunk", "False") == "True"
    wav_file = webm_to_wav_converter.convert_to_wav(
        file_manager.webm_dir,
        file_manager.wav_dir,
        filename
        )
    transcript_file = ObjTranscriber.transcribe_file(Path(wav_file))
    file_manager.append_and_delete_transcript(transcript_file)

    if last_chunk:
        print("Tous les chunks reçus, génération finale...")
        message_queue.message_queue_handler.publish("Transcriber_topic", {"filepath": f"{file_manager.transcript_dir}/transcript_final.txt"})

def handle_new_transcript(msg, ObjLlama):
    file_path = msg["filepath"]
    ObjLlama.generate_from_file(Path(file_path))

    backup_dir= file_manager.backup_transcript
    backup_dir.mkdir(exist_ok=True, parents=True)

    for src_dir in [file_manager.transcript_dir, file_manager.sub_resume_dir]:
        for item in src_dir.iterdir():
            dest = backup_dir / item.name
            shutil.copy2(item, dest)

    milo_wav = tts.myTTS.text_to_speech(file_manager.sub_resume_dir / "transcript_final_resume.txt", file_manager.milo_wav_response_dir)
    milo_webm = webm_to_wav_converter.convert_to_webm(
        milo_wav,
        file_manager.milo_webm_response_dir
    )
    socketio.emit("new_audio", {"filename": os.path.basename(milo_webm)})

def handle_new_question(msg, ObjTranscriber):
    print(f"NEW Question :{msg}")
    filename = msg["filename"]
    wav_file = webm_to_wav_converter.convert_to_wav(
        file_manager.milo_webm_question_dir,
        file_manager.milo_wav_question_dir,
        filename
        )
    transcript_path=ObjTranscriber.transcribe_file(Path(wav_file),file_manager.question_transcript_dir)
    message_queue.message_queue_handler.publish("Response_topic",{"filepath": f"{file_manager.question_transcript_dir}/{transcript_path}"})

def handle_new_response(msg, ObjLlama):
    file_path = msg["filepath"]
    output_name=ObjLlama.generate_from_file(Path(file_path),True,file_manager.milo_response_dir)
    milo_response_wav = tts.myTTS.text_to_speech(file_manager.milo_response_dir / output_name, file_manager.milo_wav_question_response_dir)
    #milo_response_webm = webm_to_wav_converter.convert_to_webm(
    #    milo_response_wav,
    #    file_manager.milo_webm_question_response_dir
    #)
    socketio.emit("new_response_audio", {"filename": os.path.basename(milo_response_wav)})

def setup_listeners():
    message_queue.message_queue_handler.subscribe("Audio_topic", "Audio_listener", callback=lambda msg: handle_new_audio_file(msg, transcriber.myTranscrib))
    message_queue.message_queue_handler.subscribe("Transcriber_topic", "Transcriber_listener", callback=lambda msg: handle_new_transcript(msg, subsynthetizer.mySynthetizer))
    message_queue.message_queue_handler.subscribe("Question_topic", "Question_listener", callback=lambda msg: handle_new_question(msg, transcriber.myTranscrib))
    message_queue.message_queue_handler.subscribe("Response_topic", "Response_listener", callback=lambda msg: handle_new_response(msg, subsynthetizer.mySynthetizer))

if __name__ == "__main__":
    file_manager.clearAllDirectories()
    message_queue.clearAllStreams()
    file_manager.create_final_transcript()
    setup_listeners()
    transcriber.myTranscrib.load_model()
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False)


