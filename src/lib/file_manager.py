from pathlib import Path

project_root_dir = Path(__file__).resolve().parent.parent.parent

wav_dir = project_root_dir/ "audio" / "recorder" / "wav"
webm_dir = project_root_dir / "audio" / "recorder" / "webm"

milo_wav_response_dir = project_root_dir / "audio" / "milo_audio" / "wav"
milo_webm_response_dir = project_root_dir / "audio" / "milo_audio" / "webm"

milo_wav_question_dir = project_root_dir / "audio" / "milo_question" / "wav"
milo_webm_question_dir = project_root_dir / "audio" / "milo_question" / "webm"

milo_webm_question_response_dir = project_root_dir / "audio" / "milo_response" / "webm"
milo_wav_question_response_dir = project_root_dir / "audio" / "milo_response" / "wav"

transcript_dir = project_root_dir / "synthetiser" / "transcripts"
question_transcript_dir = project_root_dir / "synthetiser" / "question_transcripts"
milo_response_dir = project_root_dir / "synthetiser" / "response"
sub_resume_dir = project_root_dir / "synthetiser" / "sub_resumes"

tts_model_dir = project_root_dir / "audio" / "tts_models"

backup_transcript = project_root_dir / "backup_transcripts"

FINAL_TRANSCRIPT = transcript_dir / "transcript_final.txt"

def clearDirectory(path):
    if not path.exists():
            print(f"Folder {path} don't exist.")
            return

    file_count = 0
    for file in path.iterdir():
        if file.is_file():
            try:
                file.unlink()
                file_count += 1
            except Exception as e:
                print(f"Error: {file.name} : {e}")

        print(f"{file_count} file deleted from {path}")

def clearAllDirectories():
    clearDirectory(webm_dir)
    clearDirectory(wav_dir)
    clearDirectory(milo_wav_response_dir)
    clearDirectory(milo_webm_response_dir)
    clearDirectory(sub_resume_dir)
    clearDirectory(transcript_dir)
    clearDirectory(milo_wav_question_dir)
    clearDirectory(milo_webm_question_dir)
    clearDirectory(milo_wav_question_response_dir)
    clearDirectory(milo_webm_question_response_dir)
    clearDirectory(question_transcript_dir)
    clearDirectory(milo_response_dir)

def create_final_transcript():
    transcript_dir.mkdir(parents=True, exist_ok=True)
    if not FINAL_TRANSCRIPT.exists():
        FINAL_TRANSCRIPT.touch()
        print(f"{FINAL_TRANSCRIPT} créé.")
    else:
        print(f"{FINAL_TRANSCRIPT} existe déjà.")


def append_and_delete_transcript(filename):

    file_path = transcript_dir / filename
    if not file_path.exists():
        print(f"Le fichier {file_path} n'existe pas.")
        return

    with file_path.open("r", encoding="utf-8") as f_src, FINAL_TRANSCRIPT.open("a", encoding="utf-8") as f_dest:
        f_dest.write(f_src.read())

    try:
        file_path.unlink()
        print(f"{file_path} a été concaténé dans {FINAL_TRANSCRIPT} et supprimé.")
    except Exception as e:
        print(f"Erreur lors de la suppression de {file_path}: {e}")