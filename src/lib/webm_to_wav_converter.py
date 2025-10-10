import os
import subprocess
from pathlib import Path


def convert_to_wav(input_dir, output_dir, file_name):
    """
    Convertit un fichier .webm en .wav
    - input_dir: dossier contenant le fichier source
    - output_dir: dossier de sortie
    - file_name: nom du fichier webm
    """
    input_path = Path(input_dir) / file_name

    if not input_path.is_file():
        raise FileNotFoundError(f"Le fichier {input_path} n'existe pas.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = input_path.stem
    output_path = output_dir / f"{base_name}.wav"

    print(f"Conversion de {input_path} -> {output_path}")
    command = [
        "ffmpeg",
        "-i", str(input_path),
        "-ar", "22050",  # fréquence d'échantillonnage
        "-ac", "2",      # stéréo
        str(output_path)
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)

    return str(output_path)


def convert_to_webm(input_file_path, output_dir):
    """
    Convertit un fichier audio en .webm (Opus)
    - input_file_path: chemin complet du fichier source
    - output_dir: dossier de sortie
    """
    input_path = Path(input_file_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Le fichier {input_path} n'existe pas.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{input_path.stem}.webm"

    print(f"[convert_to_webm] Conversion de {input_path} -> {output_path}")

    command = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-c:a", "libopus",
        "-b:a", "128k",
        str(output_path)
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("[convert_to_webm] Erreur ffmpeg :", result.stderr)
        raise RuntimeError(f"ffmpeg a échoué pour {input_path}")

    return str(output_path)
