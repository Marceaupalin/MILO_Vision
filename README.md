# MILO Vision

Assistant IA pour l'analyse audio et visuelle en temps réel.

## Installation

### 1. Dépendances système

**FFmpeg** (requis pour la conversion audio) :
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt install ffmpeg`
- **Windows**: Télécharger depuis https://ffmpeg.org/download.html

### 2. Environnement Python

```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances Python
pip install -r requirements.txt
```

### 3. Pré-télécharger les modèles IA

```bash
python src/preload_models.py
```

## Lancement

```bash
python src/back_launcher.py
```

Ouvrir http://127.0.0.1:5000/ dans votre navigateur.

## Troubleshooting

**Port 5000 occupé** :
- **macOS** : Désactiver 'AirPlay Receiver' dans Préférences Système → Général → AirDrop et Handoff
- **Alternative** : Identifier et arrêter le programme utilisant le port 5000

## Fonctionnalités

- 🎤 **Enregistrement audio** : Transcription et résumé automatique
- 👁️ **Vision** : Détection d'objets et description de scènes
- 🤖 **IA conversationnelle** : Questions/réponses contextuelles
