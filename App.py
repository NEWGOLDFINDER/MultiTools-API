import os
from flask import Flask, request, jsonify
import whisper
import uuid # <-- Ajoutez cette ligne pour générer des noms de fichiers uniques
from flask_cors import CORS # <-- Ajoutez cette ligne (si vous aviez des erreurs CORS)

app = Flask(__name__)
CORS(app) # <-- Ajoutez cette ligne (si vous aviez des erreurs CORS)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

WHISPER_MODEL = "base" # ou 'tiny' si 'base' est trop lent/gourmand

print(f"Chargement du modèle Whisper '{WHISPER_MODEL}'... Cela peut prendre un moment la première fois.")
try:
    model = whisper.load_model(WHISPER_MODEL)
    print(f"Modèle '{WHISPER_MODEL}' chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle Whisper : {e}")
    exit(1)


@app.route('/')
def home():
    return "API Speech-to-Text en marche ! Utilisez la route /transcribe pour envoyer un fichier audio."

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio_file' not in request.files:
        return jsonify({"error": "Aucun fichier 'audio_file' fourni dans la requête."}), 400

    audio_file = request.files['audio_file']

    if audio_file.filename == '':
        return jsonify({"error": "Nom de fichier vide."}), 400

    if audio_file:
        # Générer un nom de fichier unique et sûr
        # Conserver l'extension du fichier original
        original_filename = audio_file.filename
        file_extension = os.path.splitext(original_filename)[1] # Récupère ".mp3", ".wav" etc.
        unique_filename = str(uuid.uuid4()) + file_extension # Génère un UUID + extension
        
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename) # Utilise le nom unique
        
        audio_file.save(filepath)
        print(f"Fichier sauvegardé temporairement sous : {filepath} (Original : {original_filename})")

        try:
            print(f"Début de la transcription de {unique_filename} avec le modèle {WHISPER_MODEL}...")
            result = model.transcribe(filepath)
            transcription = result["text"]
            print(f"Transcription terminée pour {unique_filename}.")

            os.remove(filepath)
            print(f"Fichier temporaire supprimé : {filepath}")

            return jsonify({"success": True, "transcription": transcription}), 200

        except Exception as e:
            print(f"Erreur lors de la transcription de {unique_filename} : {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"success": False, "error": str(e)}), 500
    
    return jsonify({"error": "Une erreur inattendue s'est produite."}), 500


if __name__ == '__main__':
    print("\nAPI Flask démarrée. Accédez à http://127.0.0.1:5000/")
    print("Pour tester la transcription, vous devrez envoyer un fichier POST à http://127.0.0.1:5000/transcribe")
    app.run(debug=True)