from flask import Flask, request, jsonify, send_file, render_template
import os
import torch
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from utils import generate_tts

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load TTS model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tts_model_name = "ai4bharat/indic-parler-tts"
tts_model = ParlerTTSForConditionalGeneration.from_pretrained(tts_model_name).to(device)
tts_tokenizer = AutoTokenizer.from_pretrained(tts_model_name)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/tts', methods=['POST'])
def tts():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request."}), 400

    text = data["text"]
    lang = data.get("lang", "hin")
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], "tts_output.wav")

    try:
        generate_tts(text, lang, output_path, tts_model, tts_tokenizer, device)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return send_file(output_path, mimetype="audio/wav", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
