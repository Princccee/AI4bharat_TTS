import torch
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
import argparse

# 1. Load the TTS model and tokenizers
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📦 Loading Indic-Parler-TTS model on {device.upper()}...")

tts_model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
tts_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
desc_tokenizer = AutoTokenizer.from_pretrained(tts_model.config.text_encoder._name_or_path)

print("✅ Model loaded successfully!\n")

# 2. Function to generate speech
def text_to_speech(text, description="एक महिला वक्ता हिंदी में बोल रही हैं", output_path="output.wav"):
    print("🎙️ Converting text to speech...")

    # Encode description and input text
    desc_inputs = desc_tokenizer(description, return_tensors="pt").to(device)
    text_inputs = tts_tokenizer(text, return_tensors="pt").to(device)

    # Generate audio
    audio = tts_model.generate(
        input_ids=desc_inputs.input_ids,
        attention_mask=desc_inputs.attention_mask,
        prompt_input_ids=text_inputs.input_ids,
        prompt_attention_mask=text_inputs.attention_mask
    )

    # Save audio
    sf.write(output_path, audio.cpu().numpy().squeeze(), tts_model.config.sampling_rate)
    print(f"🔊 Audio saved as: {output_path}")

# 3. Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Indian language text to speech using Indic-Parler-TTS.")
    parser.add_argument("--text", type=str, required=True, help="Input text in any Indian language (e.g. हिंदी, தமிழ், etc.)")
    parser.add_argument("--desc", type=str, default="एक महिला वक्ता हिंदी में बोल रही हैं", help="Optional description (controls speaker characteristics)")
    parser.add_argument("--out", type=str, default="output.wav", help="Output audio file name")

    args = parser.parse_args()
    text_to_speech(args.text, args.desc, args.out)
