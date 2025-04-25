import torch
import soundfile as sf

def generate_tts(text, lang="hin", output_path="tts_output.wav", tts_model=None, tts_tokenizer=None, device="cpu"):
    prompt = f"A clear studio recording of a native {lang} speaker speaking the following text naturally."

    text_inputs = tts_tokenizer(text, return_tensors="pt").to(device)
    prompt_inputs = tts_tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_audio = tts_model.generate(
            input_ids=text_inputs.input_ids,
            prompt_input_ids=prompt_inputs.input_ids,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
        )

    audio_array = generated_audio.cpu().numpy().squeeze()
    sf.write(output_path, audio_array, 16000)
