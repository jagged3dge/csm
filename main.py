from dotenv import load_dotenv
load_dotenv()

from generator import load_csm_1b, Segment
import torchaudio
import torch
import time

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Start process metric timer
process_start_time = time.time()

# Start model metric timer
model_start_time = time.time()
generator = load_csm_1b(device=device)
model_end_time = time.time()
model_duration = model_end_time - model_start_time
print(f"Model duration: {model_duration:.2f} seconds")


# Start audio metric timer
audio_start_time = time.time()
speakers = [0, 0, 0, 0]
transcripts = [
    "Hello, thank you for calling Sunnyvale Medical Center. I'm here to assist you today—how may I help?",
    "I'd be happy to help you schedule an appointment. Could you please let me know your preferred date and time, and which doctor you'd like to see?",
    "Just to confirm, your appointment with Dr. Smith is scheduled for October 15th at 2:30 PM. We look forward to seeing you then.",
    "I'm pleased to share that your test results came back normal. There's nothing to worry about, but feel free to reach out if you have any questions.",
]
audio_paths = [
    "/home/jagged/Work/apex-ehr/research/zonos-outputs/11-jessica-01.mp3",
    "/home/jagged/Work/apex-ehr/research/zonos-outputs/11-jessica-02.mp3",
    "/home/jagged/Work/apex-ehr/research/zonos-outputs/11-jessica-03.mp3",
    "/home/jagged/Work/apex-ehr/research/zonos-outputs/11-jessica-04.mp3",
]


def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor


segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]
audio_end_time = time.time()
audio_duration = audio_end_time - audio_start_time
print(f"Audio duration: {audio_duration:.2f} seconds")


# Start generation metric timer
generation_start_time = time.time()
audio = generator.generate(
    text="I hear the urgency in your voice. Please stay on the line while I connect you to our emergency team immediately.",
    speaker=1,
    context=segments,
    max_audio_length_ms=10_000,
)
generation_end_time = time.time()
generation_duration = generation_end_time - generation_start_time
print(f"Generation duration: {generation_duration:.2f} seconds")


# Save audio
audio_save_start_time = time.time()
torchaudio.save("audio-02.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
audio_save_end_time = time.time()
audio_save_duration = audio_save_end_time - audio_save_start_time
print(f"Audio save duration: {audio_save_duration:.2f} seconds")

# End process metric timer
process_end_time = time.time()
process_duration = process_end_time - process_start_time
print(f"Process duration: {process_duration:.2f} seconds")
