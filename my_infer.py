import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.my_generation import my_generate_diffusion_cond

device = "cuda" if torch.cuda.is_available() else "cpu"

# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)

# Ensure output directory exists, and skip saving if the file already exists
audio_dir = "/nas/home/fmessina/"
os.makedirs(audio_dir, exist_ok=True)
audio_path = os.path.join(audio_dir, "audio.wav")

total_duration = 10.0  # seconds
cfg_scale = 7
c1 = 4
c2 = 3
c3 = 100
lambda_min = 0.7
lambda_max = 0.8
denoising_steps = 100

prompt = "An acoustic drum loop, 110 bpm"

# Set up text and timing conditioning
conditioning = [{
    "prompt": prompt,
    "seconds_start": 0, 
    "seconds_total": total_duration
}]
negative_conditioning = [{
    "prompt": "",
    "seconds_start": 0,
    "seconds_total": total_duration
}]

output = my_generate_diffusion_cond(
    model,
    steps=denoising_steps,
    cfg_scale=cfg_scale,
    conditioning=conditioning,
    negative_conditioning=negative_conditioning,
    sample_size=sample_size,
    sample_rate=sample_rate,
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="my-dpmpp-3m-sde",
    device=device,
    c1=c1,
    c2=c2,
    c3=c3,
    lambda_min=lambda_min,
    lambda_max=lambda_max,
)

# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")

num_samples = int(conditioning[0]["seconds_total"] * sample_rate)
output = output[:, :num_samples]

# Peak normalize, clip, convert to int16, and save to file
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()


torchaudio.save(audio_path, output, sample_rate)
print(f"[INFO] Saved: {audio_path}")