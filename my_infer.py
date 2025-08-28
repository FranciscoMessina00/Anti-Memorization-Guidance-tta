import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.my_generation import my_generate_diffusion_cond

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda:3"
amg_type = "dedup_despec"

# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)

tot = 14.436
cfg_scale = 7
id = 2627

# Set up text and timing conditioning
conditioning = [{
    # "prompt": 'Drum Loop by Stolting Media Group.\nvisit <a href=\"http://www.stoltingmediagroup.com\" rel=\"nofollow\">http://www.stoltingmediagroup.com</a>',
    "prompt": 'This is loop 5 in a series of 16 variations. The style is pure electro. The pitch of the melodic sounds is C. Created with the Waldorf Attack VSTi within Cubase SX. I used the Analog kit 2 preset to create this 133 bpm loop. It lasts 8 measures in 4/4. Mastered using Elemental and TC plugins within Wavelab.',
    "seconds_start": 0, 
    "seconds_total": tot
}]
negative_conditioning = [{
    "prompt": "",
    "seconds_start": 0,
    "seconds_total": tot
}]

print("Model objective: " + model.diffusion_objective)
for i in range(100):
# Generate stereo audio
    # with torch.no_grad():
        output, closest_id = my_generate_diffusion_cond(
            model,
            steps=100,
            cfg_scale=cfg_scale,
            conditioning=conditioning,
            negative_conditioning=negative_conditioning,
            sample_size=sample_size,
            sample_rate=sample_rate,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="my-dpmpp-3m-sde",
            device=device,
            # seed = 2556487306,
            c1=cfg_scale-1, # cfg_scale-1
            c2=cfg_scale-1, # cfg_scale-1
            c3=0, #1000
            generation=i
        )

        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")

        num_samples = int(conditioning[0]["seconds_total"] * sample_rate)
        output = output[:, :num_samples]

        # Peak normalize, clip, convert to int16, and save to file
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

        # Ensure output directory exists, and skip saving if the file already exists
        audio_dir = f"/nas/home/fmessina/Experiments/{id}_gens/{amg_type}"
        os.makedirs(audio_dir, exist_ok=True)
        audio_path = os.path.join(audio_dir, f"{id}_generation_{i}.wav")
        if os.path.exists(audio_path):
            print(f"[INFO] Audio already exists, skipping save: {audio_path}")
        else:
            torchaudio.save(audio_path, output, sample_rate)
            print(f"[INFO] Saved: {audio_path}")

        # Ensure closest list directory exists, check file, then append
        closest_dir = f"/nas/home/fmessina/Experiments/{id}_gens/graphs/closest_list"
        os.makedirs(closest_dir, exist_ok=True)
        closest_file = os.path.join(closest_dir, f"closest_{amg_type}.txt")
        if not os.path.exists(closest_file):
            # create the file if missing
            open(closest_file, "w").close()
            print(f"[INFO] Created closest list file: {closest_file}")
        with open(closest_file, "a") as f:
            f.write(f"{closest_id}\n")