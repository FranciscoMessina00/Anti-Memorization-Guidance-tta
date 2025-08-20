import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.my_generation import my_generate_diffusion_cond

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda:4"
amg_type = "despec_dissim"

# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)

tot = 6
cfg_scale = 7

# Set up text and timing conditioning
conditioning = [{
    "prompt": 'Drum Loop by Stolting Media Group.\nvisit <a href=\"http://www.stoltingmediagroup.com\" rel=\"nofollow\">http://www.stoltingmediagroup.com</a>',
    # "prompt": 'A rhytmic guitar in Bossa Nova style',
    "seconds_start": 0, 
    "seconds_total": tot
}]
negative_conditioning = [{
    "prompt": "",
    "seconds_start": 0,
    "seconds_total": tot
}]

print("Model objective: " + model.diffusion_objective)
for i in range(1000):
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
            c3=1000, #1000
            generation=i
        )

        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")

        num_samples = int(conditioning[0]["seconds_total"] * sample_rate)
        output = output[:, :num_samples]

        # Peak normalize, clip, convert to int16, and save to file
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        torchaudio.save(f"/nas/home/fmessina/Experiments/596573_gens/{amg_type}/596573_generation_{i}.wav", output, sample_rate)

        # save the closest list to a file
        with open(f"/nas/home/fmessina/Experiments/596573_gens/graphs/closest_list/closest_{amg_type}.txt", "a") as f:
            f.write("%s\n" % closest_id)