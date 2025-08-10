import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.my_generation import my_generate_diffusion_cond

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda:5"

# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)

tot = 1.714

# Set up text and timing conditioning
conditioning = [{
    "prompt": '"ATTACK loop 140 bpm-00.wav" till "ATTACK loop 140 bpm-31.wav" are all part of the "ATTACK LOOP 6" sample package and belong together as they are all variations on the same 1 measure 4/4 140 bpm drumloop. The loop has a techno-trance feel. The first four loops (00 till 03) contain some variations of the pure drumloop, where 00 is the most minimal and 03 the fullest. All other variations add other sound effects, some of them being sounds with a certain pitch, mostly C. These loop are suitable for your trance and techno productions. They were created using the Waldorf Attack VSTi within Cubase SX. Mastering (EQ, Stereo Enhancer, Multi-Band expand/compress/limit, dither, fades at start and/or end) done within Wavelab.',
    # "prompt": 'A drum loop with a lot of ride, 120 bpm',
    "seconds_start": 0, 
    "seconds_total": tot
}]
negative_conditioning = [{
    "prompt": "",
    "seconds_start": 0,
    "seconds_total": tot
}]

print("Model objective: " + model.diffusion_objective)
for i in range(0, 1):
# Generate stereo audio
    # with torch.no_grad():
        output, closest_id = my_generate_diffusion_cond(
            model,
            steps=100,
            cfg_scale=7,
            conditioning=conditioning,
            negative_conditioning=negative_conditioning,
            sample_size=sample_size,
            sample_rate=sample_rate,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="my-dpmpp-3m-sde",
            device=device,
            # seed = 2556487306,
            c1=0, #10 # 40 when alone
            c2=0, #15 # 40 when alone
            c3=10, #20 # 40 when alone # 80 actually reduces
            generation=i
        )

        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")

        num_samples = int(conditioning[0]["seconds_total"] * sample_rate)
        output = output[:, :num_samples]

        # Peak normalize, clip, convert to int16, and save to file
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        torchaudio.save(f"/nas/home/fmessina/Experiments/5121_gens/tests/5121_generation_{i}.wav", output, sample_rate)

        # save the closest list to a file
        # with open("Closest_list/closest_list_5121_NO_AMG.txt", "a") as f:
        #     f.write("%s\n" % closest_id)