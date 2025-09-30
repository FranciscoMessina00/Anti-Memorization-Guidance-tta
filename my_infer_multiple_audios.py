import torch
import os
import json
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torchaudio
import math
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.my_generation import my_generate_diffusion_cond

# ---------------- Configuration ---------------- #
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda:7"
amg_type = "despec"

# Optional filtering by cluster index (outer JSON keys)
# Set to None to include entire range
start_id = 46  # e.g., 10 (cluster index)
end_id = None    # e.g., 25 (cluster index)

# Inputs for representatives and embeddings
reps_json_path = Path('/nas/home/fmessina/split_embeddings/cluster_outputs_2/cluster_representatives.json')
embeddings_json_path = Path('/nas/home/fmessina/stable-audio-tools/embeddings_new.json')

# Generation parameters
steps = 100
cfg_scale = 7
generations_per_id = 100  # how many variations to generate per representative id

# Duration handling
default_seconds = 10.0
min_seconds = 0.001
max_seconds = 47.0
# ------------------------------------------------ #

# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)

print("Model objective: " + model.diffusion_objective)

# Load representatives and embeddings
with reps_json_path.open('r', encoding='utf-8') as f:
    reps = json.load(f)  # {cluster_id: {representative_id, ...}}
with embeddings_json_path.open('r', encoding='utf-8') as f:
    embeddings = json.load(f)  # {id: {embedding, conditioning: {...}}}

# Compute a dynamic minimum duration to ensure sufficient latent length for decoder kernels
latent_ratio = getattr(getattr(model, 'pretransform', None), 'downsampling_ratio', 1)
kernel_min = 7  # minimum latent length needed to avoid Conv1d kernel_size issues
min_seconds_required = max(min_seconds, (kernel_min * latent_ratio) / sample_rate)
if min_seconds_required > min_seconds:
    print(f"[INFO] Using dynamic minimum seconds_total = {min_seconds_required:.4f}s (ratio={latent_ratio}, sr={sample_rate})")

# Build sorted list of clusters (by cluster index)
cluster_items = sorted(((int(cid), data) for cid, data in reps.items()), key=lambda x: x[0])

# Optionally filter by cluster index range
if start_id is not None or end_id is not None:
    all_cluster_indices = [cid for cid, _ in cluster_items]
    min_c, max_c = min(all_cluster_indices), max(all_cluster_indices)
    s_val = int(start_id) if start_id is not None else min_c
    e_val = int(end_id) if end_id is not None else max_c
    if s_val > e_val:
        s_val, e_val = e_val, s_val
    cluster_items = [(cid, data) for cid, data in cluster_items if s_val <= cid <= e_val]
    print(f"[INFO] Filtering clusters in [{s_val}, {e_val}] → {len(cluster_items)} clusters")

# Map selected clusters to their representative IDs (strings)
rep_ids = [str(data.get('representative_id')) for _, data in cluster_items if data.get('representative_id') is not None]

print(f"[INFO] Generating for {len(rep_ids)} cluster representatives; {generations_per_id} generations each.")

for rep_idx, fs_id in enumerate(rep_ids):
    meta = embeddings.get(str(fs_id)) or {}
    cond = meta.get('conditioning') or {}
    prompt = cond.get('prompt', '')
    tot_raw = float(cond.get('seconds_total', default_seconds))
    tot = max(min_seconds_required, min(max_seconds, tot_raw))
    if tot != tot_raw:
        print(f"[WARN] seconds_total too short ({tot_raw:.4f}s). Clamped to {tot:.4f}s to satisfy decoder constraints.")

    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": tot,
    }]
    negative_conditioning = [{
        "prompt": "",
        "seconds_start": 0,
        "seconds_total": tot,
    }]

    print(f"\n[ID {rep_idx+1}/{len(rep_ids)}] {fs_id} | seconds_total={tot:.2f} | prompt_len={len(prompt)}")

    for i in range(generations_per_id):
        
        # Ensure output directory exists, and skip saving if the file already exists
        audio_dir = f"/nas/home/fmessina/Experiments/global_experiment/{fs_id}_gens/{amg_type}"
        os.makedirs(audio_dir, exist_ok=True)
        audio_path = os.path.join(audio_dir, f"{fs_id}_generation_{i}.wav")
        if os.path.exists(audio_path):
            print(f"[INFO] Audio already exists, skipping save: {audio_path}")
        else:
            output, closest_id = my_generate_diffusion_cond(
                model,
                steps=steps,
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
                c1=cfg_scale-1,
                c2=0,  # cfg_scale-1
                c3=0,
                generation=i,
            )

            # Rearrange audio batch to a single sequence
            output = rearrange(output, "b d n -> d (b n)")

            num_samples = int(conditioning[0]["seconds_total"] * sample_rate)
            output = output[:, :num_samples]

            # Peak normalize, clip, convert to int16, and save to file
            output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

            torchaudio.save(audio_path, output, sample_rate)
            print(f"[INFO] Saved: {audio_path}")

            # Ensure closest list directory exists, check file, then append
            closest_dir = f"/nas/home/fmessina/Experiments/global_experiment/{fs_id}_gens/graphs/closest_list"
            os.makedirs(closest_dir, exist_ok=True)
            closest_file = os.path.join(closest_dir, f"closest_{amg_type}.txt")
            if not os.path.exists(closest_file):
                # create the file if missing
                open(closest_file, "w").close()
                print(f"[INFO] Created closest list file: {closest_file}")
            with open(closest_file, "a") as f:
                f.write(f"{closest_id}\n")