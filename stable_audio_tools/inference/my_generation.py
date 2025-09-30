import numpy as np
from numpy import dot
from numpy.linalg import norm
import torch 
import typing as tp
import k_diffusion as K
#import CLAP.src.laion_clap as laion_clap
from sklearn.metrics.pairwise import cosine_similarity

from .utils import prepare_audio
from .sampling import sample_rf
from .my_sampling import my_sample_k, make_cond_model_fn

import os, sys, json
import matplotlib.pyplot as plt

# 0) before you start sampling, create an empty list:
s1_list = []
closest_id_final = None

# 1) Compute the absolute path to CLAP/src
#    Adjust the number of '..' to go from 
#    stable-audio-tools/.../inference back to /nas/home/fmessina
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..','..','..'))  
LOCAL_CLAP = os.path.join(ROOT, 'CLAP', 'src')

# 2) Insert it at the front of sys.path so it wins over the env package
sys.path.insert(0, LOCAL_CLAP)

# 3) Now import exactly your local code:
import laion_clap 


def my_generate_diffusion_cond(
        model,
        steps: int = 250,
        cfg_scale=6,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: dict = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        sample_rate: int = 48000,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        return_latents = False,
        c1=5.0,
        c2=5.0,
        c3=5.0,
        generation=0,
        **sampler_kwargs
        ) -> torch.Tensor: 
    """
    Generate audio from a prompt using a diffusion model.
    
    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale 
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        sample_rate: The sample rate of the audio to generate (Deprecated, now pulled from the model directly)
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        init_noise_level: The noise level to use when generating from an initial audio sample.
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.    
    """
    global s1_list
    # The length of the output in audio samples 
    s1_list = []
    audio_sample_size = sample_size
    effective_audio_length = int(conditioning[0]["seconds_total"] * sample_rate)

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
        print("Downsampling ratio: ", model.pretransform.downsampling_ratio)
        
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    print(seed)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
            
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    else:
        negative_conditioning_tensors = {}

    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        init_audio = init_audio.repeat(batch_size, 1, 1)

        sampler_kwargs["sigma_max"] = init_noise_level        

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}
    # Now the generative AI part:
    # k-diffusion denoising process go!

    diff_objective = model.diffusion_objective

    if diff_objective == "v":     
        # Clap init
        CLAP = laion_clap.CLAP_Module(enable_fusion=False, device=device)
        CLAP.load_ckpt()
        print("CLAP loaded")
        # CLAP tokenizer
        e_prompt = CLAP.get_text_embedding([conditioning[0]["prompt"]])
        # e_prompt = CLAP.get_text_embedding(["A bird chirping"]) # Example of closet embedding really far from the prompt
        e_prompt = e_prompt[0]
        # e_prompt = torch.from_numpy(e_prompt).to(device)
        # e_prompt = e_prompt / e_prompt.norm(dim=-1, keepdim=True)
        ####
        base_denoiser = K.external.VDenoiser(model.model)
        despec_fn     = make_despec_fn(base_denoiser, e_prompt, s0=cfg_scale, c1=c1, c2=c2, c3=c3, CLAP=CLAP, device=device, length=effective_audio_length, model=model)
        guided        = make_cond_model_fn(base_denoiser, despec_fn, conditioning_inputs, negative_conditioning_tensors)
        ####

        sampled = my_sample_k(guided, noise, init_audio, steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=False, device=device)
        # s_eff = [cfg_scale - v for v in s1_list]

        # plt.figure()
        # plt.plot(s_eff, linewidth=2)      # one distinct plot; no color specifier
        # plt.xlabel("Sampling step")
        # # plt.ylabel("Effective guidance scale\n$(s_0 - s_1)$")
        # plt.ylabel("Similarity")
        # plt.title("Similarity over sampling steps")
        # plt.tight_layout()
        # plt.savefig(f"AMG_30Gens/Graphs/505_similarityscale_gen_{generation}.png", dpi=300)
        # plt.show()
    elif diff_objective == "rectified_flow":

        if "sigma_min" in sampler_kwargs:
            del sampler_kwargs["sigma_min"]

        if "rho" in sampler_kwargs:
            del sampler_kwargs["rho"]

        sampled = sample_rf(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, dist_shift=model.dist_shift, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)

    # v-diffusion: 
    #sampled = sample(model.model, noise, steps, 0, **conditioning_tensors, embedding_scale=cfg_scale)
    del noise
    del conditioning_tensors
    del conditioning_inputs
    torch.cuda.empty_cache()
    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    # if model.pretransform is not None and not return_latents:
    #     #cast sampled latents to pretransform dtype
    #     sampled = sampled.to(next(model.pretransform.parameters()).dtype)
    #     sampled = model.pretransform.decode(sampled)
    if model.pretransform is not None and not return_latents:
        
        # Store original device of the pretransform model
        # pretransform_original_device = next(model.pretransform.parameters()).device
        
        # Move pretransform model to CPU
        # model.pretransform.to('cpu')
        
        # Move sampled tensor to CPU
        sampled_on_cpu = sampled.detach()
        
        # Cast sampled latents to pretransform dtype (this operation will also be on CPU)
        sampled_on_cpu = sampled_on_cpu.to(next(model.pretransform.parameters()).dtype) # dtype is fine, parameters are now on CPU
        
        # Perform decode on CPU
        sampled = model.pretransform.decode(sampled_on_cpu) # Now both model and data are on CPU
        
        # Move pretransform model back to its original device
        # model.pretransform.to(pretransform_original_device)
        
        # 'sampled' is now on CPU. If you need it on the GPU for subsequent steps:
        sampled = sampled.to('cpu') # where 'device' is your target CUDA device
    # Return audio
    return sampled, closest_id_final

# 1) Load once (outside your step loop!), convert embeddings to tensors:
with open('embeddings_new.json','r') as f:
    data = json.load(f)

# make a list of IDs and a single tensor of shape (N, D)
ids         = list(data.keys())
emb_matrix  = torch.stack([
    torch.tensor(data[sound_id]['embedding'], 
                dtype=torch.float32)
    for sound_id in ids
], dim=0).cuda("cuda:0")  # → (N, D)

def make_despec_fn(base_model_fn, e_prompt, s0=7.5, c1=5.0, c2=5.0, c3=5.0, CLAP=None, device="cuda", length=2097152, model=None):
    """Return a cond_fn that applies AMG‐despecification at each step."""
    def despec_cond_fn(x, sigma, denoised,
                       conditioning_inputs, negative_conditioning_inputs, **_):
        global closest_id_final
        x.requires_grad_(True)
        # 1. Unconditional and conditional x0 predictions (VDenoiser outputs x0)
        x0_uncond = denoised  # shape [B,C,L]
        x0_cond   = base_model_fn(x, sigma, **conditioning_inputs)

        # 2. Compute alpha_bar (per batch) and derive epsilon estimates from x and x0
        alpha_bar = 1.0 / (1.0 + sigma.pow(2))  # [B]
        sqrt_ab   = alpha_bar.sqrt()
        sqrt_1mab = (1 - alpha_bar).sqrt().clamp_min(1e-8)

        # Broadcasting helper
        expand = lambda t: t.view(-1, *([1] * (x0_uncond.ndim - 1)))

        # Convert model x0 predictions to epsilon predictions: eps = (x_t - sqrt(alpha_bar)*x0)/sqrt(1-alpha_bar)
        eps_uncond = (x - expand(sqrt_ab) * x0_uncond) / expand(sqrt_1mab)
        eps_cond   = (x - expand(sqrt_ab) * x0_cond)   / expand(sqrt_1mab)

        # 3. Decode (trim to original audio length in latent space) for CLAP similarity using uncond branch
        latent_ratio  = model.pretransform.downsampling_ratio
        latent_length = length // latent_ratio
        x0_trim = x0_cond[:, :, :latent_length]
        x0_trim = model.pretransform.decode(x0_trim)

        audio_batch = x0_trim.mean(dim=1)  # mono
        # Normalize to [-1,1] scale for CLAP (avoid division by zero)
        peak = audio_batch.abs().max().clamp_min(1e-6)
        e_t = CLAP.get_audio_embedding_from_data(x=audio_batch/peak, use_tensor=True)
        e_t = e_t[0].to(device)

        # 4. Nearest neighbour in embedding space
        with torch.no_grad():  # search doesn't need gradients
            dists = torch.linalg.norm(emb_matrix - e_t.unsqueeze(0), dim=1)
            best_idx = torch.argmin(dists).item()
            best_id  = ids[best_idx]
            best_dist= dists[best_idx].item()
        neighbour_cond = data[best_id]['conditioning']
        closest_id_final = best_id
        print(f"Closest ID (Euclidean) is {best_id} dist {best_dist:.4f}")
        audio_embed = torch.tensor(data[best_id]['embedding'], device=device, dtype=e_t.dtype)

        # 5. Conditional prediction for neighbour caption
        conditioning_tensors_N = model.conditioner([neighbour_cond], device)
        conditioning_inputs_N = model.get_conditioning_inputs(conditioning_tensors_N, negative=False)
        x0_cond_N = base_model_fn(x, sigma, **conditioning_inputs_N)
        eps_cond_N = (x - expand(sqrt_ab) * x0_cond_N) / expand(sqrt_1mab)

        # 6. Similarity scalar (dot). (Could normalize embeddings if desired.)
        cos_sim = (e_t * audio_embed).sum(dim=-1)  # scalar
        sim_scalar = cos_sim.sum()
        G_sim = torch.zeros_like(x0_cond_N)
        if c3 > 0:
            # Gradient of similarity w.r.t. x (through x0_uncond -> decode -> embedding model) if graph allows
            try:
                grad_sigma = torch.autograd.grad(sim_scalar, x, retain_graph=True, allow_unused=True)[0]
                if grad_sigma is not None:
                    G_sim = - c3 * torch.sqrt(1 - alpha_bar) * grad_sigma
                    print(f"Norm of sim: {G_sim.norm().item():.4f}")
            except RuntimeError:
                pass  # Fallback: leave G_sim zeros if path not differentiable

        # 7. Dynamic scales s1, s2 based on similarity
        s1 = (c1 * cos_sim).clamp(0, s0 - 1)
        s1_list.append(float(s1.item()))
        s2 = (c2 * cos_sim).clamp(0, s0 - s1.item() - 1)
        # print(f"Similarity: {cos_sim.item():.4f}")

        # 8. Guidance terms computed in epsilon space then mapped back to x0 space.
        #    x0 = (x/sqrt_ab) - (sqrt_1mab/sqrt_ab)*eps  =>  Δx0 = -(sqrt_1mab/sqrt_ab) * Δeps
        delta_eps   = eps_cond   - eps_uncond
        delta_eps_N = eps_cond_N - eps_uncond
        scale_eps2x0 = - (sqrt_1mab / sqrt_ab)  # [B]
        scale_b = expand(scale_eps2x0)
        delta_x0   = scale_b * delta_eps
        delta_x0_N = scale_b * delta_eps_N
        G_cfg   = s0 * delta_x0
        # print(f"Norm of cfg: {G_cfg.norm().item():.4f}")
        G_spe   = -s1 * delta_x0
        # print(f"Norm of spec: {G_spe.norm().item():.4f}")
        G_dedup = -s2 * delta_x0_N
        # print(f"Norm of dedup: {G_dedup.norm().item():.4f}")

        # 9. Parabolic gating based on alpha_bar
        lambda_min, lambda_max = 0.4, 0.5
        lambda_t = lambda_min + (lambda_max - lambda_min) * (alpha_bar ** 2)
        mask = (cos_sim > lambda_t).float().view(-1, *([1] * (G_spe.ndim - 1)))
        print(f"lambda_t: {lambda_t.item():.4f}")
        additional = G_spe + G_dedup + G_sim
        return G_cfg + mask * additional

    return despec_cond_fn
