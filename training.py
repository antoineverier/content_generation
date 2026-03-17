import os
import glob
import torch
import numpy as np
import cv2
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel
)
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "dataset_dir": "AerialImageDataset/train",
    "patch_dir": "AerialImageDataset/patches", # Where extracted patches will be saved
    "patch_size": 512,
    "stride": 512, # Set smaller than patch_size for overlapping patches
    "test_split_ratio": 0.1, # 10% of images from each city go to test set
    "base_model_id": "runwayml/stable-diffusion-v1-5",
    "batch_size": 8,
    "learning_rate": 1e-5,
    "num_epochs": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "mixed_precision": "fp16",
    "output_dir": "controlnet_xs_aerial_1",
    "seed": 42,
    "debug_mode": False,       # Set to True to use only 10 images for testing
    "debug_samples": 10,      # Number of images to use in debug mode
    "save_every_n_epochs": 1  # How often to save a checkpoint
}

# ==========================================
# 2. DATA PREPARATION (Patching)
# ==========================================
def extract_patches(img_path, mask_path, save_dir, split, city):
    """Slices 5000x5000 images into 512x512 patches and saves them."""
    img = cv2.imread(img_path)[..., ::-1] # BGR to RGB
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    filename = os.path.basename(img_path).split('.')[0]
    h, w, _ = img.shape
    p = CONFIG["patch_size"]
    s = CONFIG["stride"]
    
    os.makedirs(os.path.join(save_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, split, "masks"), exist_ok=True)
    
    patch_idx = 0
    for y in range(0, h - p + 1, s):
        for x in range(0, w - p + 1, s):
            img_patch = img[y:y+p, x:x+p]
            mask_patch = mask[y:y+p, x:x+p]
            
            # Save patches
            Image.fromarray(img_patch).save(os.path.join(save_dir, split, "images", f"{filename}_{patch_idx}.png"))
            Image.fromarray(mask_patch).save(os.path.join(save_dir, split, "masks", f"{filename}_{patch_idx}.png"))
            patch_idx += 1

def prepare_dataset():
    """Organizes the Inria dataset into train/test patches per city."""
    if os.path.exists(CONFIG["patch_dir"]):
        print("Patches already exist. Skipping extraction.")
        return

    print("Extracting patches from 5000x5000 images...")
    img_paths = sorted(glob.glob(os.path.join(CONFIG["dataset_dir"], "images", "*.tif")))
    
    # Group by city (e.g., 'austin', 'chicago')
    cities = {}
    for p in img_paths:
        city_name = os.path.basename(p).rstrip('0123456789.tif')
        if city_name not in cities:
            cities[city_name] = []
        cities[city_name].append(p)
        
    for city, paths in cities.items():
        # Split train/test per city
        test_count = max(1, int(len(paths) * CONFIG["test_split_ratio"]))
        test_paths = paths[-test_count:]
        train_paths = paths[:-test_count]
        
        for p in tqdm(train_paths, desc=f"Patching {city} (Train)"):
            mask_p = p.replace("/images/", "/gt/")
            extract_patches(p, mask_p, CONFIG["patch_dir"], "train", city)
            
        for p in tqdm(test_paths, desc=f"Patching {city} (Test)"):
            mask_p = p.replace("/images/", "/gt/")
            extract_patches(p, mask_p, CONFIG["patch_dir"], "test", city)

# ==========================================
# 3. PYTORCH DATASET
# ==========================================
# ==========================================
# 3. PYTORCH DATASET
# ==========================================
class AerialDataset(Dataset):
    def __init__(self, split="train"):
        self.img_dir = os.path.join(CONFIG["patch_dir"], split, "images")
        self.mask_dir = os.path.join(CONFIG["patch_dir"], split, "masks")
        self.img_names = os.listdir(self.img_dir)
        
        # --- NEW: DEBUG MODE SLICING ---
        if CONFIG["debug_mode"]:
            print(f"DEBUG MODE ACTIVE: Slicing {split} dataset to {CONFIG['debug_samples']} images.")
            self.img_names = self.img_names[:CONFIG["debug_samples"]]
        
        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])
        self.mask_transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        city_name = img_name.split('_')[0].rstrip('0123456789')
        prompt = f"Aerial satellite view of {city_name}."
        
        img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, img_name)).convert("RGB")
        
        return {
            "pixel_values": self.image_transforms(img),
            "conditioning_pixel_values": self.mask_transforms(mask),
            "prompt": prompt,
            "city": city_name 
        }

# ==========================================
# 4. TRAINING & EVALUATION FUNCTIONS
# ==========================================
def evaluate_model(pipeline, test_loader, device):
    print("Evaluating model...")
    # Initialize overall metrics
    global_fid = FrechetInceptionDistance(feature=64).to(device)
    global_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    
    # Initialize dictionary to hold per-city metrics dynamically
    city_lpips_metrics = {}
    
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    for batch in tqdm(test_loader, desc="Calculating Metrics"):
        masks = batch["conditioning_pixel_values"].to(device)
        real_images = batch["pixel_values"].to(device)
        prompts = batch["prompt"]
        cities = batch["city"] # Extract cities from the batch
        
        # Generate images from masks
        generated_images = pipeline(
            prompt=prompts,
            image=masks,
            num_inference_steps=20,
            output_type="pt"
        ).images
        
        # Global FID expects uint8 [0, 255]
        real_images_uint8 = ((real_images / 2 + 0.5) * 255).byte()
        gen_images_uint8 = (generated_images * 255).byte()
        global_fid.update(real_images_uint8, real=True)
        global_fid.update(gen_images_uint8, real=False)
        
        # LPIPS expects float [-1, 1]
        gen_images_norm = (generated_images * 2) - 1
        global_lpips.update(gen_images_norm, real_images)
        
        # --- NEW: Per-City LPIPS update ---
        for i in range(len(cities)):
            city = cities[i]
            # If we haven't seen this city yet, create a new LPIPS tracker for it
            if city not in city_lpips_metrics:
                city_lpips_metrics[city] = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
            
            # Slice the tensors to grab just the individual image [1, 3, 512, 512] and update
            city_lpips_metrics[city].update(gen_images_norm[i:i+1], real_images[i:i+1])
        
    final_fid = global_fid.compute().item()
    final_lpips = global_lpips.compute().item()
    
    # --- NEW: Print a clean report for your terminal ---
    print("\n" + "="*40)
    print("🏆 FINAL EVALUATION REPORT")
    print("="*40)
    print(f"Overall FID:   {final_fid:.2f}")
    print(f"Overall LPIPS: {final_lpips:.4f}")
    print("-" * 40)
    print("📍 Per-City LPIPS Breakdown:")
    for city, metric in city_lpips_metrics.items():
        print(f" - {city.capitalize()}: {metric.compute().item():.4f}")
    print("="*40 + "\n")
    
    return final_fid, final_lpips, city_lpips_metrics

def main():
    torch.manual_seed(CONFIG["seed"])
    prepare_dataset()
    
    train_dataset = AerialDataset(split="train")
    test_dataset = AerialDataset(split="test")
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    tokenizer = CLIPTokenizer.from_pretrained(CONFIG["base_model_id"], subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(CONFIG["base_model_id"], subfolder="text_encoder").to(CONFIG["device"])
    vae = AutoencoderKL.from_pretrained(CONFIG["base_model_id"], subfolder="vae").to(CONFIG["device"])
    unet = UNet2DConditionModel.from_pretrained(CONFIG["base_model_id"], subfolder="unet").to(CONFIG["device"])
    noise_scheduler = DDPMScheduler.from_pretrained(CONFIG["base_model_id"], subfolder="scheduler")
    
    controlnet = ControlNetModel.from_unet(unet).to(CONFIG["device"])
    
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()
    
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=CONFIG["learning_rate"])
    
    scaler = torch.cuda.amp.GradScaler(enabled=(CONFIG["mixed_precision"] == "fp16"))

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    loss_history = []
    
    print("Starting Training...")
    for epoch in range(CONFIG["num_epochs"]):
        total_loss = 0.0 # Track total loss for the epoch
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            optimizer.zero_grad()
            
            images = batch["pixel_values"].to(CONFIG["device"])
            masks = batch["conditioning_pixel_values"].to(CONFIG["device"])
            text_inputs = tokenizer(batch["prompt"], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(CONFIG["device"])
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(CONFIG["mixed_precision"] == "fp16")):
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
                    encoder_hidden_states = text_encoder(text_inputs.input_ids)[0]
                
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_cond=masks, return_dict=False
                )
                
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample
                ).sample
                
                loss = F.mse_loss(noise_pred, noise)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            
        # --- NEW: CALCULATE AVERAGE LOSS & SAVE CHECKPOINTS ---
        avg_epoch_loss = total_loss / len(train_loader)
        loss_history.append({"epoch": epoch + 1, "loss": avg_epoch_loss})
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.6f}")
        
        # Save a checkpoint every N epochs
        if (epoch + 1) % CONFIG["save_every_n_epochs"] == 0:
            ckpt_path = os.path.join(CONFIG["output_dir"], f"checkpoint-epoch-{epoch+1}")
            controlnet.save_pretrained(ckpt_path)
            print(f"Checkpoint saved to: {ckpt_path}")
            
    # Save final model and loss history
    controlnet.save_pretrained(CONFIG["output_dir"])
    
    with open(os.path.join(CONFIG["output_dir"], "training_metrics.json"), "w") as f:
        json.dump(loss_history, f, indent=4)
    print(f"Training metrics saved to {CONFIG['output_dir']}/training_metrics.json")
    
    controlnet = controlnet.to(torch.float16)
    # Evaluate
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        CONFIG["base_model_id"], controlnet=controlnet, torch_dtype=torch.float16
    )
    evaluate_model(pipeline, test_loader, CONFIG["device"])

if __name__ == "__main__":
    main()