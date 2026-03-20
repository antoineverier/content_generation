import os
import json
import torch
import glob
import itertools
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    UNet2DConditionModel
)
from transformers import CLIPTextModel, CLIPTokenizer
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "patch_dir": "AerialImageDataset/patches", 
    "checkpoint_path": "controlnet_xs_aerial_1/checkpoint-epoch-3", # Put your best epoch here
    "base_model_id": "runwayml/stable-diffusion-v1-5",
    "batch_size": 8,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "mixed_precision": "fp16",
    "results_file": "cross_city_matrix_results.json"
}

# ==========================================
# 2. DATASET HELPERS
# ==========================================
def get_all_cities(img_dir):
    """Scans the test directory and automatically extracts all unique city names."""
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Directory not found: {img_dir}")
        
    img_names = os.listdir(img_dir)
    cities = set()
    for name in img_names:
        # Extract the city base name just like in the training script
        city_name = name.split('_')[0].rstrip('0123456789')
        cities.add(city_name)
    return sorted(list(cities))

class SingleCityDataset(Dataset):
    """Loads images/masks for a specific city."""
    def __init__(self, city_name, prompt_target_city=None):
        self.img_dir = os.path.join(CONFIG["patch_dir"], "test", "images")
        self.mask_dir = os.path.join(CONFIG["patch_dir"], "test", "masks")
        
        all_files = os.listdir(self.img_dir)
        self.img_names = [f for f in all_files if f.startswith(city_name)]
        
        self.city_name = city_name
        self.prompt_city = prompt_target_city if prompt_target_city else city_name
        
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
        prompt = f"Aerial satellite view of {self.prompt_city}."
        
        img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, img_name)).convert("RGB")
        
        return {
            "pixel_values": self.image_transforms(img),
            "conditioning_pixel_values": self.mask_transforms(mask),
            "prompt": prompt
        }

# ==========================================
# 3. EVALUATION LOGIC
# ==========================================
def evaluate_cross_city(pipeline, source_city, target_city, device):
    print(f"\n--- Swapping: Mask({source_city}) -> Style({target_city}) ---")
    
    source_dataset = SingleCityDataset(source_city, prompt_target_city=target_city)
    source_loader = DataLoader(source_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    target_dataset = SingleCityDataset(target_city)
    target_loader = DataLoader(target_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    # Initialize Metrics
    kid_metric = KernelInceptionDistance(subset_size=min(100, len(target_dataset))).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)

    # Step A: Load real target images into KID
    for batch in target_loader:
        real_target_images = batch["pixel_values"].to(device)
        real_target_uint8 = ((real_target_images / 2 + 0.5) * 255).byte()
        kid_metric.update(real_target_uint8, real=True)

    # Step B: Generate new images and compute metrics
    for batch in tqdm(source_loader, desc=f"{source_city}->{target_city}"):
        masks = batch["conditioning_pixel_values"].to(device)
        real_source_images = batch["pixel_values"].to(device) 
        prompts = batch["prompt"] 
        
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(CONFIG["mixed_precision"] == "fp16")):
            generated_images = pipeline(
                prompt=prompts,
                image=masks,
                num_inference_steps=20,
                output_type="pt"
            ).images
            
        gen_images_uint8 = (generated_images * 255).byte()
        kid_metric.update(gen_images_uint8, real=False)
        
        gen_images_norm = (generated_images * 2) - 1
        lpips_metric.update(gen_images_norm, real_source_images)
        
    kid_mean, kid_std = kid_metric.compute()
    lpips_score = lpips_metric.compute().item()
    
    print(f"✅ Result: {source_city}->{target_city} | KID (vs {target_city}): {kid_mean.item():.4f} | LPIPS (vs {source_city}): {lpips_score:.4f}")
    
    return {
        "source": source_city,
        "target": target_city,
        "kid_mean": kid_mean.item(),
        "kid_std": kid_std.item(),
        "lpips_vs_source": lpips_score
    }

# ==========================================
# 4. MAIN SCRIPT
# ==========================================
def main():
    test_img_dir = os.path.join(CONFIG["patch_dir"], "test", "images")
    
    # Automatically detect cities and create pairs
    cities = get_all_cities(test_img_dir)
    print(f"Detecting cities: {cities}")
    
    # Generate all permutations (A->B, B->A, etc.) excluding self-pairs (A->A)
    swaps = list(itertools.permutations(cities, 2))
    print(f"Total cross-city pairs to evaluate: {len(swaps)}")
    
    print("Loading Base Models...")
    tokenizer = CLIPTokenizer.from_pretrained(CONFIG["base_model_id"], subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(CONFIG["base_model_id"], subfolder="text_encoder", torch_dtype=torch.float16).to(CONFIG["device"])
    vae = AutoencoderKL.from_pretrained(CONFIG["base_model_id"], subfolder="vae", torch_dtype=torch.float16).to(CONFIG["device"])
    unet = UNet2DConditionModel.from_pretrained(CONFIG["base_model_id"], subfolder="unet", torch_dtype=torch.float16).to(CONFIG["device"])
    
    print(f"Loading ControlNet from {CONFIG['checkpoint_path']}...")
    controlnet = ControlNetModel.from_pretrained(CONFIG["checkpoint_path"], torch_dtype=torch.float16).to(CONFIG["device"])
    
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        CONFIG["base_model_id"],
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16
    ).to(CONFIG["device"])
    pipeline.set_progress_bar_config(disable=True)
    
    results = []
    
    # If a previous run crashed, we can load the existing results so we don't start from scratch
    if os.path.exists(CONFIG["results_file"]):
        print(f"Found existing {CONFIG['results_file']}, loading previous progress...")
        with open(CONFIG["results_file"], "r") as f:
            results = json.load(f)
            
    completed_swaps = [(r["source"], r["target"]) for r in results]
    
    with torch.no_grad():
        for source_city, target_city in swaps:
            if (source_city, target_city) in completed_swaps:
                print(f"Skipping {source_city}->{target_city} (Already completed)")
                continue
                
            metrics = evaluate_cross_city(pipeline, source_city, target_city, CONFIG["device"])
            results.append(metrics)
            
            # Save incrementally after EVERY city pair in case of a crash
            with open(CONFIG["results_file"], "w") as f:
                json.dump(results, f, indent=4)
                
    print(f"\n🎉 Full Matrix Evaluation Complete! All results saved to {CONFIG['results_file']}")

if __name__ == "__main__":
    main()