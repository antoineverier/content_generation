import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from diffusers import StableDiffusionImg2ImgPipeline
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "patch_dir": "AerialImageDataset/patches", 
    "base_model_id": "runwayml/stable-diffusion-v1-5",
    "batch_size": 8,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "mixed_precision": "fp16",
    "results_file": "baseline_img2img_results.json",
    "img2img_strength": 0.75 # Adjust this! 
}

# ==========================================
# 2. DATASET
# ==========================================
class AerialTestDataset(Dataset):
    def __init__(self):
        self.img_dir = os.path.join(CONFIG["patch_dir"], "test", "images")
        self.mask_dir = os.path.join(CONFIG["patch_dir"], "test", "masks")
        self.img_names = os.listdir(self.img_dir)
        
        # Real images for metrics need [-1, 1] normalization
        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])
        
        # Masks for Img2Img usually work best in [0, 1] range in diffusers
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
# 3. EVALUATION LOGIC
# ==========================================
def evaluate_baseline_img2img(pipeline, test_loader, device):
    print(f"\n--- Evaluating Img2Img Baseline (Strength: {CONFIG['img2img_strength']}) ---")
    
    # Initialize KID (subset_size=50 is standard for smaller datasets in torchmetrics)
    global_kid = KernelInceptionDistance(subset_size=50).to(device)
    global_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    city_lpips_metrics = {}
    
    for batch in tqdm(test_loader, desc="Generating & Scoring"):
        real_images = batch["pixel_values"].to(device)
        masks = batch["conditioning_pixel_values"].to(device)
        prompts = batch["prompt"]
        cities = batch["city"] 
        
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(CONFIG["mixed_precision"] == "fp16")):
            generated_images = pipeline(
                prompt=prompts,
                image=masks, 
                strength=CONFIG["img2img_strength"],
                num_inference_steps=20,
                output_type="pt"
            ).images
        
        # KID expects uint8 [0, 255]
        real_images_uint8 = ((real_images / 2 + 0.5) * 255).byte()
        gen_images_uint8 = (generated_images * 255).byte()
        global_kid.update(real_images_uint8, real=True)
        global_kid.update(gen_images_uint8, real=False)
        
        # LPIPS expects float [-1, 1]
        gen_images_norm = (generated_images * 2) - 1
        global_lpips.update(gen_images_norm, real_images)
        
        # Per-City LPIPS update
        for i in range(len(cities)):
            city = cities[i]
            if city not in city_lpips_metrics:
                city_lpips_metrics[city] = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
            city_lpips_metrics[city].update(gen_images_norm[i:i+1], real_images[i:i+1])
        
    # Compute returns a tuple of (mean, std) for KID
    kid_mean, kid_std = global_kid.compute()
    final_kid_mean = kid_mean.item()
    final_kid_std = kid_std.item()
    
    final_lpips = global_lpips.compute().item()
    computed_city_lpips = {city: metric.compute().item() for city, metric in city_lpips_metrics.items()}
    
    print(f"✅ Baseline Results -> KID: {final_kid_mean:.4f} ± {final_kid_std:.4f} | Overall LPIPS: {final_lpips:.4f}")
    
    return final_kid_mean, final_kid_std, final_lpips, computed_city_lpips

# ==========================================
# 4. MAIN SCRIPT
# ==========================================
def main():
    test_dataset = AerialTestDataset()
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    print(f"Loading Img2Img base model ({CONFIG['base_model_id']})...")
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        CONFIG["base_model_id"],
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(CONFIG["device"])
    pipeline.set_progress_bar_config(disable=True)
    
    with torch.no_grad():
        kid_mean, kid_std, lpips, city_lpips = evaluate_baseline_img2img(pipeline, test_loader, CONFIG["device"])
        
    baseline_results = {
        "model": "baseline_img2img_sd_v1.5",
        "strength": CONFIG["img2img_strength"],
        "kid_mean": kid_mean,
        "kid_std": kid_std,
        "lpips": lpips,
        "city_lpips": city_lpips
    }
    
    with open(CONFIG["results_file"], "w") as f:
        json.dump(baseline_results, f, indent=4)
        
    print(f"\n🎉 Img2Img Baseline evaluated! Results saved to {CONFIG['results_file']}")

if __name__ == "__main__":
    main()