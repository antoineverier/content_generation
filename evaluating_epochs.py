import os
import glob
import json
import torch
import re
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
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "patch_dir": "AerialImageDataset/patches", # Path to your existing patches
    "checkpoints_dir": "controlnet_xs_aerial_1", # Where your epoch folders are saved
    "base_model_id": "runwayml/stable-diffusion-v1-5",
    "batch_size": 8,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "mixed_precision": "fp16",
    "results_file": "evaluation_results.json"
}

# ==========================================
# 2. DATASET (Test Split Only)
# ==========================================
class AerialTestDataset(Dataset):
    def __init__(self):
        # We only need the test split for inference evaluation
        self.img_dir = os.path.join(CONFIG["patch_dir"], "test", "images")
        self.mask_dir = os.path.join(CONFIG["patch_dir"], "test", "masks")
        self.img_names = os.listdir(self.img_dir)
        
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
# 3. EVALUATION LOGIC
# ==========================================
def evaluate_epoch(pipeline, test_loader, device, epoch_name):
    print(f"\n--- Evaluating {epoch_name} ---")
    
    global_fid = FrechetInceptionDistance(feature=64).to(device)
    global_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    city_lpips_metrics = {}
    
    for batch in tqdm(test_loader, desc="Generating & Scoring"):
        masks = batch["conditioning_pixel_values"].to(device)
        real_images = batch["pixel_values"].to(device)
        prompts = batch["prompt"]
        cities = batch["city"] 
        
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(CONFIG["mixed_precision"] == "fp16")):
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
        
        # Per-City LPIPS update
        for i in range(len(cities)):
            city = cities[i]
            if city not in city_lpips_metrics:
                city_lpips_metrics[city] = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
            city_lpips_metrics[city].update(gen_images_norm[i:i+1], real_images[i:i+1])
        
    final_fid = global_fid.compute().item()
    final_lpips = global_lpips.compute().item()
    computed_city_lpips = {city: metric.compute().item() for city, metric in city_lpips_metrics.items()}
    
    print(f"✅ {epoch_name} Results -> FID: {final_fid:.2f} | Overall LPIPS: {final_lpips:.4f}")
    
    return final_fid, final_lpips, computed_city_lpips

# ==========================================
# 4. MAIN LOOP
# ==========================================
def main():
    # 1. Find and sort checkpoints
    # Looks for folders like "checkpoint-epoch-1", "checkpoint-epoch-2"
    checkpoint_dirs = glob.glob(os.path.join(CONFIG["checkpoints_dir"], "checkpoint-epoch-*"))
    
    if not checkpoint_dirs:
        print(f"No checkpoints found in {CONFIG['checkpoints_dir']}!")
        return

    # Sort checkpoints numerically rather than alphabetically (e.g., 2 before 10)
    def extract_epoch_num(path):
        match = re.search(r'epoch-(\d+)', path)
        return int(match.group(1)) if match else -1
    
    checkpoint_dirs.sort(key=extract_epoch_num)

    # 2. Setup Data
    test_dataset = AerialTestDataset()
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    # 3. Load Base Models ONCE to save VRAM and time
    print("Loading base models (VAE, UNet, Text Encoder)...")
    tokenizer = CLIPTokenizer.from_pretrained(CONFIG["base_model_id"], subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(CONFIG["base_model_id"], subfolder="text_encoder", torch_dtype=torch.float16).to(CONFIG["device"])
    vae = AutoencoderKL.from_pretrained(CONFIG["base_model_id"], subfolder="vae", torch_dtype=torch.float16).to(CONFIG["device"])
    unet = UNet2DConditionModel.from_pretrained(CONFIG["base_model_id"], subfolder="unet", torch_dtype=torch.float16).to(CONFIG["device"])
    
    evaluation_history = []
    
    # 4. Loop through each checkpoint
    for ckpt_path in checkpoint_dirs:
        epoch_name = os.path.basename(ckpt_path)
        
        # Load the specific ControlNet weights for this epoch
        controlnet = ControlNetModel.from_pretrained(ckpt_path, torch_dtype=torch.float16).to(CONFIG["device"])
        controlnet.eval()
        
        # Create pipeline
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
        
        # Run evaluation
        with torch.no_grad():
            fid, lpips, city_lpips = evaluate_epoch(pipeline, test_loader, CONFIG["device"], epoch_name)
            
        evaluation_history.append({
            "epoch": extract_epoch_num(ckpt_path),
            "checkpoint": epoch_name,
            "fid": fid,
            "lpips": lpips,
            "city_lpips": city_lpips
        })
        
        # Save results iteratively in case the script is interrupted
        with open(CONFIG["results_file"], "w") as f:
            json.dump(evaluation_history, f, indent=4)
            
        # VERY IMPORTANT: Free up VRAM before loading the next ControlNet
        del pipeline
        del controlnet
        torch.cuda.empty_cache()
        
    print(f"\n🎉 All epochs evaluated! Results saved to {CONFIG['results_file']}")

if __name__ == "__main__":
    main()