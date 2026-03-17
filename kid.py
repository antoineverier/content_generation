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
from torchmetrics.image.kid import KernelInceptionDistance

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "patch_dir": "AerialImageDataset/patches", 
    "checkpoints_dir": "controlnet_xs_aerial_1", 
    "base_model_id": "runwayml/stable-diffusion-v1-5",
    "batch_size": 8,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "mixed_precision": "fp16",
    "results_file": "evaluation_results_kid_only.json" 
}

# ==========================================
# 2. DATASET (Test Split Only)
# ==========================================
class AerialTestDataset(Dataset):
    def __init__(self):
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
            "prompt": prompt
        }

# ==========================================
# 3. EVALUATION LOGIC
# ==========================================
def evaluate_epoch(pipeline, test_loader, device, epoch_name):
    print(f"\n--- Evaluating {epoch_name} ---")
    
    # Initialize KID with a subset_size of 100 for your 960-image dataset
    global_kid = KernelInceptionDistance(subset_size=100).to(device)
    
    for batch in tqdm(test_loader, desc="Generating & Scoring"):
        masks = batch["conditioning_pixel_values"].to(device)
        real_images = batch["pixel_values"].to(device)
        prompts = batch["prompt"]
        
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(CONFIG["mixed_precision"] == "fp16")):
            generated_images = pipeline(
                prompt=prompts,
                image=masks,
                num_inference_steps=20,
                output_type="pt"
            ).images
        
        # KID expects uint8 [0, 255]
        real_images_uint8 = ((real_images / 2 + 0.5) * 255).byte()
        gen_images_uint8 = (generated_images * 255).byte()
        
        global_kid.update(real_images_uint8, real=True)
        global_kid.update(gen_images_uint8, real=False)
        
    # Extract the mean from the (mean, std) tuple
    kid_mean, kid_std = global_kid.compute()
    final_kid = kid_mean.item()
    
    print(f"✅ {epoch_name} Results -> KID (mean): {final_kid:.4f}")
    
    return final_kid

# ==========================================
# 4. MAIN LOOP
# ==========================================
def main():
    checkpoint_dirs = glob.glob(os.path.join(CONFIG["checkpoints_dir"], "checkpoint-epoch-*"))
    
    if not checkpoint_dirs:
        print(f"No checkpoints found in {CONFIG['checkpoints_dir']}!")
        return

    def extract_epoch_num(path):
        match = re.search(r'epoch-(\d+)', path)
        return int(match.group(1)) if match else -1
    
    checkpoint_dirs.sort(key=extract_epoch_num)

    test_dataset = AerialTestDataset()
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    print("Loading base models (VAE, UNet, Text Encoder)...")
    tokenizer = CLIPTokenizer.from_pretrained(CONFIG["base_model_id"], subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(CONFIG["base_model_id"], subfolder="text_encoder", torch_dtype=torch.float16).to(CONFIG["device"])
    vae = AutoencoderKL.from_pretrained(CONFIG["base_model_id"], subfolder="vae", torch_dtype=torch.float16).to(CONFIG["device"])
    unet = UNet2DConditionModel.from_pretrained(CONFIG["base_model_id"], subfolder="unet", torch_dtype=torch.float16).to(CONFIG["device"])
    
    evaluation_history = []
    
    for ckpt_path in checkpoint_dirs:
        epoch_name = os.path.basename(ckpt_path)
        
        controlnet = ControlNetModel.from_pretrained(ckpt_path, torch_dtype=torch.float16).to(CONFIG["device"])
        controlnet.eval()
        
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
        
        with torch.no_grad():
            kid = evaluate_epoch(pipeline, test_loader, CONFIG["device"], epoch_name)
            
        evaluation_history.append({
            "epoch": extract_epoch_num(ckpt_path),
            "checkpoint": epoch_name,
            "kid": kid
        })
        
        with open(CONFIG["results_file"], "w") as f:
            json.dump(evaluation_history, f, indent=4)
            
        del pipeline
        del controlnet
        torch.cuda.empty_cache()
        
    print(f"\n🎉 All epochs evaluated! Results saved to {CONFIG['results_file']}")

if __name__ == "__main__":
    main()