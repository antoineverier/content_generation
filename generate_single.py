import os
import random
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "test_dir": "AerialImageDataset/patches/test",
    "checkpoint_path": "controlnet_xs_aerial_1/checkpoint-epoch-3",
    "base_model_id": "runwayml/stable-diffusion-v1-5",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def main():
    # 1. Select a random test patch
    img_dir = os.path.join(CONFIG["test_dir"], "images")
    mask_dir = os.path.join(CONFIG["test_dir"], "masks")
    all_patches = os.listdir(img_dir)
    
    random_patch_name = random.choice(all_patches)
    city_name = random_patch_name.split('_')[0].rstrip('0123456789')
    prompt = f"Aerial satellite view of {city_name}."
    
    print(f"📍 Selected Patch: {random_patch_name} | Prompt: '{prompt}'")
    
    # Load Real Image and Mask
    real_img = Image.open(os.path.join(img_dir, random_patch_name)).convert("RGB")
    mask_img = Image.open(os.path.join(mask_dir, random_patch_name)).convert("RGB")
    
    # 2. Load Models (Optimal Epoch 3)
    print(f"⚙️ Loading ControlNet from {CONFIG['checkpoint_path']}...")
    controlnet = ControlNetModel.from_pretrained(
        CONFIG["checkpoint_path"], 
        torch_dtype=torch.float16
    ).to(CONFIG["device"])
    
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        CONFIG["base_model_id"],
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(CONFIG["device"])
    pipeline.set_progress_bar_config(disable=True)
    
    # 3. Generate Image
    print("🚀 Generating synthetic aerial patch...")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        generated_img = pipeline(
            prompt=prompt,
            image=mask_img,
            num_inference_steps=20
        ).images[0]
        
    # 4. Save Visualization (Mask | Generated | Real)
    w, h = mask_img.size
    combined_img = Image.new('RGB', (w * 3, h))
    
    # Paste side-by-side
    combined_img.paste(mask_img, (0, 0))
    combined_img.paste(generated_img, (w, 0))
    combined_img.paste(real_img, (w * 2, 0))

    output_filename = f"qualitative_result_{city_name}.png"
    combined_img.save(output_filename)
    print(f"✅ Success! Comparison saved to {output_filename}")

if __name__ == "__main__":
    main()