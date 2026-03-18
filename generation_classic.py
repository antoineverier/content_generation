import os
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "base_model_id": "runwayml/stable-diffusion-v1-5",
    "controlnet_path": "controlnet_xs_aerial_1/checkpoint-epoch-3", 
    "input_mask_path": "AerialImageDataset/train/gt/austin1.tif",  
    "output_image_path": "austin1_generated_no_overlap.png",
    "prompt": "Aerial satellite view of austin.",
    "patch_size": 512,
    "device": "cuda",
    "num_inference_steps": 20
}

def main():
    print("Loading models into memory...")
    controlnet = ControlNetModel.from_pretrained(
        CONFIG["controlnet_path"], torch_dtype=torch.float16
    )
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        CONFIG["base_model_id"], 
        controlnet=controlnet, 
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(CONFIG["device"])
    
    pipe.set_progress_bar_config(disable=True) 

    print(f"Loading full mask from {CONFIG['input_mask_path']}...")
    full_control_mask = Image.open(CONFIG["input_mask_path"]).convert("RGB")
    width, height = full_control_mask.size
    
    # ==========================================
    # 2. CANVAS INITIALIZATION & STEP LOGIC
    # ==========================================
    canvas = Image.new("RGB", (width, height), (0, 0, 0))

    # Calculate steps using only patch_size
    def get_steps(max_dim, p_size):
        steps = list(range(0, max_dim - p_size + 1, p_size))
        if not steps or steps[-1] != max_dim - p_size:
            steps.append(max_dim - p_size) # Force the window to the exact edge
        return steps

    y_steps = get_steps(height, CONFIG["patch_size"])
    x_steps = get_steps(width, CONFIG["patch_size"])
    
    # ==========================================
    # 3. GENERATION LOOP (No overlap)
    # ==========================================
    for y in tqdm(y_steps, desc="Rows Processed"):
        for x in tqdm(x_steps, desc="Columns Processed", leave=False):
            
            control_patch = full_control_mask.crop((x, y, x + CONFIG["patch_size"], y + CONFIG["patch_size"]))
            
            with torch.autocast("cuda", dtype=torch.float16):
                result_patch = pipe(
                    prompt=CONFIG["prompt"],
                    image=control_patch,
                    num_inference_steps=CONFIG["num_inference_steps"],
                ).images[0]
            
            canvas.paste(result_patch, (x, y))

    print(f"Generation complete! Saving to {CONFIG['output_image_path']}...")
    canvas.save(CONFIG["output_image_path"])

if __name__ == "__main__":
    main()