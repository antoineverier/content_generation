import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    "base_model_id": "runwayml/stable-diffusion-v1-5",
    "controlnet_path": "controlnet_xs_aerial_1/checkpoint-epoch-3", # Update to your best epoch
    "input_mask_path": "AerialImageDataset/train/gt/austin1.tif",  # The 5000x5000 mask
    "output_image_path": "austin1_generated_seamless.png",
    "prompt": "Aerial satellite view of austin.",
    "patch_size": 512,
    "stride": 256, # 256 gives a 50% overlap for perfect continuity
    "device": "cuda",
    "num_inference_steps": 20
}

def main():
    print("Loading models into memory...")
    # Load your trained ControlNet
    controlnet = ControlNetModel.from_pretrained(
        CONFIG["controlnet_path"], torch_dtype=torch.float16
    )
    
    # Load the Inpaint Pipeline (this allows conditional overlap generation)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        CONFIG["base_model_id"], 
        controlnet=controlnet, 
        torch_dtype=torch.float16
    ).to(CONFIG["device"])
    
    pipe.set_progress_bar_config(disable=True) # Hide the bar for individual patches

    print(f"Loading full mask from {CONFIG['input_mask_path']}...")
    full_control_mask = Image.open(CONFIG["input_mask_path"]).convert("RGB")
    width, height = full_control_mask.size
    
    # ==========================================
    # 2. CANVAS INITIALIZATION & STEP LOGIC
    # ==========================================
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas_tracker = Image.new("L", (width, height), 0) 

    # Helper function to calculate steps and force the final edge
    def get_sliding_steps(max_dim, p_size, stride):
        steps = list(range(0, max_dim - p_size + 1, stride))
        if steps[-1] != max_dim - p_size:
            steps.append(max_dim - p_size) # Force the window to the exact edge
        return steps

    y_steps = get_sliding_steps(height, CONFIG["patch_size"], CONFIG["stride"])
    x_steps = get_sliding_steps(width, CONFIG["patch_size"], CONFIG["stride"])
    
    # ==========================================
    # 3. AUTOREGRESSIVE GENERATION LOOP
    # ==========================================
    for y in tqdm(y_steps, desc="Rows Processed"):
        for x in tqdm(x_steps, desc="Columns Processed", leave=False):
            
            # 1. Crop the structural ControlNet mask for this 512x512 area
            control_patch = full_control_mask.crop((x, y, x + CONFIG["patch_size"], y + CONFIG["patch_size"]))
            
            # 2. Crop the current canvas (contains overlap from previous patches)
            canvas_patch = canvas.crop((x, y, x + CONFIG["patch_size"], y + CONFIG["patch_size"]))
            
            # 3. Crop the tracker to see what has already been generated
            tracker_patch = canvas_tracker.crop((x, y, x + CONFIG["patch_size"], y + CONFIG["patch_size"]))
            
            # 4. Create the Inpaint Mask
            # Inpaint pipeline needs: White (255) = Generate this, Black (0) = Keep this intact
            # So we invert the tracker patch.
            inpaint_mask = Image.eval(tracker_patch, lambda val: 255 - val)
            
            # 5. Generate the Patch
            with torch.autocast("cuda"):
                result_patch = pipe(
                    prompt=CONFIG["prompt"],
                    image=canvas_patch,
                    mask_image=inpaint_mask,
                    control_image=control_patch,
                    num_inference_steps=CONFIG["num_inference_steps"],
                ).images[0]
            
            # 6. Paste ONLY the newly generated pixels back onto the canvas
            # We use the inpaint_mask as an alpha channel so we don't overwrite the overlap
            canvas.paste(result_patch, (x, y), inpaint_mask)
            
            # 7. Update the tracker to mark this 512x512 square as completely generated
            draw = ImageDraw.Draw(canvas_tracker)
            draw.rectangle([x, y, x + CONFIG["patch_size"] - 1, y + CONFIG["patch_size"] - 1], fill=255)

    # Save the final masterpiece
    print(f"Generation complete! Saving to {CONFIG['output_image_path']}...")
    canvas.save(CONFIG["output_image_path"])

if __name__ == "__main__":
    main()