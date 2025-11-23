import os
import uuid

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

# load the model and processor
try:
    path = "ByteDance/Sa2VA-Qwen3-VL-4B"
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()

    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, use_fast=False)
except Exception as e:
    print(f"Error: {e}")

def test():
    image_path = "C:\\hook-extraction-service\\src\\sa2va\\test.jpeg"
    text_prompts = "<image>Could you please give me a brief description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer."
    image = Image.open(image_path).convert('RGB')
    input_dict = {
        'image': image,
        'text': text_prompts,
        'past_text': '',
        'mask_prompts': None,
        'processor': processor,
    }
    return_dict = model.predict_forward(**input_dict)
    answer = return_dict["prediction"] # the text format answer
    masks = return_dict['prediction_masks']

def get_masks(image_path, targets):
    """Generate masks for a list of targets, each with 'label' and 'bbox_2d' keys.
    
    Args:
        image_path: Path to the input image
        targets: List of dicts, each containing:
            - 'label': str describing the target object
            - 'bbox_2d': tuple/list of (x0, y0, x1, y1) coordinates
    
    Returns:
        List of dicts, one per target, each containing:
            - 'label': The target label
            - 'bbox_2d': The bounding box coordinates
            - 'masks': List of mask arrays for this target
            - 'overlay_paths': List of paths to overlay visualizations
    """
    image = Image.open(image_path).convert('RGB')
    img_width, img_height = image.size
    base_image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    global_merged_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for target_info in targets:
        label = target_info['label']
        bbox_2d = target_info['bbox_2d']
        
        text_prompts = f"""
        <image>Could you please help me find the {label} in the image?
        Please respond with interleaved segmentation masks for the corresponding parts of the answer.
        """
        
        coords = list(bbox_2d)
        x0, y0, x1, y1 = coords
        
        # Extend the bounding box by 10 pixels on each edge, clamped to image boundaries
        x0 = max(0, x0 - 10)
        y0 = max(0, y0 - 10)
        x1 = min(img_width, x1 + 10)
        y1 = min(img_height, y1 + 10)

        cropped_image = image.crop((x0, y0, x1, y1))
        cropped_width, cropped_height = cropped_image.size
        resized_image = cropped_image.resize((1000, 1000), Image.Resampling.LANCZOS)

        input_dict = {
            'image': resized_image,
            'text': text_prompts,
            'past_text': '',
            'mask_prompts': None,
            'processor': processor,
        }
        return_dict = model.predict_forward(**input_dict)
        masks = return_dict.get('prediction_masks', [])

        full_sized_masks = []

        for idx, mask in enumerate(masks):
            mask_np = np.array(mask)
            mask_np = np.squeeze(mask_np)
            if mask_np.size == 0:
                continue

            if mask_np.dtype != np.uint8:
                if mask_np.max() <= 1.0:
                    mask_np = (mask_np * 255).astype(np.uint8)
                else:
                    mask_np = mask_np.astype(np.uint8)

            mask_resized = Image.fromarray(mask_np).resize((cropped_width, cropped_height), Image.Resampling.NEAREST)
            mask_full = np.zeros((img_height, img_width), dtype=np.uint8)
            mask_full[y0:y1, x0:x1] = np.array(mask_resized)
            full_sized_masks.append(mask_full)
            
            # Merge into the global combined mask
            global_merged_mask = np.maximum(global_merged_mask, mask_full)

        if not full_sized_masks:
            print(f"No masks were generated for target '{label}' with bbox {bbox_2d}.")
        
        results.append({
            'label': label,
            'bbox_2d': bbox_2d,
            'masks': full_sized_masks,
        })
    
    # Create a single overlay image with all masks from all targets merged
    overlay_path = None
    if np.any(global_merged_mask > 0):
        color_mask = np.zeros_like(base_image_bgr)
        color_mask[global_merged_mask > 0] = (0, 0, 255)
        overlay = cv2.addWeighted(base_image_bgr, 0.7, color_mask, 0.3, 0)

        overlay_path = os.path.join(output_dir, f"mask_overlay_all_{uuid.uuid4().hex}.png")
        cv2.imwrite(overlay_path, overlay)
    
    return {
        'results': results,
        'overlay_path': overlay_path,
    }


# get_masks("C:\\hook-extraction-service\\src\\bg_removal\\resources\\test.jpg",
#           [{'label': 'person2', 'bbox_2d': (231, 299, 362, 451), 'is_behind_persons': False},
#            {'label': 'shoe in front of person2', 'bbox_2d': (237, 373, 324, 419), 'is_behind_persons': False},
#            {'label': 'person1', 'bbox_2d': (72, 262, 200, 449), 'is_behind_persons': False},
#            {'label': 'shoe and box in front of person1', 'bbox_2d': (117, 346, 207, 449), 'is_behind_persons': False}]
#           )
# test()
