import cv2
import numpy as np
from PIL import Image as PILImage
from matplotlib import pyplot as plt
from google import genai
from google.genai import types
import random
import base64
from IPython.display import Markdown, HTML
from base64 import b64encode
import json
import io
from io import BytesIO
import matplotlib.pyplot as plt
import uuid
import sys
import os


# Comfigure Gemini API key.
client = genai.Client(api_key="AIzaSyAwq5CWJcX78-mcdhVCyl7I8LqxxJxxsbU")

# Configure model.
MODEL_ID = "gemini-2.5-pro" # @param ["gemini-2.5-flash-lite","gemini-2.5-flash","gemini-2.5-pro"] {"allow-input":true}

bounding_box_system_instructions = """
    Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
    If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
"""

safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]

def parse_json(text: str) -> str:
    return text.strip().removeprefix("```json").removesuffix("```")

"""
Function adapted from:
https://github.com/google-gemini/cookbook/blob/main/quickstarts/Spatial_understanding.ipynb
"""
def generate_mask(predicted_str: str, *, img_height: int, img_width: int) -> list[tuple[np.ndarray, str]]:
    try:
        items = json.loads(parse_json(predicted_str))
        if not isinstance(items, list):
            print("Error: Parsed JSON is not a list.")
            return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Problematic string snippet: {predicted_str[:200]}...")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during JSON parsing: {e}")
        return []

    segmentation_data = []
    default_label = "unknown"

    for item_idx, item in enumerate(items):
        if not isinstance(item, dict) or "box_2d" not in item or "mask" not in item:
            print(f"Skipping invalid item structure at index {item_idx}: {item}")
            continue

        label = item.get("label", default_label)
        if not isinstance(label, str) or not label:
            label = default_label

        png_str = item["mask"]
        if not isinstance(png_str, str) or not png_str.startswith("data:image/png;base64,"):
            print(f"Skipping item {item_idx} (label: {label}) with invalid mask format.")
            continue
        png_str = png_str.removeprefix("data:image/png;base64,")
        try:
            png_bytes = base64.b64decode(png_str)
            
            # Display the image from png_bytes
            mask_image = PILImage.open(BytesIO(png_bytes))
            plt.figure(figsize=(8, 8))
            plt.imshow(mask_image, cmap='gray')
            plt.title(f"Mask for item {item_idx} (label: {label})")
            plt.axis('off')
            plt.savefig(f'mask_preview_{item_idx}_{label}.png')
            
            bbox_mask = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            if bbox_mask is None:
                print(f"Skipping item {item_idx} (label: {label}) because mask decoding failed.")
                continue
        except (base64.binascii.Error, ValueError, Exception) as e:
            print(f"Error decoding base64 or image data for item {item_idx} (label: {label}): {e}")
            continue

        try:
            box = item["box_2d"]
            if not isinstance(box, list) or len(box) != 4:
                print(f"Skipping item {item_idx} (label: {label}) with invalid box_2d format: {box}")
                continue
            y0_norm, x0_norm, y1_norm, x1_norm = map(float, box)
            abs_y0 = max(0, min(int(y0_norm / 1000.0 * img_height), img_height - 1))
            abs_x0 = max(0, min(int(x0_norm / 1000.0 * img_width), img_width - 1))
            abs_y1 = max(0, min(int(y1_norm / 1000.0 * img_height), img_height))
            abs_x1 = max(0, min(int(x1_norm / 1000.0 * img_width), img_width))
            bbox_height = abs_y1 - abs_y0
            bbox_width = abs_x1 - abs_x0
            if bbox_height <= 0 or bbox_width <= 0:
                print(f"Skipping item {item_idx} (label: {label}) with invalid bbox dims: {box} -> ({bbox_width}x{bbox_height})")
                continue
        except (ValueError, TypeError) as e:
            print(f"Skipping item {item_idx} (label: {label}) due to error processing box_2d: {e}")
            continue

        try:
            if bbox_mask.shape[0] > 0 and bbox_mask.shape[1] > 0:
                resized_bbox_mask = cv2.resize(
                    bbox_mask, (bbox_width, bbox_height), interpolation=cv2.INTER_LINEAR
                )
            else:
                print(f"Skipping item {item_idx} (label: {label}) due to empty decoded mask before resize.")
                continue
        except cv2.error as e:
            print(f"Error resizing mask for item {item_idx} (label: {label}): {e}")
            continue

        full_mask = np.zeros((img_height, img_width), dtype=np.uint8)  # Start with a black mask (0s)
        try:
            full_mask[abs_y0:abs_y1, abs_x0:abs_x1] = resized_bbox_mask  # Place the white mask in the right location
        except ValueError as e:
            print(f"Error placing mask for item {item_idx} (label: {label}): {e}. Shape mismatch: slice=({bbox_height},{bbox_width}), resized={resized_bbox_mask.shape}. Attempting correction.")
            try:
                resized_bbox_mask_corrected = cv2.resize(bbox_mask, (bbox_width, bbox_height), interpolation=cv2.INTER_LINEAR)
                full_mask[abs_y0:abs_y1, abs_x0:abs_x1] = resized_bbox_mask_corrected
                print("  -> Corrected placement.")
            except Exception as inner_e:
                print(f"  -> Failed to correct placement: {inner_e}")
                continue

        segmentation_data.append((full_mask, label))

    return segmentation_data


"""
Function adapted from:
https://github.com/google-gemini/cookbook/blob/main/quickstarts/Spatial_understanding.ipynb
"""
def create_binary_mask_overlay(
        img: PILImage.Image,
        segmentation_data: list[tuple[np.ndarray, str]],
        alpha: float = 0.8
) -> np.ndarray:

    binary_mask = np.zeros(img.size[::-1], dtype=np.uint8)

    for mask, label in segmentation_data:
        if mask is not None and mask.shape == binary_mask.shape:
            binary_mask = np.maximum(binary_mask, mask)

    result = np.zeros_like(binary_mask, dtype=np.uint8)
    result[binary_mask > 0] = 255

    return result
    

def get_bounding_rect_covering_mask_percentage(binary_mask: np.ndarray, coverage_threshold: float = 0.8) -> tuple:
    """
    Find a rectangle that covers at least n% of the masked area using contours.
    
    Args:
        binary_mask: Binary mask array where non-zero values indicate the object
        coverage_threshold: Percentage of mask to cover (0.0 to 1.0), default 0.8 (80%)
        
    Returns:
        tuple: (x, y, w, h) of the bounding rectangle, or None if not found
    """
    # Ensure mask is uint8
    mask_uint8 = binary_mask.astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Calculate mask area
    total_mask_area = np.count_nonzero(mask_uint8)
    
    # Calculate rectangle area
    rect_area = w * h
    
    # Calculate coverage percentage
    coverage = total_mask_area / rect_area if rect_area > 0 else 0
    
    print(f"Bounding rect: ({x}, {y}, {w}, {h})")
    print(f"Mask area: {total_mask_area}, Rect area: {rect_area}, Coverage: {coverage:.2%}")
    
    if coverage >= coverage_threshold:
        return (x, y, w, h)
    else:
        print(f"Warning: Coverage {coverage:.2%} is below threshold {coverage_threshold:.2%}")
        return (x, y, w, h)  # Still return the rect, but with warning


def analyze_scene(image_path: str, depth_image_path: str = None) -> dict:
    """
    Analyze a live selling show image to determine scene type and banner presence.
    
    Args:
        image_path: Path to the image file
        depth_image_path: Optional path to the depth image for reference
        
    Returns:
        dict: {
            "sceneType": str ("studio" or "b-roll"),
            "hasBanner": bool,
            "confidence": float (0.0-1.0)
        }
    """
    try:
        # Load image
        img = PILImage.open(image_path)
        
        # Load depth image if provided
        depth_img = None
        if depth_image_path and os.path.exists(depth_image_path):
            depth_img = PILImage.open(depth_image_path)
            print(f"Using depth image for scene analysis: {depth_image_path}")
        
        # Analyze the image to determine scene type and banner presence
        if depth_img:
            analysis_prompt = """Analyze these images from a live selling show (original image and depth map) and provide:
1. Whether this is a STUDIO scene (the hosts sitting by a table) or B-ROLL scene (showing products in detail)
2. Whether there is a banner area or a colored area suitable for banner that spans across the entire width of the image at the bottom (below the main content)

Use the depth map to help identify spatial relationships and depth information.

Return a JSON object with:
{
    "sceneType": "studio" or "b-roll",
    "hasBanner": true or false,
}"""
            contents = [analysis_prompt, img, "Depth map:", depth_img]
        else:
            analysis_prompt = """Analyze this image from a live selling show and provide:
1. Whether this is a STUDIO scene (showing the host/anchor and products on display) or B-ROLL scene (showing products in detail without the host)
2. Whether there is a banner area or a colored area suitable for banner that spans across the entire width of the image at the bottom (below the main content)

Return a JSON object with:
{
    "sceneType": "studio" or "b-roll",
    "hasBanner": true or false,
}"""
            contents = [analysis_prompt, img]
        
        print("Calling Gemini API for scene analysis...")
        analysis_response = client.models.generate_content(
            model=MODEL_ID,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.5,
                safety_settings=safety_settings,
            )
        )
        
        # Parse analysis result
        analysis_text = analysis_response.text
        analysis_data = json.loads(parse_json(analysis_text))
        print(f"Scene analysis: {analysis_data}")
        
        return analysis_data
        
    except Exception as e:
        print(f"Error analyzing scene: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "sceneType": "unknown",
            "hasBanner": False,
            "confidence": 0.0
        }


def detect_banner_top_edge(image_path: str, coverage_threshold: float = 0.98) -> dict:
    """
    Analyze a live selling show image to detect banner area and determine scene type.
    
    Args:
        image_path: Path to the image file
        coverage_threshold: Percentage of mask to cover (0.0 to 1.0), default 0.98 (98%)
        
    Returns:
        dict: {
            "isValid": bool,
            "y1": int (only if isValid is True),
            "sceneType": str ("studio" or "b-roll"),
            "hasBanner": bool,
            "reason": str (explanation if isValid is False)
        }
    """
    try:
        # Load image
        img = PILImage.open(image_path)
        img_height, img_width = img.size[1], img.size[0]
        print(f"Image dimensions: {img_width}x{img_height}")
        
        # First, analyze the image to determine scene type and banner presence
        analysis_prompt = """Analyze this image from a live selling show and provide:
1. Whether this is a STUDIO scene (showing the host/anchor and products on display) or B-ROLL scene (showing products in detail without the host)
2. Whether there is a banner area or a black area suitable for banner that spans across the entire width of the image at the bottom (below the main content)

Return a JSON object with:
{
    "sceneType": "studio" or "b-roll",
    "hasBanner": true or false,
    "confidence": 0.0-1.0
}"""
        
        print("Calling Gemini API for scene analysis...")
        analysis_response = client.models.generate_content(
            model=MODEL_ID,
            contents=[analysis_prompt, img],
            config=types.GenerateContentConfig(
                temperature=0.5,
                safety_settings=safety_settings,
            )
        )
        
        # Parse analysis result
        analysis_text = analysis_response.text
        analysis_data = json.loads(parse_json(analysis_text))
        print(f"Scene analysis: {analysis_data}")
        
        scene_type = analysis_data.get("sceneType", "unknown")
        has_banner = analysis_data.get("hasBanner", False)
        
        # Return invalid if it's b-roll or no banner detected
        if scene_type == "b-roll":
            return {
                "isValid": False,
                "sceneType": scene_type,
                "hasBanner": has_banner,
                "reason": "B-roll scene detected, not a studio scene"
            }
        
        if not has_banner:
            return {
                "isValid": False,
                "sceneType": scene_type,
                "hasBanner": has_banner,
                "reason": "No banner area detected in the image"
            }
        
        # Now segment the banner area
        object_to_segment = 'The banner area that is across the whole width of the image below the table'
        segmentation_prompt = f"""Give the segmentation masks for {object_to_segment}. 
Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key 'box_2d', 
the segmentation mask in key 'mask', and the text label in the key 'label'."""
        
        print("Calling Gemini API for banner segmentation...")
        segmentation_response = client.models.generate_content(
            model=MODEL_ID,
            contents=[segmentation_prompt, img],
            config=types.GenerateContentConfig(
                temperature=0.5,
                safety_settings=safety_settings,
            )
        )
        
        # Parse response and generate mask
        result = segmentation_response.text
        segmentation_data = generate_mask(result, img_height=img_height, img_width=img_width)
        
        if not segmentation_data:
            print("No segmentation data found")
            return {
                "isValid": False,
                "sceneType": scene_type,
                "hasBanner": has_banner,
                "reason": "Failed to segment banner area"
            }
        
        print(f"Segmentation data found: {len(segmentation_data)} masks")
        
        # Create binary mask from segmentation
        binary_mask = create_binary_mask_overlay(img, segmentation_data, alpha=0.8)
        binary_array = np.array(binary_mask)
        
        # Get bounding rectangle covering the masked area
        bounding_rect = get_bounding_rect_covering_mask_percentage(binary_array, coverage_threshold=coverage_threshold)
        
        if bounding_rect is None:
            print("Could not determine bounding rectangle")
            return {
                "isValid": False,
                "sceneType": scene_type,
                "hasBanner": has_banner,
                "reason": "Could not determine banner bounding rectangle"
            }
        
        # Extract y1 (top edge)
        x1, y1, w, h = bounding_rect
        print(f"Banner top edge (y1): {y1}")
        
        return {
            "isValid": True,
            "y1": int(y1),
            "sceneType": scene_type,
            "hasBanner": has_banner
        }
        
    except Exception as e:
        print(f"Error detecting banner: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "isValid": False,
            "reason": f"Error: {str(e)}"
        }


# Example usage:
def test_banner_location():
    input_image = 'first_frame_temp.jpg'
    image_path = f"resources/{input_image}"
    
    # Detect banner and analyze scene
    result = detect_banner_top_edge(image_path, coverage_threshold=0.99)
    print(f"Result: {result}")
    
    if result.get("isValid"):
        print(f"Valid studio scene with banner at y1={result['y1']}")
    else:
        print(f"Invalid: {result.get('reason', 'Unknown reason')}")
    
def generate_remove_bg_mask(image_path: str):
    """
    Generate a background removal mask using Gemini segmentation.
    
    Args:
        image_path: Path to the input image file
        
    Returns:
        Binary mask as numpy array, or None if generation fails
    """
    object_to_segment = 'The table surface, all persons and the things they are holding'
    segmentation_prompt = f"""Give the segmentation masks for {object_to_segment}. 
Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key 'box_2d', 
the segmentation mask in key 'mask', and the text label in the key 'label'."""
        
    print("Calling Gemini API for background segmentation...")
    img = PILImage.open(image_path)
    img_height, img_width = img.size[1], img.size[0]
    segmentation_response = client.models.generate_content(
            model=MODEL_ID,
            contents=[segmentation_prompt, img],
            config=types.GenerateContentConfig(
                temperature=0.5,
                safety_settings=safety_settings,
            )
        )
        
    # Parse response and generate mask
    result = segmentation_response.text
    segmentation_data = generate_mask(result, img_height=img_height, img_width=img_width)
        
    if not segmentation_data:
        print("No segmentation data found")
        return None
        
    print(f"Segmentation data found: {len(segmentation_data)} masks")
        
    # Create binary mask from segmentation
    binary_mask = create_binary_mask_overlay(img, segmentation_data, alpha=0.8)
    binary_array = np.array(binary_mask)

    img_array = np.array(img)
    binary_array = np.array(binary_mask)

    binary_array_3d = np.expand_dims(binary_array, axis=2)
    binary_array_3d = np.repeat(binary_array_3d, 3, axis=2)

    gap_width = 20
    empty_img = np.full((img_array.shape[0], gap_width, img_array.shape[2]), 255, dtype=img_array.dtype)
    combined_image = np.hstack((img_array, empty_img, binary_array_3d))
    cv2.imwrite(f'combined_image_{uuid.uuid4()}.jpg', combined_image)

    return binary_mask


def generate_remove_items_mask(image_path: str):
    """
    Generate a background removal mask for items to remove using Gemini segmentation.
    
    Args:
        image_path: Path to the input image file
        
    Returns:
        Binary mask as numpy array, or None if generation fails
    """
    object_to_segment = 'goods which centers are above the head of the persons'
    segmentation_prompt = f"""Give the segmentation masks for {object_to_segment}. 
Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key 'box_2d', 
the segmentation mask in key 'mask', and the text label in the key 'label'."""
        
    print("Calling Gemini API for background items segmentation...")
    img = PILImage.open(image_path)
    img_height, img_width = img.size[1], img.size[0]
    segmentation_response = client.models.generate_content(
            model=MODEL_ID,
            contents=[segmentation_prompt, img],
            config=types.GenerateContentConfig(
                temperature=0.5,
                safety_settings=safety_settings,
            )
        )
        
    # Parse response and generate mask
    result = segmentation_response.text
    segmentation_data = generate_mask(result, img_height=img_height, img_width=img_width)
        
    if not segmentation_data:
        print("No segmentation data found for items to remove")
        return None
        
    print(f"Segmentation data found: {len(segmentation_data)} masks")
        
    # Create binary mask from segmentation
    binary_mask = create_binary_mask_overlay(img, segmentation_data, alpha=0.8)
    binary_array = np.array(binary_mask)

    img_array = np.array(img)
    binary_array = np.array(binary_mask)

    binary_array_3d = np.expand_dims(binary_array, axis=2)
    binary_array_3d = np.repeat(binary_array_3d, 3, axis=2)

    gap_width = 20
    empty_img = np.full((img_array.shape[0], gap_width, img_array.shape[2]), 255, dtype=img_array.dtype)
    combined_image = np.hstack((img_array, empty_img, binary_array_3d))
    cv2.imwrite(f'combined_image_remove_items_{uuid.uuid4()}.jpg', combined_image)

    return binary_mask


def generate_goods_behind_persons_mask(
    image_path: str,
    output_path: str,
    num_steps: int = 2,
    ensemble_size: int = 4,
    processing_res: int = -1,
    dtype: str = "fp16"
):
    """
    Generate a mask for goods behind persons using Gemini segmentation with depth information.
    
    Args:
        image_path: Path to the input RGB image file
        output_path: Path to save the output mask image
        num_steps: Number of steps for depth ODE solver
        ensemble_size: Number of ensemble members for depth estimation
        processing_res: Processing resolution for depth estimation (-1 to disable resizing)
        dtype: Data type for depth estimation ("fp32", "bf16", "fp16")
        
    Returns:
        Binary mask as numpy array, or None if generation fails
    """
    # Import depth inference function
    try:
        from src.depth.depthfm.inference import get_depth_image
    except ImportError:
        # Try alternative import path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'depth'))
        from depthfm.inference import get_depth_image
    
    print("Generating depth image...")
    depth_img_pil = get_depth_image(
        image_path=image_path,
        num_steps=num_steps,
        ensemble_size=ensemble_size,
        processing_res=processing_res,
        dtype=dtype,
        return_pil=True
    )
    
    object_to_segment = 'goods behind the persons (refer to the depth image to identify items that are further away)'
    segmentation_prompt = f"""Give the segmentation masks for {object_to_segment}. 
Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key 'box_2d', 
the segmentation mask in key 'mask', and the text label in the key 'label'."""
        
    print("Calling Gemini API for goods behind persons segmentation...")
    img = PILImage.open(image_path)
    img_height, img_width = img.size[1], img.size[0]
    
    # Resize image and depth to 1024x1024 for LLM
    img_resized = img.resize((1024, 1024), PILImage.Resampling.LANCZOS)
    depth_img_resized = depth_img_pil.resize((1024, 1024), PILImage.Resampling.LANCZOS)
    
    segmentation_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[segmentation_prompt, img_resized, depth_img_resized],
            config=types.GenerateContentConfig(
                temperature=0.5,
                safety_settings=safety_settings,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        
    # Parse response and generate mask (using resized dimensions)
    result = segmentation_response.text
    segmentation_data = generate_mask(result, img_height=1024, img_width=1024)
        
    if not segmentation_data:
        print("No segmentation data found for goods behind persons")
        return None
        
    print(f"Segmentation data found: {len(segmentation_data)} masks")
        
    # Create binary mask from segmentation (on resized dimensions)
    binary_mask = create_binary_mask_overlay(img_resized, segmentation_data, alpha=0.8)
    binary_array = np.array(binary_mask)
    
    # Scale mask back to original image dimensions
    binary_mask_scaled = PILImage.fromarray(binary_array).resize((img_width, img_height), PILImage.Resampling.LANCZOS)
    binary_array = np.array(binary_mask_scaled)

    img_array = np.array(img)
    
    # Create overlay visualization with mask in color at 0.5 opacity
    overlay_img = img_array.copy().astype(np.float32)
    
    # Create colored mask overlay (red for masked areas)
    mask_overlay = np.zeros_like(img_array, dtype=np.float32)
    mask_overlay[binary_array > 0] = [0, 0, 255]  # Red color for masked areas
    
    # Blend with 0.5 opacity
    result_img = cv2.addWeighted(overlay_img, 0.5, mask_overlay, 0.5, 0)
    result_img = result_img.astype(np.uint8)
    
    cv2.imwrite(output_path, result_img)
    print(f"==> Saved masked overlay image to {output_path}")

    return binary_mask


def generate_persons_and_front_objects_mask(
    image_path: str,
    output_path: str,
    num_steps: int = 2,
    ensemble_size: int = 4,
    processing_res: int = -1,
    dtype: str = "fp16"
):
    """
    Generate a mask for persons and objects in front of them using Gemini segmentation with depth information.
    
    Args:
        image_path: Path to the input RGB image file
        output_path: Path to save the output mask image
        num_steps: Number of steps for depth ODE solver
        ensemble_size: Number of ensemble members for depth estimation
        processing_res: Processing resolution for depth estimation (-1 to disable resizing)
        dtype: Data type for depth estimation ("fp32", "bf16", "fp16")
        
    Returns:
        Binary mask as numpy array, or None if generation fails
    """
    # Import depth inference function
    try:
        from src.depth.depthfm.inference import get_depth_image
    except ImportError:
        # Try alternative import path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'depth'))
        from depthfm.inference import get_depth_image
    
    print("Generating depth image...")
    depth_img_pil = get_depth_image(
        image_path=image_path,
        num_steps=num_steps,
        ensemble_size=ensemble_size,
        processing_res=processing_res,
        dtype=dtype,
        return_pil=True
    )
    
    object_to_segment = 'persons and objects which are in front of the persons (refer to the depth image and your reasoning to identify items that are closer)'
    segmentation_prompt = f"""Give the segmentation masks for {object_to_segment}. 
Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key 'box_2d', 
the segmentation mask in key 'mask', and the text label in the key 'label'. 
If there is one person, label him as person1, and if there are two, label them as person1, person2, and etc.
"""
        
    print("Calling Gemini API for persons and front objects segmentation...")
    img = PILImage.open(image_path)
    img_height, img_width = img.size[1], img.size[0]

    img_resized = img.resize((1024, 1024), PILImage.Resampling.LANCZOS)
    depth_img_resized = depth_img_pil.resize((1024, 1024), PILImage.Resampling.LANCZOS)
    
    segmentation_response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[segmentation_prompt, img_resized, depth_img_resized],
            config=types.GenerateContentConfig(
                temperature=1,
                safety_settings=safety_settings,
                thinking_config=types.ThinkingConfig(thinking_budget=-1)
            )
        )
        
    # Parse response and generate mask
    result = segmentation_response.text
    segmentation_data = generate_mask(result, img_height=1024, img_width=1024)
        
    if not segmentation_data:
        print("No segmentation data found for persons and front objects")
        return None
        
    print(f"Segmentation data found: {len(segmentation_data)} masks")
        
    # Create binary mask from segmentation (on resized dimensions)
    binary_mask = create_binary_mask_overlay(img_resized, segmentation_data, alpha=0.8)
    binary_array = np.array(binary_mask)
    
    # Scale mask back to original image dimensions
    binary_mask_scaled = PILImage.fromarray(binary_array).resize((img_width, img_height), PILImage.Resampling.LANCZOS)
    binary_array = np.array(binary_mask_scaled)

    img_array = np.array(img)
    
    # Create overlay visualization with mask in color at 0.5 opacity
    overlay_img = img_array.copy().astype(np.float32)
    
    # Create colored mask overlay (green for masked areas)
    mask_overlay = np.zeros_like(img_array, dtype=np.float32)
    mask_overlay[binary_array > 0] = [0, 255, 0]  # Green color for masked areas
    
    # Blend with 0.5 opacity
    result_img = cv2.addWeighted(overlay_img, 0.5, mask_overlay, 0.5, 0)
    result_img = result_img.astype(np.uint8)
    
    cv2.imwrite(output_path, result_img)
    print(f"==> Saved masked overlay image to {output_path}")

    return binary_mask


def generate_depth_image(
    image_path: str,
    output_path: str = None,
    num_steps: int = 2,
    ensemble_size: int = 4,
    processing_res: int = -1,
    dtype: str = "fp16"
):
    """
    Generate depth image from RGB image.
    
    Args:
        image_path: Path to the input RGB image file
        output_path: Optional path to save the depth image
        num_steps: Number of steps for depth ODE solver
        ensemble_size: Number of ensemble members for depth estimation
        processing_res: Processing resolution for depth estimation (-1 to disable resizing)
        dtype: Data type for depth estimation ("fp32", "bf16", "fp16")
        
    Returns:
        PIL.Image: Depth image
    """
    # Import depth inference function
    try:
        from src.depth.depthfm.inference import get_depth_image
    except ImportError:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'depth'))
        from depthfm.inference import get_depth_image
    
    print("Generating depth image...")
    depth_img_pil = get_depth_image(
        image_path=image_path,
        num_steps=num_steps,
        ensemble_size=ensemble_size,
        processing_res=processing_res,
        dtype=dtype,
        return_pil=True
    )
    
    # Save if output path provided
    if output_path:
        depth_img_pil.save(output_path)
        print(f"Saved depth image to {output_path}")
    
    return depth_img_pil


def get_all_objects_and_persons_with_depth(
    image_path: str,
    depth_image_path: str,
    output_path: str = None
):
    """
    Extract all objects and persons from the image with bounding boxes and depth classification.
    
    Args:
        image_path: Path to the input RGB image file
        depth_image_path: Path to the depth image file (required)
        output_path: Optional path to save visualization with bounding boxes
        
    Returns:
        List of dicts with keys:
            - label: str (e.g., "person1", "bottle", etc.)
            - bbox_2d: tuple (x0, y0, x1, y1) in absolute pixel coordinates
            - is_behind_persons: bool (True if object is behind persons, False for persons themselves)
    """
    # Load depth image
    if not depth_image_path or not os.path.exists(depth_image_path):
        raise ValueError(f"Depth image path is required and must exist: {depth_image_path}")
    
    print(f"Loading depth image from {depth_image_path}")
    depth_img_pil = PILImage.open(depth_image_path)
    
    img = PILImage.open(image_path)
    img_height, img_width = img.size[1], img.size[0]
    
    # Resize for Gemini API
    img_resized = img.resize((1024, 1024), PILImage.Resampling.LANCZOS)
    depth_img_resized = depth_img_pil.resize((1024, 1024), PILImage.Resampling.LANCZOS)
    
    # First, get all persons and front objects
    front_prompt = """Give the segmentation masks for persons. 
Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key 'box_2d', 
the segmentation mask in key 'mask', and the text label in the key 'label'. 
If there is one person, label him as person1, and if there are two, label them as person1, person2, and etc."""
    
    print("Calling Gemini API for persons and front objects...")
    front_response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[front_prompt, img_resized, depth_img_resized],
        config=types.GenerateContentConfig(
            temperature=1,
            safety_settings=safety_settings,
            thinking_config=types.ThinkingConfig(thinking_budget=-1)
        )
    )
    
    # Second, get objects behind persons
    behind_prompt = """Give the segmentation masks for goods behind the persons in the scene background, typically large and at the center of the background (refer to the depth image to identify items that are further away, where darker areas are closer). 
Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key 'box_2d'.
If there is no such goods found, return an empty JSON list.
The segmentation mask in key 'mask', and the text label in the key 'label'."""
    
    print("Calling Gemini API for objects behind persons...")
    behind_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[behind_prompt, img_resized, depth_img_resized],
        config=types.GenerateContentConfig(
            temperature=0.5,
            safety_settings=safety_settings,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )
    
    # Parse both responses
    def parse_boxes_from_response(response_text, is_behind):
        """Extract bounding boxes from Gemini response."""
        try:
            items = json.loads(parse_json(response_text))
            if not isinstance(items, list):
                print(f"Error: Parsed JSON is not a list for {'behind' if is_behind else 'front'} objects.")
                return []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for {'behind' if is_behind else 'front'} objects: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error parsing {'behind' if is_behind else 'front'} objects: {e}")
            return []
        
        boxes = []
        for item_idx, item in enumerate(items):
            if not isinstance(item, dict) or "box_2d" not in item:
                print(f"Skipping invalid item at index {item_idx}")
                continue
            
            label = item.get("label", "unknown")
            if not isinstance(label, str) or not label:
                label = "unknown"
            
            try:
                box = item["box_2d"]
                if not isinstance(box, list) or len(box) != 4:
                    print(f"Skipping item {item_idx} (label: {label}) with invalid box_2d format")
                    continue
                
                # Convert normalized coordinates to absolute pixels
                y0_norm, x0_norm, y1_norm, x1_norm = map(float, box)
                
                # Scale from 1024x1024 to original dimensions
                x0 = max(0, min(int(x0_norm / 1000.0 * 1024 * img_width / 1024), img_width - 1))
                y0 = max(0, min(int(y0_norm / 1000.0 * 1024 * img_height / 1024), img_height - 1))
                x1 = max(0, min(int(x1_norm / 1000.0 * 1024 * img_width / 1024), img_width))
                y1 = max(0, min(int(y1_norm / 1000.0 * 1024 * img_height / 1024), img_height))
                
                if x1 <= x0 or y1 <= y0:
                    print(f"Skipping item {item_idx} (label: {label}) with invalid bbox dimensions")
                    continue
                
                boxes.append({
                    "label": label,
                    "bbox_2d": (x0, y0, x1, y1),
                    "is_behind_persons": is_behind
                })
                
            except (ValueError, TypeError) as e:
                print(f"Error processing box for item {item_idx} (label: {label}): {e}")
                continue
        
        return boxes
    
    # Collect all objects
    all_objects = []
    
    front_boxes = parse_boxes_from_response(front_response.text, is_behind=False)
    all_objects.extend(front_boxes)
    print(f"Found {len(front_boxes)} front objects/persons")
    
    behind_boxes = parse_boxes_from_response(behind_response.text, is_behind=True)
    all_objects.extend(behind_boxes)
    print(f"Found {len(behind_boxes)} objects behind persons")
    
    print(f"Total objects detected: {len(all_objects)}")
    
    # Create visualization if output path is provided
    if output_path:
        img_array = np.array(img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        for obj in all_objects:
            x0, y0, x1, y1 = obj["bbox_2d"]
            label = obj["label"]
            is_behind = obj["is_behind_persons"]
            
            # Color code: green for persons/front objects, red for behind objects
            color = (0, 0, 255) if is_behind else (0, 255, 0)  # BGR format
            thickness = 2
            
            # Draw rectangle
            cv2.rectangle(img_bgr, (x0, y0), (x1, y1), color, thickness)
            
            # Draw label with background
            label_text = f"{label} ({'behind' if is_behind else 'front'})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            
            # Draw filled rectangle for text background
            cv2.rectangle(img_bgr, (x0, y0 - text_height - baseline - 5), (x0 + text_width, y0), color, -1)
            
            # Draw text
            cv2.putText(img_bgr, label_text, (x0, y0 - baseline - 2), font, font_scale, (255, 255, 255), font_thickness)
        
        cv2.imwrite(output_path, img_bgr)
        print(f"==> Saved visualization with {len(all_objects)} bounding boxes to {output_path}")
    
    return all_objects


# print(get_all_objects_and_persons_with_depth(image_path="resources/test.jpg", output_path="resources/test_all_objects.png"))