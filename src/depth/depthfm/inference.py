import os
import torch
import einops
import argparse
import numpy as np
from PIL import Image
from PIL.Image import Resampling
from dfm import DepthFM
import matplotlib.pyplot as plt

# Global model - loaded once when module is imported
_depth_model = None

def _load_depth_model():
    """Load the depth model on module import."""
    global _depth_model
    try:
        default_ckpt = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'depthfm-v1.ckpt')
        default_ckpt = os.path.abspath(default_ckpt)
        if os.path.exists(default_ckpt):
            print(f"Loading DepthFM model from {default_ckpt}...")
            _depth_model = DepthFM(default_ckpt)
            _depth_model.cuda(0).eval()
            print("DepthFM model loaded successfully")
        else:
            print(f"Warning: Checkpoint not found at {default_ckpt}")
    except Exception as e:
        print(f"Warning: Failed to load model on import: {e}")

# Load model when module is imported
_load_depth_model()

def get_dtype_from_str(dtype_str):
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype_str]

def resize_max_res(
    img: Image.Image, max_edge_resolution: int, resample_method=Resampling.BILINEAR
) -> Image.Image:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.

    Args:
        img (`Image.Image`):
            Image to be resized.
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
        resample_method (`PIL.Image.Resampling`):
            Resampling method used to resize images.

    Returns:
        `Image.Image`: Resized image.
    """
    original_width, original_height = img.size
    downscale_factor = min( max_edge_resolution / original_width, max_edge_resolution / original_height)

    new_width  = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    new_width  = round(new_width / 64) * 64
    new_height = round(new_height / 64) * 64

    print(f"Resizing image from {original_width}x{original_height} to {new_width}x{new_height}")

    resized_img = img.resize((new_width, new_height), resample=resample_method)
    return resized_img, (original_width, original_height)

def load_im(fp, processing_res=-1):
    assert os.path.exists(fp), f"File not found: {fp}"
    im = Image.open(fp).convert('RGB')
    if processing_res < 0:
        processing_res = max(im.size)
    im, orig_res = resize_max_res(im, processing_res)
    x = np.array(im)
    x = einops.rearrange(x, 'h w c -> c h w')
    x = x / 127.5 - 1
    x = torch.tensor(x, dtype=torch.float32)[None]
    return x, orig_res

def get_depth_image(
    image_path: str,
    output_path: str = None,
    num_steps: int = 2,
    ensemble_size: int = 4,
    processing_res: int = -1,
    dtype: str = "fp16",
    no_color: bool = False,
    return_pil: bool = False
):
    """
    Generate depth image for a given input image using the pre-loaded model.
    
    Args:
        image_path: Path to the input image file
        output_path: Path to save the depth image. If None, uses image_path + '_depth.png'
        num_steps: Number of steps for ODE solver
        ensemble_size: Number of ensemble members
        processing_res: Longer edge of the image will be resized to this resolution. -1 to disable resizing.
        dtype: Run with specific precision ("fp32", "bf16", "fp16")
        no_color: If True, the depth map will be grayscale instead of colored
        return_pil: If True, return PIL Image object instead of just saving
        
    Returns:
        Path to saved depth image, or PIL Image if return_pil=True
    """
    if _depth_model is None:
        raise RuntimeError("Depth model failed to load during module initialization")
    
    print(f"Generating depth for: {image_path}")
    
    # Load image
    im, orig_res = load_im(image_path, processing_res)
    im = im.cuda(0)
    
    # Generate depth
    dtype_torch = get_dtype_from_str(dtype)
    _depth_model.model.dtype = dtype_torch
    with torch.autocast(device_type="cuda", dtype=dtype_torch):
        depth = _depth_model.predict_depth(im, num_steps=num_steps, ensemble_size=ensemble_size)
    depth = depth.squeeze(0).squeeze(0).cpu().numpy()  # (h, w) in [0, 1]
    
    # Convert depth to [0, 255] range
    if no_color:
        depth = (depth * 255).astype(np.uint8)
    else:
        depth = plt.get_cmap('magma')(depth, bytes=True)[..., :3]
    
    # Create PIL Image and resize to original resolution
    depth_img = Image.fromarray(depth)
    if depth_img.size != orig_res:
        depth_img = depth_img.resize(orig_res, Resampling.BILINEAR)
    
    # Save the depth map
    if output_path is None:
        output_path = image_path + '_depth.png'
    depth_img.save(output_path)
    print(f"==> Saved depth map to {output_path}")
    
    if return_pil:
        return depth_img
    return output_path


def main(args):
    print(f"{'Input':<10}: {args.img}")
    print(f"{'Steps':<10}: {args.num_steps}")
    print(f"{'Ensemble':<10}: {args.ensemble_size}")

    # Use the pre-loaded model
    get_depth_image(
        image_path=args.img,
        num_steps=args.num_steps,
        ensemble_size=args.ensemble_size,
        processing_res=args.processing_res,
        dtype=args.dtype,
        no_color=args.no_color
    )

def test(input_dir: str = "assets"):
    """
    Process all images in a directory and save depth maps to a results directory.
    
    Args:
        input_dir: Directory containing input images
    """
    import glob
    
    # Create results directory
    results_dir = os.path.join(input_dir, "result")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        try:
            filename = os.path.basename(image_path)
            output_filename = f"{os.path.splitext(filename)[0]}_depth.png"
            output_path = os.path.join(results_dir, output_filename)
            
            print(f"\n[{idx}/{len(image_files)}] Processing: {filename}")
            
            get_depth_image(
                image_path=image_path,
                output_path=output_path,
                num_steps=2,
                ensemble_size=4,
                processing_res=-1,
                dtype="fp16",
                no_color=False
            )
            
            print(f"✓ Saved to: {output_filename}")
            
        except Exception as e:
            print(f"✗ Failed to process {filename}: {str(e)}")
    
    print(f"\n✓ Batch processing complete! Results saved to: {results_dir}")

# test(input_dir="D:\\hook-extraction-service\\src\\bg_removal\\resources\\mask_test")