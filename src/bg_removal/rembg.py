# functions for calling rembg restful API to remove the background
import requests
import io
import numpy as np
from PIL import Image


def get_mask_with_rembg_api(image_path: str) -> np.ndarray | None:
    api_key = "CEyVf91XSKSBjLWuKoVaPpFw"
    if not api_key:
        return None
    try:
        with open(image_path, 'rb') as f:
            response = requests.post(
                'https://api.remove.bg/v1.0/removebg',
                files={'image_file': f},
                data={'size': 'auto', 'format': 'png'},  # PNG preserves alpha
                headers={'X-Api-Key': api_key},
                timeout=60,
            )
        if response.status_code != 200:
            print(f"remove.bg error: {response.status_code} {response.text[:200]}")
            return None

        # Load PNG with alpha and convert alpha to mask
        original = Image.open(image_path)
        im = Image.open(io.BytesIO(response.content)).convert("RGBA")
        if im.size != original.size:
            im = im.resize(original.size, Image.Resampling.LANCZOS)
        arr = np.asarray(im)
        alpha = arr[..., 3]
        mask = alpha > 0
        return mask.astype(np.float32)
    except Exception as e:
        print(f"remove.bg call failed: {e}")
        return None

