import os
import uuid
import cv2
from models.area import Area


def cut_image_area(area: Area, image_path: str, output_dir: str) -> str:
    img = cv2.imread(image_path)
    crop = img[area.y0:area.y1, area.x0:area.x1]
    os.makedirs(output_dir, exist_ok=True)
    name, ext = os.path.splitext(os.path.basename(image_path))
    out = os.path.join(output_dir, f"{name}_crop_{uuid.uuid4().hex}{ext}")
    cv2.imwrite(out, crop)
    return out


def sample_image_1000(image_path: str, output_dir: str) -> str:
    img = cv2.imread(image_path)
    os.makedirs(output_dir, exist_ok=True)
    h, w = img.shape[:2]
    interp = cv2.INTER_CUBIC if (h < 1000 or w < 1000) else cv2.INTER_AREA
    resized = cv2.resize(img, (1000, 1000), interpolation=interp)
    name, ext = os.path.splitext(os.path.basename(image_path))
    out = os.path.join(output_dir, f"{name}_{uuid.uuid4().hex}{ext}")
    cv2.imwrite(out, resized)
    return out
