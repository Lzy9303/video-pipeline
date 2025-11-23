import os
import sys
import json
import cv2

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(root, 'src'))
from llm import locate_banner, locate_caption, locate_products


def test_locate_banner():
    base = os.path.join(root, 'test', 'resources', 'llm', "seed_vl")
    image_path = os.path.join(base, 'input.png')
    banner_location = locate_banner(image_path)
    (x0, y0, x1, y1) = banner_location['banner_bbox']
    image = cv2.imread(image_path)

    h, w = image.shape[:2]
    x_min_real = int(x0 * w / 1000)
    y_min_real = int(y0 * h / 1000)
    x_max_real = int(x1 * w / 1000)
    y_max_real = int(y1 * h / 1000)

    cv2.rectangle(image, (x_min_real, y_min_real), (x_max_real, y_max_real), color=(255, 0, 255), thickness=-1)

    output_path = os.path.splitext(image_path)[0] + "_without_banner.png"
    cv2.imwrite(output_path, image)
    print(f"成功保存标注图片: {output_path}")


def test_locate_caption():
    base = os.path.join(root, 'test', 'resources', 'llm', "seed_vl")
    image_path = os.path.join(base, 'input_without_banner.png')
    banner_location = locate_caption(image_path)
    (x0, y0, x1, y1) = banner_location['caption_bbox']
    image = cv2.imread(image_path)

    h, w = image.shape[:2]
    x_min_real = int(x0 * w / 1000)
    y_min_real = int(y0 * h / 1000)
    x_max_real = int(x1 * w / 1000)
    y_max_real = int(y1 * h / 1000)

    cv2.rectangle(image, (x_min_real, y_min_real), (x_max_real, y_max_real), color=(0, 255, 255), thickness=-1)

    output_path = os.path.splitext(image_path)[0] + "_without_caption.png"
    cv2.imwrite(output_path, image)
    print(f"成功保存标注图片: {output_path}")


def test_locate_products():
    base = os.path.join(root, 'test', 'resources', 'llm', "seed_vl")
    image_path = os.path.join(base, 'input_without_banner_without_caption.png')
    locations = locate_products(image_path)
    product_bboxes = locations['product_bboxes']
    image = cv2.imread(image_path)

    h, w = image.shape[:2]
    for product_bbox in product_bboxes:
        (x0, y0, x1, y1) = product_bbox
        x_min_real = int(x0 * w / 1000)
        y_min_real = int(y0 * h / 1000)
        x_max_real = int(x1 * w / 1000)
        y_max_real = int(y1 * h / 1000)
        cv2.rectangle(image, (x_min_real, y_min_real), (x_max_real, y_max_real), color=(255, 255, 0), thickness=-1)

    output_path = os.path.splitext(image_path)[0] + "_without_product.png"
    cv2.imwrite(output_path, image)
    print(f"成功保存标注图片: {output_path}")


test_locate_banner()
test_locate_caption()
test_locate_products()