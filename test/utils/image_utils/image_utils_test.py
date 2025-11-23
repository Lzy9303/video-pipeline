import os
import sys
import cv2


def test_cut_image_area():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    sys.path.insert(0, os.path.join(root, 'src'))
    from utils.image_utils import cut_image_area
    from models import Area
    base = os.path.join(root, 'test', 'resources', 'utils', 'image_utils', 'image_crop')
    os.makedirs(base, exist_ok=True)
    src_img = os.path.join(base, 'input.png')
    out_dir = os.path.join(base, 'result')
    cut_image_area(Area(146, 373, 432, 557), src_img, out_dir)
    cut_image_area(Area(132, 858, 318, 969), src_img, out_dir)


def test_image_resize():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    sys.path.insert(0, os.path.join(root, 'src'))
    from utils.image_utils import sample_image_1000
    base = os.path.join(root, 'test', 'resources', 'utils', 'image_utils', 'image_resize')
    os.makedirs(base, exist_ok=True)
    src_img = os.path.join(base, 'input.png')
    out_dir = os.path.join(base, 'result')
    sample_image_1000(src_img, out_dir)


# test_cut_image_area()
# test_image_resize()
