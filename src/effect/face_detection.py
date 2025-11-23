import cv2
import numpy as np
import math
from retinaface import RetinaFace
from PIL import Image


def overlay_on_eyes(image_path, overlay_image_path, output_path=None, scale_factor=2.5):
    """
    Overlay an image on the eyes area of each detected face.
    The overlay is rotated to be parallel to the line between the two eyes.
    
    Args:
        image_path: Path to the input image
        overlay_image_path: Path to the overlay image (e.g., sunglasses, censor bar)
        output_path: Optional path to save the result. If None, returns the image array.
        scale_factor: Coefficient to multiply eye distance for overlay width (default: 2.5)
        
    Returns:
        numpy.ndarray: Image with overlays applied
        
    Example output format from RetinaFace.detect_faces():
    {'face_1': {'score': 0.9995044469833374, 'facial_area': [244, 316, 282, 364],
                'landmarks': {'right_eye': [252.85898, 336.1748], 'left_eye': [270.33414, 335.65085],
                              'nose': [260.7771, 344.0823], 'mouth_right': [255.73444, 353.86722],
                              'mouth_left': [268.20093, 353.34763]}},
     'face_2': {'score': 0.9991769194602966, 'facial_area': [92, 286, 124, 327],
                'landmarks': {'right_eye': [101.63407, 302.39655], 'left_eye': [116.17397, 302.86218],
                              'nose': [108.91647, 310.6682], 'mouth_right': [103.07585, 317.63895],
                              'mouth_left': [113.791725, 318.00272]}}}
    """
    if output_path is None:
        output_path = image_path

    # Load the base image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Load the overlay image with alpha channel
    overlay = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        raise ValueError(f"Could not load overlay image from {overlay_image_path}")
    
    # Detect faces
    faces = RetinaFace.detect_faces(image_path)
    
    if not faces or not isinstance(faces, dict):
        print("No faces detected")
        if output_path:
            cv2.imwrite(output_path, img)
        return img
    
    # Process each detected face
    for face_key, face_data in faces.items():
        landmarks = face_data.get('landmarks', {})
        right_eye = landmarks.get('right_eye')  # [x, y]
        left_eye = landmarks.get('left_eye')    # [x, y]
        
        if not right_eye or not left_eye:
            print(f"Skipping {face_key}: missing eye landmarks")
            continue
        
        # Calculate angle between eyes
        dx = left_eye[0] - right_eye[0]
        dy = left_eye[1] - right_eye[1]
        angle = math.degrees(math.atan2(dy, dx))
        
        # Calculate distance between eyes
        eye_distance = math.sqrt(dx**2 + dy**2)
        
        # Calculate center point between eyes
        center_x = (right_eye[0] + left_eye[0]) / 2
        center_y = (right_eye[1] + left_eye[1]) / 2
        
        # Scale overlay to match eye distance
        # overlay width = scale_factor * eye_distance
        overlay_scale = (eye_distance * scale_factor) / overlay.shape[1]
        new_width = int(overlay.shape[1] * overlay_scale)
        new_height = int(overlay.shape[0] * overlay_scale)
        
        # Resize overlay
        overlay_resized = cv2.resize(overlay, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Rotate overlay around its center
        overlay_center = (new_width // 2, new_height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(overlay_center, angle, 1.0)
        
        # Calculate bounding box for rotated image
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((new_height * sin) + (new_width * cos))
        new_h = int((new_height * cos) + (new_width * sin))
        
        # Adjust rotation matrix for new dimensions
        rotation_matrix[0, 2] += (new_w / 2) - overlay_center[0]
        rotation_matrix[1, 2] += (new_h / 2) - overlay_center[1]
        
        # Rotate overlay
        if overlay_resized.shape[2] == 4:  # Has alpha channel
            overlay_rotated = cv2.warpAffine(overlay_resized, rotation_matrix, (new_w, new_h), 
                                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                                            borderValue=(0, 0, 0, 0))
        else:
            overlay_rotated = cv2.warpAffine(overlay_resized, rotation_matrix, (new_w, new_h), 
                                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                                            borderValue=(0, 0, 0))
        
        # Calculate position to place overlay (centered on eyes center)
        x_offset = int(center_x - new_w // 2)
        y_offset = int(center_y - new_h // 2)
        
        # Ensure overlay is within image bounds
        x1 = max(0, x_offset)
        y1 = max(0, y_offset)
        x2 = min(img.shape[1], x_offset + new_w)
        y2 = min(img.shape[0], y_offset + new_h)
        
        # Calculate corresponding overlay region
        overlay_x1 = x1 - x_offset
        overlay_y1 = y1 - y_offset
        overlay_x2 = overlay_x1 + (x2 - x1)
        overlay_y2 = overlay_y1 + (y2 - y1)
        
        # Extract regions
        img_region = img[y1:y2, x1:x2]
        overlay_region = overlay_rotated[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
        
        # Blend with alpha channel if available
        if overlay_region.shape[2] == 4:
            alpha = overlay_region[:, :, 3] / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            
            overlay_rgb = overlay_region[:, :, :3]
            blended = (overlay_rgb * alpha + img_region * (1 - alpha)).astype(np.uint8)
            img[y1:y2, x1:x2] = blended
        else:
            # No alpha channel, direct overlay
            img[y1:y2, x1:x2] = overlay_region
        
        print(f"Applied overlay on {face_key} at angle {angle:.2f}°")
    
    # Save or return result
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Saved result to {output_path}")
    
    return img


# overlay_on_eyes(image_path="C:\\hook-extraction-service\\src\\effect\\resources\\test_raw.jpg",
#                 overlay_image_path="C:\\hook-extraction-service\\src\\effect\\resources\\assets\\goggles.png",
#                 output_path="C:\\hook-extraction-service\\src\\effect\\resources\\test_raw_overlay.jpg", scale_factor=2.5)
