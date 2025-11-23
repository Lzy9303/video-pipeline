import shutil
import sys, os

import numpy as np
import torch, torchaudio
import cv2
import av
import uuid

# import get_mask
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(root, 'src'))
sys.path.insert(1, os.path.join(root, 'sam2'))

from sam2.build_sam import build_sam2_video_predictor
from bg_removal import get_mask_with_rembg_api, detect_banner_top_edge, generate_remove_items_mask, generate_goods_behind_persons_mask, generate_depth_image, analyze_scene
from llm.seed_vl import locate_banner
from llm.gemini import generate_cover

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = os.getenv("SAM2_CHECKPOINT", "model/checkpoints/sam2.1_hiera_large.pt")
model_cfg = os.getenv("SAM2_CONFIG", "model/configs/sam2.1/sam2.1_hiera_l.yaml")

if not os.path.exists(model_cfg):
    try:
        import sam2  # type: ignore

        _sam2_pkg = os.path.dirname(sam2.__file__)
        _candidate_cfg = os.path.join(_sam2_pkg, "configs", "sam2.1", "sam2.1_hiera_l.yaml")
        if os.path.exists(_candidate_cfg):
            model_cfg = _candidate_cfg
            print(f"[INFO] Using SAM2 config from package: {model_cfg}")
        else:
            print(f"[WARN] SAM2 config not found at default or package path. Set SAM2_CONFIG explicitly.")
    except Exception:
        print(f"[WARN] Could not import sam2 to locate config. Set SAM2_CONFIG explicitly.")

if not os.path.exists(sam2_checkpoint):
    print(
        f"[ERROR] SAM2 checkpoint not found at: {sam2_checkpoint}. Set SAM2_CHECKPOINT to your sam2.1_hiera_large.pt.")


def extract_first_frame_and_detect_banner(video_path, output_dir):
    """
    Extract the first frame from video and use detect_banner_top_edge to determine banner height.
    
    Returns:
        tuple: (first_frame_path, banner_height) or (first_frame_path, None) if detection fails
    """
    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None, None
        
        # Extract first frame
        ret, frame = video.read()
        video.release()
        
        if not ret:
            print("Error: Could not read first frame")
            return None, None
        
        # Get actual video dimensions
        actual_height, actual_width = frame.shape[:2]
        print(f"Actual video dimensions: {actual_width}x{actual_height}")
        
        # Save first frame temporarily
        first_frame_path = os.path.join(output_dir, "first_frame_temp.jpg")
        cv2.imwrite(first_frame_path, frame)
        
        # Use detect_banner_top_edge to analyze the image (retry up to 3 times)
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Banner detection attempt {attempt}/{max_retries}")
                result = detect_banner_top_edge(first_frame_path, coverage_threshold=0.99)
                print(f"Banner detection result: {result}")
                
                if result.get("isValid") and "y1" in result:
                    y1 = result["y1"]
                    # Calculate banner height: video_height - y1
                    banner_height = actual_height - y1
                    print(f"Detected banner y1: {y1}, banner height: {banner_height}")
                    return first_frame_path, banner_height
                else:
                    print(f"Banner detection invalid: {result.get('reason', 'unknown')}")
                    if attempt < max_retries:
                        print(f"Retrying... ({attempt}/{max_retries})")
                        continue
                    else:
                        return first_frame_path, None
            except Exception as e:
                print(f"Error during banner detection attempt {attempt}: {str(e)}")
                import traceback
                traceback.print_exc()
                if attempt < max_retries:
                    print(f"Retrying... ({attempt}/{max_retries})")
                    continue
                else:
                    return first_frame_path, None
    
    except Exception as e:
        print(f"Error extracting first frame: {str(e)}")
        return None, None


def extract_frames(video_path, output_dir, start_time=None, end_time=None, prefix=''):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")

    # Calculate start and end frames from time values if provided
    start_frame = 0
    if start_time is not None:
        start_frame = max(0, int(start_time * fps))

    end_frame = total_frames - 1
    if end_time is not None:
        end_frame = min(total_frames - 1, int(end_time * fps))

    # Set the starting position in the video
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = start_frame
    while frame_count <= end_frame:
        # Read a frame
        success, frame = video.read()
        if not success:
            break

        # Save the frame as an image
        frame_path = os.path.join(output_dir, f"{prefix}{frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    # Release the video file
    video.release()
    frames_extracted = (end_frame - start_frame + 1)
    print(f"Extracted {frames_extracted} frames to {output_dir}")
    return frames_extracted

def save_segments_to_video(video_segments, video_dir, output_video_dir, frame_names, vis_frame_stride=1, fps=30, bg_color=(255, 255, 255), crop_height=None, external_bg_image=None, external_crop_image=None):
    try:
        # Ensure output directory exists
        os.makedirs(output_video_dir, exist_ok=True)

        # Create output video file with unique name
        video_uuid = str(uuid.uuid4())
        output_path = os.path.join(output_video_dir, f"{video_uuid}.mp4")
        output = av.open(output_path, mode='w')

        # Get video parameters from first frame
        first_frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
        height, width = first_frame.shape[:2]

        # Create output stream
        stream = output.add_stream('h264', rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'

        # Prepare background components if provided
        cropped_area = None
        external_bg = None
        
        # Load external crop image if provided
        if external_crop_image is not None and crop_height is not None:
            # Load external image for cropped area
            external_crop = cv2.imread(external_crop_image)
            if external_crop is not None:
                external_crop = cv2.cvtColor(external_crop, cv2.COLOR_BGR2RGB)
                # Resize to fit within video width and crop_height while preserving aspect ratio
                crop_img_h, crop_img_w = external_crop.shape[:2]
                aspect_ratio = crop_img_h / crop_img_w
                
                # Calculate scaling ratios for both width and height constraints
                scale_by_width = width / crop_img_w
                scale_by_height = crop_height / crop_img_h
                
                # Use the larger scale ratio to fill the area
                scale_ratio = max(scale_by_width, scale_by_height)
                
                # Calculate the scaled dimensions
                scaled_width = int(crop_img_w * scale_ratio)
                scaled_height = int(crop_img_h * scale_ratio)
                
                # Resize the image with the calculated scale ratio
                scaled_crop = cv2.resize(external_crop, (scaled_width, scaled_height))
                
                # Crop the area to fit within the designated bounds (width x crop_height)
                # Center horizontally and vertically
                x_offset = max(0, (scaled_width - width) // 2)
                y_offset = max(0, (scaled_height - crop_height) // 2)
                
                cropped_area = scaled_crop[y_offset:y_offset + crop_height, x_offset:x_offset + width]
                
                # If the cropped area is smaller than the designated size, pad it
                if cropped_area.shape[0] < crop_height or cropped_area.shape[1] < width:
                    canvas = np.ones((crop_height, width, 3), dtype=np.uint8)
                    canvas[..., 0] = bg_color[0]
                    canvas[..., 1] = bg_color[1]
                    canvas[..., 2] = bg_color[2]
                    
                    # Place the cropped area in the center
                    h, w = cropped_area.shape[:2]
                    y_pos = (crop_height - h) // 2
                    x_pos = (width - w) // 2
                    canvas[y_pos:y_pos + h, x_pos:x_pos + w] = cropped_area
                    cropped_area = canvas
        
        if external_bg_image is not None:
            # Load and resize external image by width only
            external_bg_full = cv2.imread(external_bg_image)
            if external_bg_full is not None:
                external_bg_full = cv2.cvtColor(external_bg_full, cv2.COLOR_BGR2RGB)
                # Resize to match video width, preserving aspect ratio
                aspect_ratio = external_bg_full.shape[0] / external_bg_full.shape[1]
                external_bg_height = int(width * aspect_ratio)
                external_bg = cv2.resize(external_bg_full, (width, external_bg_height))

        try:
            # Process and write each frame
            for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
                # Read original frame
                frame_path = os.path.join(video_dir, frame_names[out_frame_idx])
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Create composite background - start with white
                bg = np.ones_like(frame)
                bg[..., 0] = bg_color[0]
                bg[..., 1] = bg_color[1]
                bg[..., 2] = bg_color[2]
                
                # Place external background image at the top if provided
                if external_bg is not None:
                    ext_h, ext_w = external_bg.shape[:2]
                    bg[:ext_h, :ext_w] = external_bg
                
                # Place cropped area at the bottom if provided
                if cropped_area is not None:
                    crop_h, crop_w = cropped_area.shape[:2]
                    bg[-crop_h:, :crop_w] = cropped_area

                # Apply all masks for this frame
                for _, out_mask in video_segments[out_frame_idx].items():
                    # Reshape mask and apply it
                    h, w = out_mask.shape[-2:]
                    mask = out_mask.reshape(h, w, 1)
                    frame = np.where(mask, frame, bg)

                # Place cropped area on top of the frame (after mask application)
                if cropped_area is not None:
                    crop_h, crop_w = cropped_area.shape[:2]
                    frame[-crop_h:, :crop_w] = cropped_area

                # Convert to PyAV frame and write
                av_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
                packet = stream.encode(av_frame)
                output.mux(packet)

            # Flush remaining frames
            packet = stream.encode(None)
            output.mux(packet)

            print(f"Video saved to {output_path} with frame rate of {fps} fps")

        finally:
            output.close()

    except Exception as e:
        print(f"Error creating video: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)

def process_video(input_video_path, output_video_dir, start=None, end=None, bg_color=(255, 255, 255),
                  vis_frame_stride=1, fps=30, cover_image_path = "./test_resources/cover.png"):
    predictor = None
    video = None
    try:
        # Build SAM2 predictor; pass device as string for compatibility
        try:
            predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device.type)
        except Exception as e:
            import traceback
            print("[ERROR] Failed to build SAM2 predictor.\n"
                  f"  CONFIG: {model_cfg}\n  CHECKPOINT: {sam2_checkpoint}\n  DEVICE: {device.type}\n"
                  f"  Exception: {e}")
            traceback.print_exc()
            raise
        video_frame_dir = "videos/frames"

        # Create the video frame directory if it doesn't exist
        if os.path.exists(video_frame_dir):
            shutil.rmtree(video_frame_dir)
        os.makedirs(video_frame_dir)

        try:
            extract_frames(input_video_path, video_frame_dir, start, end)

            # scan all the JPEG frame names in this directory
            frame_names = [
                p for p in os.listdir(video_frame_dir)
                if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
            ]
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

            video = cv2.VideoCapture(input_video_path)

            inference_state = predictor.init_state(video_path=video_frame_dir)
            predictor.reset_state(inference_state)

            ann_frame_idx = 0
            ann_obj_id = 1

            # Build seed mask for the annotation frame (remove.bg or GrabCut fallback)
            # Try frames until we get a valid mask (step by 1)
            seed_mask = None
            mask_frame_idx = 0
            rembg_mask = None
            
            depth_image_path = os.path.join(video_frame_dir, f"depth_{frame_names[mask_frame_idx]}")
            print(f"Generating depth image for frame {mask_frame_idx}...")
            generate_depth_image(os.path.join(video_frame_dir, frame_names[mask_frame_idx]), output_path=depth_image_path)
            print(f"Depth image saved to {depth_image_path}")

            # do the scene analysis
            scene_analysis = analyze_scene(os.path.join(video_frame_dir, frame_names[mask_frame_idx]), depth_image_path)
            print(f"Scene analysis: {scene_analysis}")
            if scene_analysis.get("sceneType") != "studio" or not scene_analysis.get("hasBanner", False):
                print("Skipping video: not a studio scene or no banner detected")
                return
                
            for frame_idx in range(0, len(frame_names), 1):
                frame_path = os.path.join(video_frame_dir, frame_names[frame_idx])
                mask = get_mask_with_rembg_api(frame_path)
                
                if mask is not None:
                    seed_mask = mask
                    rembg_mask = mask
                    mask_frame_idx = frame_idx
                    print(f"Got valid mask from frame {frame_idx}")
                    break
            
            if seed_mask is None:
                raise ValueError("Could not get valid mask from remove.bg API")
            else:
                # If rembg succeeded, try to get gemini mask and merge them
                max_retries = 3
                merge_success = False
                
                for attempt in range(1, max_retries + 1):
                    try:
                        print(f"Attempting to merge rembg mask with sa2va-generated mask (attempt {attempt}/{max_retries})")
                        
                        # Import sa2va get_masks and get_all_objects_and_persons_with_depth
                        from sa2va.inference import get_masks
                        from bg_removal.bg_removal_gemini import get_all_objects_and_persons_with_depth
                        
                        # Generate the depth image and save it alongside the frame
                        depth_image_path = os.path.join(video_frame_dir, f"depth_{frame_names[frame_idx]}")
                        print(f"Generating depth image for frame {frame_idx}...")
                        generate_depth_image(frame_path, output_path=depth_image_path)
                        print(f"Depth image saved to {depth_image_path}")

                        # Get all objects and persons (excluding items behind persons)
                        print("Detecting objects and persons...")
                        all_objects = get_all_objects_and_persons_with_depth(
                            image_path=frame_path, 
                            depth_image_path=depth_image_path,
                            output_path=None
                        )
                        
                        if not all_objects:
                            print("No objects detected, falling back to rembg mask only")
                            gemini_mask = None
                        else:
                            # Filter to only keep front objects/persons (not behind)
                            front_objects = [obj for obj in all_objects if not obj.get('is_behind_persons', False)]
                            print(f"Found {len(front_objects)} front objects/persons to segment")

                            back_objects = [obj for obj in all_objects if obj.get('is_behind_persons', True)]

                            # Calculate crop_height: gap between lowest edge of front items and bottom of image - 10px
                            frame_img = cv2.imread(frame_path)
                            img_height, img_width = frame_img.shape[:2]
                            
                            if front_objects:
                                print(front_objects)
                                # Find the lowest y1 (bottom edge) among all front objects
                                lowest_y1 = max(obj['bbox_2d'][3] for obj in front_objects)
                                crop_height = img_height - lowest_y1 - 10
                                crop_height = max(0, crop_height)  # Ensure non-negative
                                print(f"Calculated crop_height: {crop_height}px (image height: {img_height}, lowest edge: {lowest_y1})")
                            else:
                                crop_height = 0
                            
                            if not front_objects:
                                print("No front objects found, falling back to rembg mask only")
                                gemini_mask = None
                            else:
                                # Run sa2va get_masks on the detected objects
                                print("Running sa2va segmentation on detected objects...")
                                mask_results = get_masks(frame_path, front_objects)
                                
                                # Merge all masks from all targets into a single mask
                                frame_img = cv2.imread(frame_path)
                                img_height, img_width = frame_img.shape[:2]
                                merged_sa2va_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                                
                                for result in mask_results['results']:
                                    for mask in result['masks']:
                                        merged_sa2va_mask = np.maximum(merged_sa2va_mask, mask)
                                
                                gemini_mask = merged_sa2va_mask
                                print(f"Sa2va segmentation completed, merged {sum(len(r['masks']) for r in mask_results['results'])} masks")
                        
                        if gemini_mask is not None:
                            # Convert both masks to numpy arrays if needed
                            rembg_array = np.array(rembg_mask)
                            gemini_array = np.array(gemini_mask)
                            
                            # Ensure both are binary
                            rembg_array = (rembg_array > 0.5).astype(np.uint8)
                            gemini_array = (gemini_array > 0.5).astype(np.uint8)
                            
                            # Resize gemini_array to match rembg_array dimensions if needed
                            if gemini_array.shape != rembg_array.shape:
                                print(f"Resizing sa2va mask from {gemini_array.shape} to {rembg_array.shape}")
                                gemini_array = cv2.resize(gemini_array, (rembg_array.shape[1], rembg_array.shape[0]), 
                                                         interpolation=cv2.INTER_NEAREST)
                            
                            # Initialize merged_mask with rembg as fallback
                            merged_mask = rembg_array.copy()
                            
                            # Remove areas of back_objects from the merged_mask
                            if back_objects:
                                print(f"Removing {len(back_objects)} back objects from mask")
                                for back_obj in back_objects:
                                    x0, y0, x1, y1 = back_obj['bbox_2d']
                                    print(f"Removing back object at {x0},{y0} to {x1},{y1}")
                                    # Set the bounding box area to 0 (remove from mask)
                                    merged_mask[y0:y1, x0:x1] = 0
                                print("Back objects removed from mask")
                            
                            merged_mask = np.logical_or(merged_mask, gemini_array).astype(np.uint8)

                            seed_mask = merged_mask
                            print("Masks merged successfully")
                            merge_success = True

                            # Save merged mask visualization
                            try:
                                frame_img = cv2.imread(frame_path)
                                if frame_img is not None:
                                    # Create visualization with mask overlay
                                    mask_vis = np.zeros_like(frame_img)
                                    mask_vis[merged_mask > 0] = [0, 255, 0]  # Green for masked area
                                    overlay = cv2.addWeighted(frame_img, 0.7, mask_vis, 0.3, 0)
                                    
                                    # Draw back_objects bounding boxes in red
                                    if back_objects:
                                        for back_obj in back_objects:
                                            x0, y0, x1, y1 = back_obj['bbox_2d']
                                            label = back_obj.get('label', 'unknown')
                                            # Draw red rectangle for back objects
                                            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 2)
                                            # Draw label with background
                                            label_text = f"{label} (back)"
                                            font = cv2.FONT_HERSHEY_SIMPLEX
                                            font_scale = 0.5
                                            font_thickness = 1
                                            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
                                            # Draw filled rectangle for text background
                                            cv2.rectangle(overlay, (x0, y0 - text_height - baseline - 5), (x0 + text_width, y0), (0, 0, 255), -1)
                                            # Draw text
                                            cv2.putText(overlay, label_text, (x0, y0 - baseline - 2), font, font_scale, (255, 255, 255), font_thickness)
                                    
                                    # Save the visualization
                                    mask_output_path = os.path.join(output_video_dir, f"seed_mask_merged_frame_{mask_frame_idx}.jpg")
                                    cv2.imwrite(mask_output_path, overlay)
                                    print(f"Saved merged mask visualization to: {mask_output_path}")
                            except Exception as e:
                                print(f"Failed to save mask visualization: {str(e)}")
                            
                            break
                        else:
                            # No sa2va mask generated, will retry or fall back to rembg
                            print("Sa2va mask is None, will retry or use rembg mask only")
                            if attempt >= max_retries:
                                seed_mask = rembg_mask
                                break
                    except Exception as e:
                        print(f"Attempt {attempt}/{max_retries} failed to merge masks: {str(e)}")
                        if attempt < max_retries:
                            print(f"Retrying... ({max_retries - attempt} attempts remaining)")
                        else:
                            print("All merge attempts failed, using rembg mask only")
                            seed_mask = rembg_mask
                
                if not merge_success:
                    print("⚠️  WARNING: Mask merge failed after all retries. Using rembg mask only without gemini enhancement and item removal.")
                    seed_mask = rembg_mask

            # Ensure mask matches first frame dimensions (safety)
            img0 = cv2.imread(os.path.join(video_frame_dir, frame_names[ann_frame_idx]))
            if img0 is None:
                raise ValueError(f"Failed to read seed frame: {os.path.join(video_frame_dir, frame_names[ann_frame_idx])}")
            mh, mw = seed_mask.shape[:2]
            ih, iw = img0.shape[:2]
            if (mh, mw) != (ih, iw):
                seed_mask = cv2.resize(seed_mask, (iw, ih), interpolation=cv2.INTER_NEAREST)
            # Ensure binary mask (0/1)
            seed_mask = (seed_mask > 0.5).astype(np.uint8)

            # Use autocast on CUDA for performance
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            )
            with autocast_ctx:
                _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    mask=seed_mask,
                )

            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            # render the segmentation results and save as video
            save_segments_to_video(
                video_segments,
                video_frame_dir,
                output_video_dir,
                frame_names,
                vis_frame_stride=vis_frame_stride,
                fps=fps,
                bg_color=bg_color,
                crop_height=crop_height,
                external_bg_image= cover_image_path,
                external_crop_image="./test_resources/banner.png"
            )
            return output_video_dir

        finally:
            # Clean up frame directory
            if os.path.exists(video_frame_dir):
                print("Cleaning up frame directory...")
                #shutil.rmtree(video_frame_dir)

    finally:
        # Clean up resources
        if video is not None:
            video.release()
        if predictor is not None:
            del predictor
        if device.type == "cuda":
            torch.cuda.empty_cache()

