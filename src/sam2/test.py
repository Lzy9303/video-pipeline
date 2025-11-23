import sys, os
from pathlib import Path

from src.llm import generate_cover

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.sam2.sam2 import process_video
from src.transition import VideoTransitionDetector
from src.effect import overlay_on_eyes
import uuid
import av
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed


def trim_video(input_video, output_video, start_frame, end_frame, fps=30):
    """Trim video to specified frame range by extracting frames and regenerating
    
    Args:
        input_video: Input video path
        output_video: Output video path
        start_frame: Start frame number (inclusive)
        end_frame: End frame number (exclusive)
        fps: Video framerate
    """
    # Create temporary directory for frames
    temp_frames_dir = output_video.replace('.mp4', '_frames')
    os.makedirs(temp_frames_dir, exist_ok=True)
    
    try:
        # Extract frames from start_frame to end_frame
        container = av.open(input_video)
        stream = container.streams.video[0]
        
        frame_idx = 0
        output_frame_idx = 0
        
        for frame in container.decode(stream):
            if frame_idx >= end_frame:
                break
            
            if frame_idx >= start_frame:
                frame_image = frame.to_image()
                frame_path = os.path.join(temp_frames_dir, f"frame_{output_frame_idx:06d}.jpg")
                frame_image.save(frame_path)
                output_frame_idx += 1
            
            frame_idx += 1
        
        container.close()
        
        # Create video from extracted frames
        if output_frame_idx > 0:
            cmd = [
                'ffmpeg', '-framerate', str(fps), '-i', os.path.join(temp_frames_dir, 'frame_%06d.jpg'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', output_video
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        
    finally:
        # Clean up temporary frames directory
        if os.path.exists(temp_frames_dir):
            import shutil
            shutil.rmtree(temp_frames_dir)


def extract_first_frame(video_path, output_frame_path):
    """Extract the first frame from a video"""
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', 'select=eq(n\\,0)',
        '-q:v', '2',
        '-y', output_frame_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def concat_videos(video_list, output_video):
    """Concatenate multiple videos using ffmpeg"""
    # Create concat demuxer file
    concat_file = output_video.replace('.mp4', '_concat.txt')
    with open(concat_file, 'w') as f:
        for video in video_list:
            f.write(f"file '{os.path.abspath(video)}'\n")
    
    # Run ffmpeg concat
    cmd = [
        'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file,
        '-c', 'copy', '-y', output_video
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    os.remove(concat_file)


def dump_source_clips(input_video, output_dir, clip_frames, fps=30):
    """Dump all source clips first and extract first frame
    
    Args:
        input_video: Input video path
        output_dir: Output directory
        clip_frames: List of (start_frame, end_frame) tuples
        fps: Video framerate
    """
    print(f"\nDumping {len(clip_frames)} source clips...")
    clip_videos = []
    
    for idx, (start_frame, end_frame) in enumerate(clip_frames):
        # Create clip directory
        clip_dir = os.path.join(output_dir, f"clip_{idx:03d}")
        os.makedirs(clip_dir, exist_ok=True)
        
        # Trim video to clip
        clip_video = os.path.join(clip_dir, f"clip_{idx:03d}_source.mp4")
        start_time = start_frame / fps
        end_time = end_frame / fps
        print(f"  Dumping clip {idx + 1}/{len(clip_frames)}: frames {start_frame}-{end_frame} ({start_time:.2f}s - {end_time:.2f}s)")
        trim_video(input_video, clip_video, start_frame, end_frame, fps)
        
        # Extract first frame
        first_frame_path = os.path.join(clip_dir, f"clip_{idx:03d}_first_frame.jpg")
        try:
            extract_first_frame(clip_video, first_frame_path)
            print(f"    First frame extracted: {first_frame_path}")
        except Exception as e:
            print(f"    Warning: Failed to extract first frame: {str(e)}")
        
        clip_videos.append((idx, clip_dir, clip_video))
    
    print(f"All {len(clip_videos)} source clips dumped")
    return clip_videos

def process_clips(clip_videos, cover_image_path):
    """Process each clip one by one"""
    print(f"\nProcessing {len(clip_videos)} clips...")
    processed_videos = []
    failed_clips = []
    
    for idx, clip_dir, clip_video in clip_videos:
        print(f"\nProcessing clip {idx + 1}/{len(clip_videos)}: {clip_video}")
        
        try:
            # Process clip with SAM2
            process_video(clip_video, clip_dir, cover_image_path=cover_image_path)
            
            # Find the output video
            output_videos = [f for f in os.listdir(clip_dir) if f.endswith('.mp4') and f != os.path.basename(clip_video)]
            if output_videos:
                processed_video = os.path.join(clip_dir, output_videos[0])
                processed_videos.append(processed_video)
                print(f"Clip {idx + 1} processed: {processed_video}")
                # Clean up source clip
                os.remove(clip_video)
            
            
            
        except Exception as e:
            print(f"ERROR processing clip {idx + 1}: {str(e)}")
            failed_clips.append((idx, str(e)))
            print(f"Skipping clip {idx + 1} and moving to next...")
            # Leave the clip untouched - source video and directory remain
    
    if failed_clips:
        print(f"\n{len(failed_clips)} clip(s) failed to process:")
        for idx, error in failed_clips:
            print(f"  Clip {idx + 1}: {error}")
    
    return processed_videos


def dump_video_frames(video_path, frames_dir, start_frame_counter, overlay_image_path=None):
    """Extract and save all frames from a video file.
    
    Args:
        video_path: Path to the input video file
        frames_dir: Directory to save extracted frames
        start_frame_counter: Starting frame number for sequential naming
        overlay_image_path: Optional path to overlay image for face detection (e.g., goggles, censor bar)
        scale_factor: Scale factor for overlay relative to eye distance (default: 2.5)
        
    Returns:
        int: Number of frames extracted
    """
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        frames = []
        
        for frame in container.decode(stream):
            frames.append(frame)
        
        container.close()
        
        # Check if video has frames
        if len(frames) == 0:
            print(f"    Warning: Video has no frames, skipping")
            return 0
        
        # Save all frames
        frame_counter = start_frame_counter
        for idx, frame in enumerate(frames):
            frame_image = frame.to_image()
            frame_path = os.path.join(frames_dir, f"frame_{frame_counter:06d}.jpg")
            frame_image.save(frame_path)
            
            # Apply overlay if provided
            if overlay_image_path:
                try:
                    overlay_on_eyes(
                        image_path=frame_path,
                        overlay_image_path=overlay_image_path
                    )
                except Exception as e:
                    print(f"    Warning: Failed to apply overlay on frame {frame_counter}: {str(e)}")
            
            frame_counter += 1
        
        frames_extracted = len(frames)
        print(f"    Extracted {frames_extracted} frames")
        return frames_extracted
    
    except Exception as e:
        print(f"    Error processing video: {str(e)}")
        return 0


def concat_all_outputs_frame_based(output_dir, framerate=24):
    """Concatenate all outputs by extracting frames, removing head/tail, and creating new video
    
    Args:
        output_dir: Directory containing clip subdirectories
        framerate: Output video framerate (default: 24)
    """
    print(f"\nConcatenating outputs using frame-based approach in {output_dir}...")
    print(f"Output framerate: {framerate} fps")
    
    # Find all clip directories
    clip_dirs = sorted([d for d in os.listdir(output_dir) if d.startswith('clip_')])
    
    # Create temporary directory for frames
    frames_dir = os.path.join(output_dir, "all_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    frame_counter = 0
    
    for idx, clip_dir_name in enumerate(clip_dirs):
        clip_dir = os.path.join(output_dir, clip_dir_name)
        if not os.path.isdir(clip_dir):
            continue
        
        # Find first .mp4 file in this clip directory
        mp4_files = [f for f in os.listdir(clip_dir) if f.endswith('.mp4')]
        if not mp4_files:
            continue
        
        mp4_files.sort()
        video_path = os.path.join(clip_dir, mp4_files[0])
        print(f"  Processing: {mp4_files[0]} from {clip_dir_name}")
        
        # Extract and save frames from this clip
        if idx == 0:
            frames_extracted = dump_video_frames(video_path, frames_dir, frame_counter, overlay_image_path="C:\\hook-extraction-service\\src\\effect\\resources\\assets\\goggles.png")
        else:
            frames_extracted = dump_video_frames(video_path, frames_dir, frame_counter)

        frame_counter += frames_extracted
    
    # Create video from all frames
    if frame_counter > 0:
        print(f"\nCreating video from {frame_counter} frames...")
        final_output = os.path.join(output_dir, "final_output_frame_based.mp4")
        
        # Get audio from first clip
        audio_source = None
        clip_dirs = sorted([d for d in os.listdir(output_dir) if d.startswith('clip_')])
        for clip_dir_name in clip_dirs:
            clip_dir = os.path.join(output_dir, clip_dir_name)
            mp4_files = [f for f in os.listdir(clip_dir) if f.endswith('.mp4')]
            if mp4_files:
                mp4_files.sort()
                audio_source = os.path.join(clip_dir, mp4_files[0])
                print(f"Using audio from: {audio_source}")
                break

        # No audio source, create video without audio
        cmd = [
                'ffmpeg', '-framerate', str(framerate), '-i', os.path.join(frames_dir, 'frame_%06d.jpg'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', final_output
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Final output saved to: {final_output}")
            return final_output
        except Exception as e:
            print(f"Error creating video: {str(e)}")
            return None

    else:
        print("No frames extracted to create video")
        return None


def concat_all_outputs(output_dir):
    """Concatenate all first .mp4 files found in each clip directory"""
    print(f"\nFinding all output videos in {output_dir}...")
    processed_videos = []
    
    # Find all clip directories
    clip_dirs = sorted([d for d in os.listdir(output_dir) if d.startswith('clip_')])
    
    for clip_dir_name in clip_dirs:
        clip_dir = os.path.join(output_dir, clip_dir_name)
        if not os.path.isdir(clip_dir):
            continue
        
        # Find first .mp4 file in this clip directory
        mp4_files = [f for f in os.listdir(clip_dir) if f.endswith('.mp4')]
        if mp4_files:
            # Sort to get consistent ordering
            mp4_files.sort()
            processed_video = os.path.join(clip_dir, mp4_files[0])
            processed_videos.append(processed_video)
            print(f"  Found: {mp4_files[0]} in {clip_dir_name}")
    
    if processed_videos:
        print(f"\nConcatenating {len(processed_videos)} videos...")
        final_output = os.path.join(output_dir, "final_output.mp4")
        concat_videos(processed_videos, final_output)
        print(f"Final output saved to: {final_output}")
        return final_output
    else:
        print("No output videos found to concatenate")
        return None


def test_with_transitions(input_video:str = "./test_resources/input.mp4", time_to_extract_cover = 0.0):
    """Process video by breaking it down into clips based on detected transitions"""
    
    # Create main output directory
    test_id = str(uuid.uuid4())
    output_dir = os.path.join("./test_result", test_id)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing video: {input_video}")
    print(f"Output directory: {output_dir}")
    
    # Detect transitions
    detector = VideoTransitionDetector(
        movement_threshold=0.6,
        scene_change_threshold=0.6,
        lighting_threshold=0.6,
        min_transition_gap=0.3
    )
    
    result = detector.detect_transitions(
        video_path=input_video,
        sample_rate=1,
        max_transitions=500,
        dump_frames=False
    )
    
    if not result.success:
        print(f"Error detecting transitions: {result.error}")
        return
    
    print(f"Found {len(result.transitions) if result.transitions else 0} transitions")
    
    # Get video properties
    container = av.open(input_video)
    stream = container.streams.video[0]
    fps = int(stream.average_rate)
    total_frames = stream.frames  # Get frame count directly from stream
    total_duration = result.duration
    container.close()
    
    # Build clip frame ranges from transitions (using frame numbers for accuracy)
    # Frame ranges use exclusive end indices: [start, end) means frames start to end-1
    clip_frames = []
    if result.transitions:
        # First clip: from frame 0 to first transition (exclusive)
        clip_frames.append((0, int(result.transitions[0].frame_number)))
        
        # Middle clips: between transitions (exclusive end)
        for i in range(len(result.transitions) - 1):
            clip_frames.append((int(result.transitions[i].frame_number), int(result.transitions[i + 1].frame_number)))
        
        # Last clip: from last transition to end (exclusive)
        clip_frames.append((int(result.transitions[-1].frame_number), total_frames))
    else:
        # No transitions: use entire video
        clip_frames.append((0, total_frames))
    
    print(f"Created {len(clip_frames)} clips from transitions (using frame numbers)")
    
    # Extract audio from original video once
    print("\nExtracting audio from original video...")
    audio_file = os.path.join(output_dir, "original_audio.aac")
    try:
        cmd_extract_audio = [
            'ffmpeg', '-i', input_video,
            '-q:a', '9', '-map', 'a', '-y', audio_file
        ]
        subprocess.run(cmd_extract_audio, check=True, capture_output=True)
        print(f"Audio extracted: {audio_file}")
    except Exception as e:
        print(f"Warning: Failed to extract audio: {str(e)}")
        audio_file = None

    # Extract a single frame at time_to_extract_cover using ffmpeg
    cover_frame_path = os.path.join(output_dir, f"cover_{time_to_extract_cover:.2f}s.jpg")
    ffmpeg_cmd = f'ffmpeg -ss {time_to_extract_cover} -i "{input_video}" -frames:v 1 -q:v 2 "{cover_frame_path}" -y'
    print(f"Extracting cover frame at {time_to_extract_cover}s using ffmpeg...")
    os.system(ffmpeg_cmd)

    # Generate cover image using Gemini
    print(f"Generating cover image from: {cover_frame_path}")
    generated_cover_path = generate_cover(cover_frame_path)
    print(f"Cover image generated: {generated_cover_path}")

    if not os.path.exists(cover_frame_path):
        raise ValueError(f"Failed to extract cover frame at {time_to_extract_cover}s")
    
    # Step 1: Dump all source clips (using frame numbers for accuracy)
    clip_videos = dump_source_clips(input_video, output_dir, clip_frames, fps)
    
    # Step 2: Process each clip
    processed_videos = process_clips(clip_videos, generated_cover_path)
    
    # Step 3: Concatenate all outputs
    # TODO: use actual framerate
    concat_all_outputs_frame_based(output_dir)
    
    # Step 4: Attach audio to final concatenated video
    if audio_file and os.path.exists(audio_file):
        print("\nAttaching audio to final output...")
        final_output = os.path.join(output_dir, "final_output_frame_based.mp4")
        final_output_with_audio = os.path.join(output_dir, "final_output_with_audio.mp4")
        
        if os.path.exists(final_output):
            try:
                cmd_attach_audio = [
                    'ffmpeg', '-i', final_output, '-i', audio_file,
                    '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
                    '-shortest', '-y', final_output_with_audio
                ]
                subprocess.run(cmd_attach_audio, check=True, capture_output=True)
                
                # Replace original with audio-attached version
                # os.remove(final_output)
                # os.rename(final_output_with_audio, final_output)
                print(f"Audio attached to: {final_output_with_audio}")
            except Exception as e:
                print(f"Error attaching audio: {str(e)}")
        
        # Clean up audio file
        # os.remove(audio_file)
    
    print(f"\nAll results saved to: {output_dir}")


# Run the test
test_with_transitions(24)

# process_video("C:\\hook-extraction-service\\src\\sam2\\test_result\\92bcc72f-c217-48cb-b008-f98f089a174f\\clip_009\\clip_009_source.mp4",
#     "C:\\hook-extraction-service\\src\\sam2\\test_result\\92bcc72f-c217-48cb-b008-f98f089a174f\\clip_009")

# concat_all_outputs_frame_based("D:\\hook-extraction-service\\src\\sam2\\test_result\\155a6ee5-a30e-4212-b734-a67f3aeb4772")

def test_overlay():
    frames_dir = "C:\\hook-extraction-service\\src\\sam2\\test_result\\bb4c8fe6-e7fc-4191-942c-96639a7a257d\\clip_000\\test"
    output_video = os.path.join(frames_dir, "output_with_overlay.mp4")
    
    # Extract frames and apply overlay
    dump_video_frames(
        video_path="C:\\hook-extraction-service\\src\\sam2\\test_result\\bb4c8fe6-e7fc-4191-942c-96639a7a257d\\clip_000\\clip_000_source.mp4",
        frames_dir=frames_dir,
        start_frame_counter=0,
        overlay_image_path="C:\\hook-extraction-service\\src\\effect\\resources\\assets\\goggles.png"
    )
    
    # Concatenate frames into MP4
    print(f"\nCreating video from frames in {frames_dir}...")
    frame_pattern = os.path.join(frames_dir, "frame_%06d.jpg")
    
    cmd = [
        'ffmpeg', '-y',
        '-framerate', '24',
        '-i', frame_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        output_video
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Video created successfully: {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e.stderr.decode()}")
    except Exception as e:
        print(f"Error creating video: {str(e)}")

# test_overlay()
# concat_all_outputs_frame_based("C:\\hook-extraction-service\\src\\sam2\\test_result\\faca8eab-735f-4bce-b3f6-a2416f991efa")


# output_dir="C:\\hook-extraction-service\\src\\sam2\\test_result\\faca8eab-735f-4bce-b3f6-a2416f991efa"
# audio_file="C:\\hook-extraction-service\\src\\sam2\\test_result\\faca8eab-735f-4bce-b3f6-a2416f991efa\\original_audio.aac"
# final_output = os.path.join(output_dir, "final_output_frame_based.mp4")
# final_output_with_audio = os.path.join(output_dir, "final_output_with_audio.mp4")
#
# if os.path.exists(final_output):
#     try:
#         cmd_attach_audio = [
#             'ffmpeg', '-i', final_output, '-i', audio_file,
#             '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
#             '-shortest', '-y', final_output_with_audio
#         ]
#         subprocess.run(cmd_attach_audio, check=True, capture_output=True)
#
#         # Replace original with audio-attached version
#         # os.remove(final_output)
#         # os.rename(final_output_with_audio, final_output)
#         print(f"Audio attached to: {final_output_with_audio}")
#     except Exception as e:
#         print(f"Error attaching audio: {str(e)}")


