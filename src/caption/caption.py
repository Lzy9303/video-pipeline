import json
import uuid

from moviepy.config import change_settings
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"})

# TODO: replace with own font file and magick executable
def add_captions_to_video(video_path, clauses_json_path, output_path, fontsize=20, color='white', font = 'c:\\\\simhei.ttf'):
    """
    Add captions to a video based on clauses from a JSON file.
    
    Args:
        video_path: Path to the input video file
        clauses_json_path: Path to JSON file containing clauses with 'text', 'start', 'end' keys
        output_path: Path to save the output video with captions
        fontsize: Font size for captions (default: 70)
        color: Text color for captions (default: 'yellow')
        font: Font family for captions (default: 'Impact')
        
    Returns:
        str: Path to the output video file
        
    Example JSON format:
    [
        {"text": "Hello world", "start": 0, "end": 2000},
        {"text": "Second caption", "start": 2000, "end": 4000}
    ]
    """
    # Load clauses from JSON file
    print(f"Loading clauses from: {clauses_json_path}")
    with open(clauses_json_path, 'r', encoding='utf-8') as f:
        clauses = json.load(f)
    
    print(f"Loaded {len(clauses)} clauses")
    
    # Load the video
    print(f"Loading video: {video_path}")
    video_clip = VideoFileClip(video_path)
    clips = [video_clip]
    
    # Create text clips for each clause
    for idx, clause in enumerate(clauses):
        text = clause.get('text', '')
        start_ms = clause.get('start', 0)
        end_ms = clause.get('end', 0)
        
        # Convert milliseconds to seconds
        start_sec = start_ms / 1000
        duration_sec = (end_ms - start_ms) / 1000
        
        if duration_sec <= 0:
            print(f"Warning: Skipping clause {idx} with invalid duration")
            continue
        
        # Skip if text is empty
        if not text.strip():
            print(f"Warning: Skipping clause {idx} with empty text")
            continue
        
        print(f"Adding caption {idx + 1}/{len(clauses)}: '{text}' at {start_sec:.2f}s for {duration_sec:.2f}s")

        scale_factor = 5

        try:
            # Create text with stroke for better visibility
            # Using Chinese-compatible font (Microsoft YaHei)
            txt = TextClip(
                text,  # Original Chinese text
                fontsize=fontsize * scale_factor,
                color=color, 
                font=font,
                stroke_color='black',
                stroke_width=1,
                method='caption',
                size=(video_clip.w * 0.8 * scale_factor, None)  # 80% of video width
            )
            txt = txt.resize(1.0 / scale_factor)
            txt = txt.margin(bottom=155, opacity=0)
            txt = txt.set_position(('center', 'bottom')).set_duration(duration_sec).set_start(start_sec)
            clips.append(txt)
            print(f"  Caption added successfully")
        except Exception as e:
            print(f"  Error creating caption: {str(e)}")
            continue
    
    # Composite all clips
    print("Compositing video with captions...")
    final_video = CompositeVideoClip(clips)
    
    # Write output video
    print(f"Writing output video to: {output_path}")
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
    
    # Clean up
    video_clip.close()
    final_video.close()
    
    print(f"Video with captions saved to: {output_path}")
    return output_path

def attach_captions():
    video_path = "C:\\hook-extraction-service\\src\\sam2\\test_result\\faca8eab-735f-4bce-b3f6-a2416f991efa\\final_output_with_audio.mp4"
    clauses_json_path = "C:\\hook-extraction-service\\src\\caption\\resources\\parsed.json"
    output_path = "C:\\hook-extraction-service\\src\\sam2\\test_result\\faca8eab-735f-4bce-b3f6-a2416f991efa\\final_output_with_audio_and_captions.mp4"
    add_captions_to_video(video_path, clauses_json_path, output_path)

attach_captions()