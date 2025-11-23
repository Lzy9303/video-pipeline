from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip


def add_background_music(video_path, audio_path, output_path, bgm_volume=0.3):
    """
    Add background music to a video file.
    
    Args:
        video_path: Path to the input video file
        audio_path: Path to the background music audio file
        output_path: Path to save the output video with background music
        bgm_volume: Volume level for background music (0.0 to 1.0, default: 0.3)
        
    Returns:
        str: Path to the output video file
        
    Example:
        add_background_music(
            video_path="input.mp4",
            audio_path="bgm.mp3",
            output_path="output_with_bgm.mp4",
            bgm_volume=0.2  # 20% volume
        )
    """
    print(f"Loading video: {video_path}")
    video = VideoFileClip(video_path)
    video_duration = video.duration
    
    print(f"Video duration: {video_duration:.2f}s")
    print(f"Loading background music: {audio_path}")
    
    # Load background music
    bgm = AudioFileClip(audio_path)
    
    # Trim or loop BGM to match video duration
    if bgm.duration > video_duration:
        # Trim BGM if it's longer than video
        print(f"Trimming BGM from {bgm.duration:.2f}s to {video_duration:.2f}s")
        bgm = bgm.subclip(0, video_duration)
    elif bgm.duration < video_duration:
        # Loop BGM if it's shorter than video
        print(f"Looping BGM to match video duration")
        loops_needed = int(video_duration / bgm.duration) + 1
        bgm = bgm.loop(n=loops_needed).subclip(0, video_duration)
    
    # Adjust BGM volume
    print(f"Setting BGM volume to {bgm_volume * 100:.0f}%")
    bgm = bgm.volumex(bgm_volume)
    
    # Mix audio tracks
    if video.audio is not None:
        print("Mixing original audio with background music")
        # Composite original audio with BGM
        final_audio = CompositeAudioClip([video.audio, bgm])
    else:
        print("Video has no audio, using only background music")
        final_audio = bgm
    
    # Set the final audio to the video
    video = video.set_audio(final_audio)
    
    # Write output video
    print(f"Writing output video to: {output_path}")
    video.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True
    )
    
    # Clean up
    video.close()
    bgm.close()
    
    print(f"Video with background music saved to: {output_path}")
    return output_path


def test():
    """Test function for adding background music"""
    video_path = "C:\\hook-extraction-service\\src\\sam2\\test_result\\faca8eab-735f-4bce-b3f6-a2416f991efa\\final_output_with_audio_and_captions.mp4"
    audio_path = "C:\\hook-extraction-service\\src\\bgm\\resources\\bgm.mp3"
    output_path = "C:\\hook-extraction-service\\src\\sam2\\test_result\\faca8eab-735f-4bce-b3f6-a2416f991efa\\final_output_complete.mp4"
    
    add_background_music(
        video_path=video_path,
        audio_path=audio_path,
        output_path=output_path,
        bgm_volume=0.07  # 20% volume for background music
    )


if __name__ == "__main__":
    test()
