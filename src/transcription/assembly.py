import assemblyai as aai
import os
import sys
import subprocess

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(root, 'src'))

aai.settings.api_key = "63f7950f907945b5852c4bd30977a0e1"

config = aai.TranscriptionConfig(
    speaker_labels=False,
    format_text=False,
    punctuate=False,
    speech_model=aai.SpeechModel.best,
    language_code="zh"
)


def transcript_audio(file_path):
    # Check if file is MP4 and transcode to MP3
    file_ext = os.path.splitext(file_path)[1].lower()
    audio_file = file_path
    temp_mp3 = None
    
    if file_ext == '.mp4':
        print(f"Transcoding MP4 to MP3: {file_path}")
        temp_mp3 = os.path.splitext(file_path)[0] + '_temp.mp3'
        
        cmd = [
            'ffmpeg', '-y',
            '-i', file_path,
            '-vn',  # No video
            '-acodec', 'libmp3lame',
            '-q:a', '2',  # High quality
            temp_mp3
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            audio_file = temp_mp3
            print(f"Transcoded to: {temp_mp3}")
        except subprocess.CalledProcessError as e:
            print(f"Error transcoding MP4 to MP3: {e.stderr.decode()}")
            # Fall back to original file
            audio_file = file_path
        except Exception as e:
            print(f"Error transcoding: {str(e)}")
            audio_file = file_path
    
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(audio_file)

    if transcript.status == aai.TranscriptStatus.error:
        print(transcript.error)
    else:
        print(transcript.text)
    
    # Clean up temporary MP3 file
    if temp_mp3 and os.path.exists(temp_mp3):
        try:
            os.remove(temp_mp3)
            print(f"Cleaned up temporary file: {temp_mp3}")
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_mp3}: {str(e)}")

    return transcript

transcript_audio("C:\\hook-extraction-service\\src\\sam2\\test_result\\798c7d53-3a2d-489b-baba-6697a683c4bc\\final_output_with_audio.mp4")