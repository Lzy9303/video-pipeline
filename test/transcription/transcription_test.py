import os
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(root, 'src'))
from transcription import transcript_audio


def test_transcript_audio():
    base = os.path.join(root, 'test', 'resources', 'transcription')
    os.makedirs(base, exist_ok=True)
    src_audio = os.path.join(base, 'sample.mp3')
    result = transcript_audio(src_audio)
    print(result)


test_transcript_audio()