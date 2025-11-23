import os
import sys
import json

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(root, 'src'))
from partition import partition_into_parts
from models import Transcript

def test_partition():
    base = os.path.join(root, 'test', 'resources', 'partition')
    os.makedirs(base, exist_ok=True)
    data_dict = json.loads(open(os.path.join(base, 'sample.json'), encoding='utf-8').read())
    transcript = Transcript.model_validate(data_dict)
    result = partition_into_parts(transcript)

    print(result)

test_partition()