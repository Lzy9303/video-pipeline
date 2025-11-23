from google import genai
from google.genai import client, types
from google.genai.types import Part, Blob, CreateCachedContentConfig
from dataclasses import dataclass, asdict
import json
import os
import sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(root, 'src'))
from models import Sentence, Hook


client = genai.Client(
    api_key="AIzaSyAwq5CWJcX78-mcdhVCyl7I8LqxxJxxsbU",
    http_options=types.HttpOptions(api_version='v1alpha')
)


def partition_into_parts(transcript) -> list[Sentence]:
    try:
        prompt = open(os.path.join(root, 'llm/prompts/gemini/partition.md'), encoding='utf-8').read()
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                types.Part.from_text(
                    text=prompt
                ),
                types.Part.from_text(text=transcript.model_dump_json())
            ],
            config=types.GenerateContentConfig(
                temperature=1.0,
                response_mime_type='application/json',
                response_schema=list[Sentence],
            ),
        )
        sentences = response.parsed
        result = Hook(sentences=sentences)
        print(result)
        return result

    except Exception as e:
        print(e)
        return []
