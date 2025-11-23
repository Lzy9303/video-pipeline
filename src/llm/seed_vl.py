# Calling seed_vl to determine which part need to be masked, mask size, and xy coordinates

import os
from volcenginesdkarkruntime import Ark
import base64
import json

client = Ark(
base_url="https://ark.cn-beijing.volces.com/api/v3",
api_key="de8cf075-14cd-4a0a-b6d5-305e554d7000"
)

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def locate_products(image_path: str):
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')
    prompt = open(os.path.join(root, 'llm/prompts/seed_vl/locate_objects.md'), encoding='utf-8').read()
    response = client.chat.completions.create(
        model="doubao-seed-1-6-vision-250815",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    print(response.choices[0])
    return json.loads(response.choices[0].message.content)


def locate_banner(image_path: str):
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')
    prompt = open(os.path.join(root, 'llm/prompts/seed_vl/locate_banner.md'), encoding='utf-8').read()
    response = client.chat.completions.create(
        model="doubao-seed-1-6-vision-250815",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    print(response.choices[0])
    return json.loads(response.choices[0].message.content)


def locate_caption(image_path: str):
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')
    prompt = open(os.path.join(root, 'llm/prompts/seed_vl/locate_caption.md'), encoding='utf-8').read()
    response = client.chat.completions.create(
        model="doubao-seed-1-6-vision-250815",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    print(response.choices[0])
    return json.loads(response.choices[0].message.content)
