import mimetypes
import os
from google import genai
from google.genai import types
from PIL import Image


def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()
    print(f"File saved to to: {file_name}")


def generate_cover(image_path: str, output_path: str = None):
    image = Image.open(image_path)
    client = genai.Client(
        api_key="AIzaSyAwq5CWJcX78-mcdhVCyl7I8LqxxJxxsbU"
    )

    model = "gemini-3-pro-image-preview"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""remove the persons and things they hold and fill it with proper content. keep everything else untouched."""),
            ],
        ),
        image
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        response_modalities=[
            "IMAGE"
        ],
        image_config=types.ImageConfig(
            image_size="1K",
        ),
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    # Generate output path if not provided
    if output_path is None:
        base_dir = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(base_dir, f"{base_name}_generated_cover.png")

    for part in response.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = part.as_image()
            image.save(output_path)
            return output_path
    
    return None
            
def test():
    output_path = generate_cover("C:\\hook-extraction-service\\src\\sam2\\test_result\\23935d1e-99b5-4dd6-956b-e4fb197f6633\\clip_004\\clip_004_first_frame.jpg")
    print(f"Generated image saved to: {output_path}")

# test()