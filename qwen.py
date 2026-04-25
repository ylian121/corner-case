import os
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
from qwen_vl_utils import process_vision_info

# Configuration
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct" 
MAX_BATCH_SIZE = 5 

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
# Load once when imported
print(f"Loading {MODEL_ID}...")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

def run_inference(image_paths, prompt):
    """
    Standardized function to handle multiple images (Batch Size 1, 3, or 5).
    """

    content = []
    for path in image_paths:
        content.append({"type": "image", "image": path})
    
    content.append({"type": "text", "text": prompt})

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    # Qwen-specific template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=1024,
            repetition_penalty=1.2
        )
    
    # Output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


'''
import os
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

# HF token for faster downloads
os.environ["HF_TOKEN"] = "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"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def extract_rdf_from_image(image_path="pic.png"):
    """
    Extract RDF triples from an image using Qwen3-VL-2B-Instruct
    """
    
    print("Loading Qwen3-VL-2B-Instruct model...")
    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    
    # load processor with correct image token handling
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        image_processor_type="Qwen2VLImageProcessor",
        vision_config={"image_size": 448}  # Match model's expected input
    )
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    print(f"Model loaded on: {model.device}")
    
    # Load image
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found!")
        return
    
    image = Image.open(image_path).convert("RGB")
    print(f"Image loaded: {image.size}")
    
    # telling the system how to output
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that extracts information from images as RDF triples. Use proper RDF format with prefixes."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image  # Pass image directly in content
                },
                {
                    "type": "text",
                    "text": """Extract all information from this street scene as RDF triples.

Format:
@prefix ex: <http://example.org/scene#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

List all objects with unique IDs (ex:Car1, ex:Car2, ex:Building1, etc.) and their relationships.
Include:
- Cars with colors, positions, directions
- Buildings with types, materials
- Streets, intersections, traffic devices
- Spatial relationships (on, near, behind, next to, over)

Each triple should end with a period."""
                }
            ]
        }
    ]
    
    # handles the image token correctly
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # checking + debugging
    print(f"Processed prompt: {text[:200]}...") 
    
    # Process with images
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Check image token detection
    print(f"Input shape: {inputs['input_ids'].shape}")
    if hasattr(processor, 'image_token_id'):
        img_token_count = (inputs['input_ids'] == processor.image_token_id).sum().item()
        print(f"Image tokens detected: {img_token_count}")
    
    # Generate with anti-repetition settings
    print("Generating RDF triples...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            repetition_penalty=1.2,
            do_sample=True,
            top_p=0.9,
        )
    
    # Decode output
    output = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("GENERATED RDF TRIPLES")
    print("="*60)
    print(output)
    print("="*60)
    
    # Save to file
    with open("qwen_rdf_output1.txt", "w") as f:
        f.write(output)
    print(f"\n Results saved to qwen_rdf_output1.txt")
    
    return output

if __name__ == "__main__":
    extract_rdf_from_image("pic.png")
'''