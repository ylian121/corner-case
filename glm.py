from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import os

def extract_with_glm_ocr(image_path):
    
    print("Loading GLM-OCR model...")
    model_id = "zai-org/GLM-OCR"
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Model loaded on: {model.device}")
    
    # Load image
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found!")
        return
    
    image = Image.open(image_path).convert("RGB")
    print(f"Image loaded: {image.size}")
    
    # Prompt for RDF extraction (using their supported formats)
    prompt = """Extract all information from this image as RDF triples.
Format each triple as: <subject> <predicate> <object> .

Include:
- All objects visible (cars, buildings, people, trees, signs)
- Their attributes (colors, sizes, materials)
- Spatial relationships (on, next to, behind, above)
- Any text visible on signs

Example:
<car> <color> "red" .
<car> <located_on> <street> .
<building> <material> "brick" .
<person> <wearing> "hat" ."""

    # Prepare messages in GLM-OCR format
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": image_path  # GLM-OCR expects URL or path
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
    
    # Process inputs 
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    # Move inputs to the same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate with anti-repetition settings
    print(" Generating RDF triples...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,              # Higher for variety
            repetition_penalty=1.2,        # Prevents loops
            do_sample=True,
            top_p=0.9,
        )
    
    # Decode 
    output_text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("GENERATED RDF TRIPLES")
    print("="*60)
    print(output_text)
    print("="*60)
    
    # Save to file
    with open("glm_ocr_rdf_output.txt", "w") as f:
        f.write(output_text)
    print(f"\n Results saved to glm_ocr_rdf_output.txt")
    
    return output_text

if __name__ == "__main__":
    # Use image
    image_path = "pic.png"
    extract_with_glm_ocr(image_path)