from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch

MODEL_ID = "vikhyatk/moondream2"

PROMPT = """Analyze this image and output RDF triples in Turtle format only.

@prefix ex: <http://example.org/scene#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

Rules:
- List every object you see (cars, people, buildings, signs, etc.)
- Include color, position, and relationships between objects
- Use simple relationships: ex:nextTo, ex:on, ex:behind, ex:inFrontOf, ex:partOf, ex:holds, ex:wears
- Every object gets a type, color, and at least one relationship

Example output style:
ex:Car1 rdf:type ex:Car ;
    ex:color "red" ;
    ex:locatedOn ex:Street1 ;
    ex:nextTo ex:Building1 .

Output only valid Turtle syntax, nothing else."""

print(f"Loading model: {MODEL_ID} (first run will download 2GB)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
print("Model loaded!\n")

def image_to_rdf(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    enc_image = model.encode_image(image)
    result = model.answer_question(enc_image, PROMPT, tokenizer)
    return result

if __name__ == "__main__":
    image_path = "pic.png" 
    print(f"Running Model: {MODEL_ID}\n")
    result = image_to_rdf(image_path)
    print(result)