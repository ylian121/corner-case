import argparse
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEFAULT_IMAGE = "pic.png"
DEFAULT_PROMPT = "What objects are in this image?"
MAX_NEW_TOKENS = 200

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SmolVLM-256M on {device.upper()}...")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if device == "cuda" else "eager",
    ).to(device)

    print("Model loaded!\n")
    return processor, model, device


def describe_image(image_path, prompt, processor, model, device):
    print(f"Image : {image_path}")
    print(f"Prompt: {prompt}\n")

    # Load image
    image = load_image(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Prepare inputs
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=[image], return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    # Decode — strip the input tokens from the output
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("Description:")
    print("-" * 50)
    print(result.strip())
    print("-" * 50)
    return result


def main():
    parser = argparse.ArgumentParser(description="Describe an image using SmolVLM-256M")
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE,
                        help="Path or URL to the image (default: Statue of Liberty)")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                        help="Question or instruction about the image")
    args = parser.parse_args()

    processor, model, device = load_model()
    describe_image(args.image, args.prompt, processor, model, device)


if __name__ == "__main__":
    main()