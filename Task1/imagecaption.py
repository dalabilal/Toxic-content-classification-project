"""Image captioning with BLIP-1. Used by app.py."""
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_NAME = "Salesforce/blip-image-captioning-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = None
model = None


def load_model():
    """Load BLIP only once, then reuse it."""
    global processor, model
    if model is None:
        processor = BlipProcessor.from_pretrained(MODEL_NAME)
        model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
        model.to(DEVICE).eval()


def generate_caption(image):
    """Take a PIL image and return a short text caption."""
    load_model()
    inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            num_beams=5,             # beam search tries 5 candidate sentences,
            max_new_tokens=40,       # enough for one full sentence
            min_new_tokens=5,        # avoids one-word captions
            repetition_penalty=1.2,  # stops loops like "a man a man a man"
            early_stopping=True,     # stop when all beams are finished
        )
    return processor.decode(ids[0], skip_special_tokens=True).strip()