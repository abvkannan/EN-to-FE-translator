from fastapi import FastAPI
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer

app = FastAPI()

# Load pre-trained MarianMT model and tokenizer from Hugging Face
model_name = "Helsinki-NLP/opus-mt-en-fr"  # Example: English to French
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

class TranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str

@app.post("/translate")
async def translate(request: TranslationRequest):
    # Translation logic
    translated = model.generate(**tokenizer(request.text, return_tensors="pt", padding=True, truncation=True))
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return {"translated_text": translated_text}
