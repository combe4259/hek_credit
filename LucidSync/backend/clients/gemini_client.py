import google.generativeai as genai
from backend.core.settings import settings

# 초기화 1회
genai.configure(api_key=settings.google_api_key)

def generate_text(prompt: str, model_name: str | None = None, temperature: float | None = None) -> str:
    model = genai.GenerativeModel(
        model_name or settings.gemini_model,
        generation_config={"temperature": temperature if temperature is not None else settings.gemini_temp}
    )
    resp = model.generate_content(prompt)
    return (getattr(resp, "text", "") or "").strip()

def generate_json(prompt: str, model_name: str | None = None, temperature: float = 0.2) -> dict:
    import json
    model = genai.GenerativeModel(
        model_name or settings.gemini_model,
        generation_config={"temperature": temperature, "response_mime_type": "application/json"}
    )
    resp = model.generate_content(prompt)
    return json.loads(resp.text.strip())