from backend.clients.gemini_client import generate_text, generate_json

def ask_gemini_service(prompt: str) -> str:
    return generate_text(prompt)

def ask_gemini_json_service(prompt: str) -> dict:
    return generate_json(prompt)
