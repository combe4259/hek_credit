from fastapi import APIRouter, HTTPException
from schemas.gemini_schema import PromptIn, TextOut, JSONOut
from services.gemini_service import ask_gemini_service, ask_gemini_json_service

router = APIRouter(prefix="/ai", tags=["AI"])

@router.post("/ask", response_model=TextOut)
def ask(body: PromptIn):
    try:
        return {"output": ask_gemini_service(body.prompt)}
    except Exception as e:
        raise HTTPException(500, f"Gemini error: {e}")

@router.post("/ask_json", response_model=JSONOut)
def ask_json(body: PromptIn):
    try:
        return {"output": ask_gemini_json_service(body.prompt)}
    except Exception as e:
        raise HTTPException(500, f"Gemini JSON error: {e}")
