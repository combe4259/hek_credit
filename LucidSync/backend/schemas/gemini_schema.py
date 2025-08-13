from pydantic import BaseModel, Field
from typing import Any, Dict

class PromptIn(BaseModel):
    prompt: str = Field(min_length=1)

class TextOut(BaseModel):
    output: str

class JSONOut(BaseModel):
    output: Dict[str, Any]