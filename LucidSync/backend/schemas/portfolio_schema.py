from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class PortfolioName(BaseModel):
    prompt: str = Field(min_length=1)

class PortfolioOut(BaseModel):
    id: int
    user_id: Optional[int] = None
    name: str
    created_at: Optional[datetime] = None

class PortfolioListOut(BaseModel):
    items: List[PortfolioOut]
    count: int

