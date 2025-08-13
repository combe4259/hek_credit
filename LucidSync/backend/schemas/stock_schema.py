from pydantic import BaseModel
from typing import Optional

class StockItem(BaseModel):
    symbol: str
    name: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    change: Optional[float] = None  
    market_state: Optional[str] = None