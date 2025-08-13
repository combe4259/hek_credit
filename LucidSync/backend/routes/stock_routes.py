from fastapi import APIRouter, Query
from typing import List
from backend.services.stock_service import get_stocks_list
from backend.schemas.stock_schema import StockItem

router = APIRouter(prefix="/stocks", tags=["Stocks"])

@router.get("/list", response_model=List[StockItem])
def list_stocks(symbols: List[str] = Query(..., description="주식 심볼 리스트")):
    return get_stocks_list(symbols)