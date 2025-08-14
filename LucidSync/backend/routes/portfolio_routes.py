from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional

from dependies.db_mysql import get_db
from models.portfolio import Portfolio
from pydantic import BaseModel, Field
from schemas.portfolio_schema import PortfolioName, PortfolioOut, PortfolioListOut
from services.portfolio_service import list_portfolios, delete_portfolio

router = APIRouter(prefix="/portfolios", tags=["Portfolios"])

@router.post("/create/{user_id}")
def create_portfolio(user_id: int, portfolio: PortfolioName, db: Session = Depends(get_db)):
    new_portfolio = Portfolio(
        user_id=user_id,
        name=portfolio.prompt
    )
    db.add(new_portfolio)
    db.commit()
    db.refresh(new_portfolio)
    return {"message": "Portfolio created successfully", "portfolio": new_portfolio}


@router.get("", response_model=PortfolioListOut)
def get_portfolios(
    user_id: Optional[int] = Query(default=None, description="필요시 특정 유저의 포트폴리오만 조회"),
    db: Session = Depends(get_db),
):
    rows = list_portfolios(db, user_id=user_id)
    items = [PortfolioOut(id=r.id, user_id=r.user_id, name=r.name, created_at=r.created_at) for r in rows]
    return {"items": items, "count": len(items)}

@router.delete("/{portfolio_id}")
def remove_portfolio(portfolio_id: int, db: Session = Depends(get_db)):
    deleted = delete_portfolio(db, portfolio_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return {"deleted": deleted, "portfolio_id": portfolio_id}