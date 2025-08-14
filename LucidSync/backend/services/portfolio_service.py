from typing import List
from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from backend.models.portfolio import Portfolio

# (선택) 포트폴리오 하위 테이블이 있으면 여기에 import 해서 같이 삭제 처리
# from backend.models.portfolio_stock import PortfolioStock

def list_portfolios(db: Session, user_id: int | None = None) -> List[Portfolio]:
    """
    user_id가 주어지면 해당 유저의 포트폴리오만,
    없으면 전체 포트폴리오.
    """
    stmt = select(Portfolio)
    if user_id is not None:
        stmt = stmt.where(Portfolio.user_id == user_id)
    return list(db.execute(stmt).scalars().all())

def delete_portfolio(db: Session, portfolio_id: int) -> int:
    """
    포트폴리오 1개 삭제. 하위 데이터가 있으면 먼저 정리해야 함.
    반환값: 삭제된 포트폴리오 개수(0 또는 1)
    """
    # 하위 테이블 삭제 필요 시 선삭제 (예시)
    # db.execute(delete(PortfolioStock).where(PortfolioStock.portfolio_id == portfolio_id))

    res = db.execute(delete(Portfolio).where(Portfolio.id == portfolio_id))
    db.commit()
    return res.rowcount or 0
