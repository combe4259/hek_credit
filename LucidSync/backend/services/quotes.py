# services/quotes.py
import asyncio, time, logging
import yfinance as yf
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from schemas.stock_schema import StockItem

logger = logging.getLogger(__name__)

_CACHE: Dict[str, tuple[float, StockItem]] = {}
_TTL = 5  

def _build_item(sym: str, t) -> StockItem:
    fi = getattr(t, "fast_info", {}) or {}
    try:
        info = t.get_info() or {}
    except Exception:
        info = {}

    last = fi.get("last_price") or info.get("regularMarketPrice")
    prev = fi.get("previous_close") or info.get("regularMarketPreviousClose")
    change_pct = None
    try:
        if last is not None and prev not in (None, 0):
            change_pct = (float(last) - float(prev)) / float(prev) * 100
    except Exception:
        pass

    return StockItem(
        symbol=sym.upper(),
        name=info.get("shortName") or info.get("longName"),
        price=None if last is None else float(last),
        currency=fi.get("currency") or info.get("currency"),
        change=None if change_pct is None else round(change_pct, 2),
        market_state=fi.get("market_state") or info.get("marketState"),
    )

def _fetch_many_sync(symbols: List[str]) -> List[StockItem]:
    s = " ".join(symbols)
    tickers = yf.Tickers(s)
    out: List[StockItem] = []
    for sym in symbols:
        try:
            t = tickers.tickers.get(sym) or yf.Ticker(sym)
            out.append(_build_item(sym, t))
        except Exception as e:
            logger.warning("yfinance error for %s: %s", sym, e)
            out.append(StockItem(symbol=sym.upper()))
    return out

async def get_quotes(symbols: List[str]) -> List[StockItem]:
    now = time.time()
    to_fetch: List[str] = []
    fresh: List[StockItem] = []

    # 캐시 히트
    for s in symbols:
        key = s.strip()
        if not key:
            continue
        cached = _CACHE.get(key)
        if cached and now - cached[0] < _TTL:
            fresh.append(cached[1])
        else:
            to_fetch.append(key)

    # 캐시 미스만 병렬로 (GIL 회피용 to_thread)
    if to_fetch:
        batch = await asyncio.to_thread(_fetch_many_sync, to_fetch)
        now2 = time.time()
        for item in batch:
            _CACHE[item.symbol] = (now2, item)
        fresh.extend(batch)

    # 원본 순서 유지
    order = {s.upper(): i for i, s in enumerate(symbols)}
    fresh.sort(key=lambda x: order.get(x.symbol, 10**9))
    return fresh
