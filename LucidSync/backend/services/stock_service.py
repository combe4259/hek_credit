import yfinance as yf
from schemas.stock_schema import StockItem

def get_stocks_list(symbols: list[str]) -> list[StockItem]:
    result = []
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            fi = getattr(t, "fast_info", {}) or {}
            info = {}
            try:
                info = t.get_info() or {}
            except Exception:
                pass

            change_pct = None
            if fi.get("previous_close") and fi.get("last_price"):
                change_pct = (
                    (fi["last_price"] - fi["previous_close"]) / fi["previous_close"] * 100
                )

            result.append(
                StockItem(
                    symbol=sym.upper(),
                    name=info.get("shortName") or info.get("longName"),
                    price=fi.get("last_price"),
                    currency=fi.get("currency") or info.get("currency"),
                    change=round(change_pct, 2) if change_pct is not None else None,
                    market_state=fi.get("market_state"),
                )
            )
        except Exception as e:
            result.append(StockItem(symbol=sym.upper()))
    return result
