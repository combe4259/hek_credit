from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from routes.gemini_routes import router as ai_router
from routes.stock_routes import router as stock_router
from routes.portfolio_routes import router as portfolio_router
from routes.user_routes import router as user_router
from dependies.db_mysql import engine, Base
from fastapi.middleware.cors import CORSMiddleware
from yahoo_ws_hub import YahooHub
import yfinance as yf
import json
from typing import Set

app = FastAPI(title="LucidSync")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            
    allow_credentials=True,           
    allow_methods=["*"],              
    allow_headers=["*"],             
)

hub = YahooHub()

app.include_router(ai_router)
app.include_router(stock_router)
app.include_router(portfolio_router)
app.include_router(user_router)

# AI Trading 라우터 추가 - 점진적 테스트
# from routes.ai_trading_routes import router as ai_trading_router
from routes.ai_realtime_routes import router as ai_realtime_router
# app.include_router(ai_trading_router)
app.include_router(ai_realtime_router)

@app.get("/quotes/prevclose")
async def get_prevclose(symbols: str = Query(..., description="comma-separated Yahoo-format symbols")):
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    out = {}
    for s in syms:
      try:
        t = yf.Ticker(s)
        fi = getattr(t, "fast_info", {}) or {}
        pc = fi.get("previous_close")
        if pc is None:
          info = t.get_info() or {}
          pc = info.get("regularMarketPreviousClose") or info.get("previousClose")
        if pc is not None:
          out[s] = float(pc)
      except Exception:
        pass
    return out

@app.websocket("/ws/quotes")
async def ws_quotes(ws: WebSocket, symbols: str | None = Query(None)):
    syms: Set[str] = set()
    if symbols:
        syms = {s.strip().upper() for s in symbols.split(",") if s.strip()}
    await hub.connect(ws, syms)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
                if msg.get("type") == "subscribe":
                    await hub.update_symbols(ws, set(map(str.upper, msg.get("symbols", []))))
            except Exception:
                pass
    except WebSocketDisconnect:
        await hub.disconnect(ws)

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    
    # 뉴스 감정 분석 모델 미리 로드
    print("🚀 서버 시작 - 뉴스 감정 분석 모델 초기화 중...")
    
    try:
        from services.news_sentiment_service import NewsSentimentService
        
        # 뉴스 감정 분석 서비스 인스턴스 생성 및 모델 로드
        news_service = NewsSentimentService()
        print("📊 뉴스 감정 분석 모델 로드 시작... (547MB, 시간이 걸릴 수 있습니다)")
        
        if news_service.load_model():
            print("✅ 뉴스 감정 분석 모델 로드 완료")
        else:
            print("⚠️ 뉴스 감정 분석 모델 로드 실패 - 기본값으로 동작")
            
        # 전역적으로 사용할 수 있도록 앱에 저장
        app.state.news_service = news_service
        
        print("🎉 뉴스 감정 분석 모델 초기화 완료")
        
    except Exception as e:
        print(f"❌ 뉴스 모델 초기화 중 오류 발생: {e}")
        # 오류가 발생해도 서버는 계속 실행

# 헬스체크
@app.get("/health")
def health():
    return {"ok": True}