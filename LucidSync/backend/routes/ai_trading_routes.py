"""
AI Trading Routes - AI 모델 API 엔드포인트
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
import sys
import os

# AI 모델 경로 추가
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../generate_data/models'))
sys.path.append(model_path)

# Trading AI Service import
from trading_ai_service import TradingAIService

router = APIRouter(prefix="/api/ai-trading", tags=["AI Trading"])

# 전역 서비스 인스턴스
trading_service = TradingAIService()

# ===== Request/Response 모델 정의 =====

class BuySignalRequest(BaseModel):
    """매수 신호 요청"""
    ticker: str
    current_price: float
    volume: float
    ma_20: Optional[float] = None
    ma_50: Optional[float] = None
    rsi: Optional[float] = None
    market_data: Optional[Dict[str, Any]] = None

class SellSignalRequest(BaseModel):
    """매도 신호 요청"""
    ticker: str
    entry_price: float
    current_price: float
    holding_days: int
    position_size: float
    market_data: Optional[Dict[str, Any]] = None

class TradeQualityRequest(BaseModel):
    """거래 품질 평가 요청"""
    ticker: str
    entry_price: float
    exit_price: float
    entry_date: str
    exit_date: str
    position_size: float
    trade_type: str = "long"  # long or short

class MarketAnalysisRequest(BaseModel):
    """시장 분석 요청"""
    tickers: List[str]
    analysis_type: str = "comprehensive"  # comprehensive, technical, fundamental

# ===== API 엔드포인트 =====

@router.post("/initialize")
async def initialize_models():
    """AI 모델 초기화 및 로드"""
    try:
        # 모델 파일 정의
        model_files = {
            'A': 'trade_quality_evaluator_20250814_013555.pkl',
            'B': 'buy_signal_predictor_20250813_015308.pkl', 
            'C': 'sell_signal_predictor_20250813_071656.pkl'
        }
        
        # 전체 경로로 변환
        full_paths = {}
        missing_files = []
        
        for model_type, filename in model_files.items():
            model_path_full = os.path.join(model_path, filename)
            if os.path.exists(model_path_full):
                full_paths[model_type] = model_path_full
            else:
                missing_files.append(f"{model_type}: {filename}")
        
        # load_models 메소드 호출 (복수형!)
        results = trading_service.load_models(full_paths, verbose=True)
        
        return {
            "status": "initialized",
            "models_loaded": results,
            "missing_files": missing_files,
            "model_status": trading_service.get_model_status(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/buy-signal")
async def predict_buy_signal(request: BuySignalRequest):
    """매수 신호 예측"""
    try:
        # 데이터 준비
        market_data = pd.DataFrame([{
            'symbol': request.ticker,  # symbol로 변경
            'current_price': request.current_price,
            'volume': request.volume,
            'ma_20': request.ma_20,
            'ma_50': request.ma_50,
            'rsi': request.rsi,
            **(request.market_data or {})
        }])
        
        # 매수 신호 예측 (get_buy_signals 사용)
        result = trading_service.get_buy_signals(
            candidate_data=market_data,
            threshold=60.0,  # 60점 이상이면 매수 신호
            verbose=True
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sell-signal")
async def predict_sell_signal(request: SellSignalRequest):
    """매도 신호 예측"""
    try:
        # 포지션 데이터 준비
        position_data = pd.DataFrame([{
            'symbol': request.ticker,  # symbol로 변경
            'entry_price': request.entry_price,
            'current_price': request.current_price,
            'holding_days': request.holding_days,
            'position_size': request.position_size,
            'current_return': (request.current_price - request.entry_price) / request.entry_price * 100,  # 퍼센트로 변환
            **(request.market_data or {})
        }])
        
        # 매도 신호 예측 (get_sell_signals 사용)
        result = trading_service.get_sell_signals(
            portfolio_data=position_data,
            threshold=0.0,  # 0점 이상이면 매도 고려
            verbose=True
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate-trade")
async def evaluate_trade_quality(request: TradeQualityRequest):
    """완료된 거래 품질 평가"""
    try:
        # 거래 데이터 준비
        trade_data = pd.DataFrame([{
            'symbol': request.ticker,  # symbol로 변경
            'entry_price': request.entry_price,
            'exit_price': request.exit_price,
            'entry_datetime': request.entry_date,
            'exit_datetime': request.exit_date,
            'position_size_pct': request.position_size,
            'return_pct': ((request.exit_price - request.entry_price) / request.entry_price) * 100,
            'trade_type': request.trade_type
        }])
        
        # 거래 품질 평가 (evaluate_trade_quality 사용)
        result = trading_service.evaluate_trade_quality(
            completed_trades=trade_data,
            verbose=True
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/market-analysis")
async def analyze_market(request: MarketAnalysisRequest):
    """시장 종합 분석"""
    try:
        # 여러 종목 동시 분석
        analysis_results = {}
        
        for ticker in request.tickers:
            # 각 종목별 분석 수행
            result = trading_service.get_comprehensive_analysis(
                ticker=ticker,
                analysis_type=request.analysis_type
            )
            analysis_results[ticker] = result
        
        return {
            "analysis_type": request.analysis_type,
            "results": analysis_results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/service-stats")
async def get_service_stats():
    """서비스 통계 조회"""
    try:
        return trading_service.get_service_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-status")
async def get_model_status():
    """모델 상태 확인"""
    try:
        return {
            "models_loaded": trading_service.models_loaded,
            "service_uptime": str(datetime.now() - trading_service.service_stats['service_start_time']),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))