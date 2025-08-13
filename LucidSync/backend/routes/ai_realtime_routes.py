"""
AI Realtime Trading Routes - 실시간 데이터를 사용한 AI 트레이딩 API
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import sys
import os

# 서비스 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.market_data_service import market_data_service

# AI 모델 경로 추가
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../generate_data/models'))
sys.path.append(model_path)
from trading_ai_service import TradingAIService

router = APIRouter(prefix="/api/ai/realtime", tags=["AI Realtime Trading"])

# 전역 서비스 인스턴스
trading_service = TradingAIService()
models_initialized = False

# ===== Request 모델 정의 =====

class RealtimeBuyRequest(BaseModel):
    """실시간 매수 신호 요청"""
    ticker: str
    position_size_pct: Optional[float] = 5.0  # 포트폴리오 대비 비중 (%)

class RealtimeSellRequest(BaseModel):
    """실시간 매도 신호 요청"""
    ticker: str
    entry_price: float
    entry_date: str  # YYYY-MM-DD format
    position_size: Optional[float] = 100  # 보유 주식 수

# ===== API 엔드포인트 =====

@router.post("/initialize")
async def initialize_ai_models():
    """AI 모델 초기화"""
    global models_initialized
    
    try:
        # 모델 파일 경로
        model_files = {
            'A': os.path.join(model_path, 'trade_quality_evaluator_20250814_013555.pkl'),
            'B': os.path.join(model_path, 'buy_signal_predictor_20250813_015308.pkl'),
            'C': os.path.join(model_path, 'sell_signal_predictor_20250813_071656.pkl')
        }
        
        # 파일 존재 확인
        available_models = {}
        for model_type, filepath in model_files.items():
            if os.path.exists(filepath):
                available_models[model_type] = filepath
            else:
                print(f"Warning: Model file not found - {filepath}")
        
        # 모델 로드
        if available_models:
            results = trading_service.load_models(available_models, verbose=True)
            models_initialized = True
            
            return {
                "status": "success",
                "models_loaded": results,
                "message": "AI models initialized successfully"
            }
        else:
            return {
                "status": "error",
                "message": "No model files found"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize models: {str(e)}")

@router.post("/buy-analysis")
async def analyze_buy_signal(request: RealtimeBuyRequest):
    """
    실시간 매수 신호 분석
    Yahoo Finance에서 실시간 데이터를 가져와 AI 모델로 분석
    """
    global models_initialized
    
    # 모델 초기화 확인
    if not models_initialized:
        # 자동 초기화 시도
        init_result = await initialize_ai_models()
        if init_result["status"] != "success":
            raise HTTPException(status_code=503, detail="AI models not initialized")
    
    try:
        # 1. Yahoo Finance에서 실시간 데이터 수집
        print(f"Fetching real-time data for {request.ticker}...")
        market_data = market_data_service.get_buy_signal_data(
            ticker=request.ticker,
            position_size_pct=request.position_size_pct
        )
        
        # 2. DataFrame으로 변환
        df = pd.DataFrame([market_data])
        
        # 3. AI 모델로 매수 신호 예측
        print(f"Analyzing buy signal for {request.ticker}...")
        result = trading_service.get_buy_signals(
            candidate_data=df,
            threshold=60.0,  # 60점 이상이면 매수 추천
            verbose=True
        )
        
        # 4. 응답 데이터 구성
        response = {
            "ticker": request.ticker,
            "current_price": market_data['current_price'],
            "analysis": {
                "signal_score": result['summary']['avg_signal'],
                "recommendation": "BUY" if result['summary']['avg_signal'] >= 60 else "HOLD",
                "confidence": result['summary']['avg_signal'] / 100,
                "threshold": 60.0
            },
            "market_data": {
                "pe_ratio": market_data['entry_pe_ratio'],
                "pb_ratio": market_data['entry_pb_ratio'],
                "roe": market_data['entry_roe'],
                "52w_high_ratio": market_data['entry_ratio_52w_high'],
                "vix": market_data['entry_vix'],
                "momentum_20d": market_data['entry_momentum_20d'],
                "volatility_20d": market_data['entry_volatility_20d']
            },
            "technical_indicators": {
                "momentum_5d": f"{market_data['entry_momentum_5d']*100:.2f}%",
                "momentum_20d": f"{market_data['entry_momentum_20d']*100:.2f}%",
                "momentum_60d": f"{market_data['entry_momentum_60d']*100:.2f}%",
                "volatility": f"{market_data['entry_volatility_20d']*100:.2f}%"
            },
            "timestamp": market_data['timestamp']
        }
        
        # 매수 추천인 경우 추가 정보
        if result['summary']['avg_signal'] >= 60:
            response["buy_recommendation"] = {
                "suggested_position_size": f"{request.position_size_pct}%",
                "signal_strength": "Strong" if result['summary']['avg_signal'] >= 80 else "Moderate",
                "risk_level": "High" if market_data['entry_volatility_20d'] > 0.3 else "Medium" if market_data['entry_volatility_20d'] > 0.2 else "Low"
            }
        
        return response
        
    except Exception as e:
        print(f"Error in buy analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/sell-analysis")
async def analyze_sell_signal(request: RealtimeSellRequest):
    """
    실시간 매도 신호 분석
    현재 보유 중인 포지션에 대한 매도 시점 분석
    """
    global models_initialized
    
    # 모델 초기화 확인
    if not models_initialized:
        init_result = await initialize_ai_models()
        if init_result["status"] != "success":
            raise HTTPException(status_code=503, detail="AI models not initialized")
    
    try:
        # 1. Yahoo Finance에서 실시간 데이터 수집
        print(f"Fetching sell signal data for {request.ticker}...")
        market_data = market_data_service.get_sell_signal_data(
            ticker=request.ticker,
            entry_price=request.entry_price,
            entry_date=request.entry_date,
            position_size=request.position_size
        )
        
        # 2. DataFrame으로 변환
        df = pd.DataFrame([market_data])
        
        # 3. AI 모델로 매도 신호 예측
        print(f"Analyzing sell signal for {request.ticker}...")
        result = trading_service.get_sell_signals(
            portfolio_data=df,
            threshold=50.0,  # 50점 이상이면 매도 고려
            verbose=True
        )
        
        # 4. 응답 데이터 구성
        current_return = market_data['current_return']
        
        # Check if result contains error
        if 'error' in result:
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {result['error']}")
        
        # Check if summary exists
        if 'summary' not in result:
            raise HTTPException(status_code=500, detail="Model did not return expected summary")
        
        response = {
            "ticker": request.ticker,
            "entry_price": request.entry_price,
            "current_price": market_data['current_price'],
            "holding_days": market_data['holding_days'],
            "current_return": f"{current_return:.2f}%",
            "analysis": {
                "signal_score": result['summary']['avg_signal'],
                "recommendation": "SELL" if result['summary']['avg_signal'] >= 70 else "HOLD" if result['summary']['avg_signal'] >= 50 else "STRONG_HOLD",
                "confidence": result['summary']['avg_signal'] / 100,
                "threshold": 50.0
            },
            "performance": {
                "total_return": f"{current_return:.2f}%",
                "market_return": f"{market_data['market_return_during_holding']:.2f}%",
                "excess_return": f"{market_data['excess_return']:.2f}%",
                "profit_status": "Profit" if current_return > 0 else "Loss"
            },
            "risk_indicators": {
                "current_volatility": f"{market_data['exit_volatility_20d']*100:.2f}%",
                "vix_level": market_data['exit_vix'],
                "momentum_change": f"{(market_data['exit_momentum_20d'] - market_data['entry_momentum_20d'])*100:.2f}%"
            },
            "timestamp": market_data['timestamp']
        }
        
        # 매도 추천 이유 추가
        if result['summary']['avg_signal'] >= 70:
            reasons = []
            if current_return > 20:
                reasons.append("Significant profit achieved")
            if market_data['exit_volatility_20d'] > market_data['entry_volatility_20d'] * 1.5:
                reasons.append("Increased volatility")
            if market_data['exit_vix'] > 25:
                reasons.append("High market fear (VIX > 25)")
            if market_data['exit_momentum_20d'] < 0:
                reasons.append("Negative momentum")
                
            response["sell_recommendation"] = {
                "action": "SELL",
                "urgency": "High" if result['summary']['avg_signal'] >= 80 else "Medium",
                "reasons": reasons if reasons else ["AI model suggests selling based on overall market conditions"]
            }
        
        return response
        
    except Exception as e:
        print(f"Error in sell analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/quick-check/{ticker}")
async def quick_price_check(ticker: str):
    """
    빠른 가격 체크 (AI 분석 없이)
    """
    try:
        current_price = market_data_service.get_current_price(ticker)
        
        if current_price > 0:
            return {
                "ticker": ticker,
                "current_price": current_price,
                "status": "success"
            }
        else:
            return {
                "ticker": ticker,
                "status": "error",
                "message": "Could not fetch price"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-status")
async def check_model_status():
    """AI 모델 상태 확인"""
    try:
        status = trading_service.get_model_status()
        return {
            "initialized": models_initialized,
            "models": status['models_loaded'],
            "ready": status['ready_for_service'],
            "statistics": status['service_stats']
        }
    except Exception as e:
        return {
            "initialized": False,
            "error": str(e)
        }