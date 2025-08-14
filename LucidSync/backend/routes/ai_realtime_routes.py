"""
AI Realtime Trading Routes - 실시간 데이터를 사용한 AI 트레이딩 API
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import sys
import os
import numpy as np
from datetime import datetime

# 서비스 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.market_data_service import market_data_service

# AI 모델 경로 추가
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../generate_data/models'))
sys.path.append(model_path)
from trading_ai_service import TradingAIService
from trade_feedback_service import DataDrivenFeedbackService

router = APIRouter(prefix="/api/ai/realtime", tags=["AI Realtime Trading"])

# 전역 서비스 인스턴스
trading_service = TradingAIService()
feedback_service = DataDrivenFeedbackService()
models_initialized = False
feedback_service_initialized = False

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

class TradeQualityRequest(BaseModel):
    """거래 품질 평가 요청"""
    symbol: str
    entryPrice: float
    exitPrice: float
    entryDate: str  # YYYY-MM-DD format
    exitDate: str   # YYYY-MM-DD format
    quantity: float
    returnPct: float
    holdingDays: int

class ComprehensiveTradeQualityRequest(BaseModel):
    """포괄적 거래 품질 평가 요청 (모든 피처 포함)"""
    # 기본 거래 정보
    symbol: str
    return_pct: float
    holding_period_days: int
    position_size_pct: float
    
    # 진입시 피처들
    entry_momentum_5d: float
    entry_momentum_20d: float
    entry_momentum_60d: float
    entry_ma_dev_5d: float
    entry_ma_dev_20d: float
    entry_ma_dev_60d: float
    entry_volatility_5d: float
    entry_volatility_20d: float
    entry_volatility_60d: float
    entry_vol_change_5d: float
    entry_vol_change_20d: float
    entry_vol_change_60d: float
    entry_vix: float
    entry_tnx_yield: float
    
    # 청산시 피처들
    exit_momentum_5d: float
    exit_momentum_20d: float
    exit_momentum_60d: float
    exit_ma_dev_5d: float
    exit_ma_dev_20d: float
    exit_ma_dev_60d: float
    exit_volatility_5d: float
    exit_volatility_20d: float
    exit_volatility_60d: float
    exit_vix: float
    exit_tnx_yield: float
    exit_ratio_52w_high: float
    
    # 변화량 피처들
    change_momentum_5d: float
    change_momentum_20d: float
    change_momentum_60d: float
    change_ma_dev_5d: float
    change_ma_dev_20d: float
    change_ma_dev_60d: float
    change_volatility_5d: float
    change_volatility_20d: float
    change_volatility_60d: float
    change_vix: float
    change_tnx_yield: float
    change_ratio_52w_high: float
    
    # 시장 대비 성과
    market_return_during_holding: float
    excess_return: float

class TradeFeedbackRequest(BaseModel):
    """거래 피드백 요청 (DataDrivenFeedbackService용)"""
    symbol: str
    entry_date: str
    exit_date: str
    return_pct: float
    holding_period_days: int
    entry_price: float
    exit_price: float
    quantity: float
    position_size_pct: Optional[float] = 5.0
    
    # 진입시 기술적 지표
    entry_momentum_5d: Optional[float] = 0
    entry_momentum_20d: Optional[float] = 0
    entry_momentum_60d: Optional[float] = 0
    entry_ma_dev_5d: Optional[float] = 0
    entry_ma_dev_20d: Optional[float] = 0
    entry_ma_dev_60d: Optional[float] = 0
    entry_volatility_5d: Optional[float] = 0.2
    entry_volatility_20d: Optional[float] = 0.2
    entry_volatility_60d: Optional[float] = 0.2
    entry_vix: Optional[float] = 20.0
    entry_tnx_yield: Optional[float] = 4.0
    entry_ratio_52w_high: Optional[float] = 0.8  # 52주 최고가 대비 비율
    
    # 펀더멘털 지표
    entry_pe_ratio: Optional[float] = 15.0
    entry_pb_ratio: Optional[float] = 2.0
    entry_roe: Optional[float] = 0.15
    entry_earnings_growth: Optional[float] = 0.05
    entry_operating_margin: Optional[float] = 0.1
    entry_debt_equity_ratio: Optional[float] = 0.5
    
    # 변동성 변화
    entry_vol_change_5d: Optional[float] = 0
    entry_vol_change_20d: Optional[float] = 0
    entry_vol_change_60d: Optional[float] = 0
    
    # 시장 관련 피처들
    market_entry_ma_return_5d: Optional[float] = 0
    market_entry_ma_return_20d: Optional[float] = 0
    market_entry_cum_return_5d: Optional[float] = 0
    market_entry_volatility_20d: Optional[float] = 0.15
    
    # 청산시 기술적 지표
    exit_momentum_5d: Optional[float] = 0
    exit_momentum_20d: Optional[float] = 0
    exit_momentum_60d: Optional[float] = 0
    exit_ma_dev_5d: Optional[float] = 0
    exit_ma_dev_20d: Optional[float] = 0
    exit_ma_dev_60d: Optional[float] = 0
    exit_volatility_5d: Optional[float] = 0.2
    exit_volatility_20d: Optional[float] = 0.2
    exit_volatility_60d: Optional[float] = 0.2
    exit_vix: Optional[float] = 20.0
    exit_tnx_yield: Optional[float] = 4.0
    exit_ratio_52w_high: Optional[float] = 0.8  # 52주 최고가 대비 비율
    
    # 변화량 피처들 (exit - entry)
    change_momentum_5d: Optional[float] = 0
    change_momentum_20d: Optional[float] = 0
    change_momentum_60d: Optional[float] = 0
    change_ma_dev_5d: Optional[float] = 0
    change_ma_dev_20d: Optional[float] = 0
    change_ma_dev_60d: Optional[float] = 0
    change_volatility_5d: Optional[float] = 0
    change_volatility_20d: Optional[float] = 0
    change_volatility_60d: Optional[float] = 0
    change_vix: Optional[float] = 0
    change_tnx_yield: Optional[float] = 0
    change_ratio_52w_high: Optional[float] = 0
    
    # 시장 성과
    market_return_during_holding: Optional[float] = 0
    excess_return: Optional[float] = 0

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
            
            # 피드백 서비스 초기화 (같은 모델 사용)
            global feedback_service_initialized
            try:
                feedback_model_paths = {
                    'buy': available_models.get('B'),
                    'sell': available_models.get('C'), 
                    'quality': available_models.get('A')
                }
                # 모델 로드 및 보정 (과거 데이터는 없지만 모델은 로드)
                feedback_success = feedback_service.load_models_and_calibrate(feedback_model_paths, verbose=True)
                feedback_service_initialized = feedback_success
                print(f"✅ 거래 피드백 서비스 초기화: {'성공' if feedback_success else '실패'}")
            except Exception as e:
                print(f"⚠️ 거래 피드백 서비스 초기화 실패: {str(e)}")
                feedback_service_initialized = False
            
            return {
                "status": "success",
                "models_loaded": results,
                "feedback_service_initialized": feedback_service_initialized,
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

@router.post("/evaluate-trade-quality-comprehensive")
async def evaluate_trade_quality_comprehensive(request: ComprehensiveTradeQualityRequest):
    """포괄적 피처 데이터를 사용한 거래 품질 AI 평가"""
    try:
        if not models_initialized:
            raise HTTPException(status_code=503, detail="AI 모델이 초기화되지 않았습니다.")
        
        # 포괄적 피처 데이터를 DataFrame으로 변환
        feature_data = request.dict()
        
        # 기본 거래 정보 추가 (AI 모델에서 요구하지는 않지만 참조용)
        trade_df = pd.DataFrame([feature_data])
        
        # AI 거래 품질 평가 실행
        result = trading_service.evaluate_trade_quality(trade_df, verbose=True)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=f"거래 품질 평가 실패: {result['error']}")
        
        # 결과 파싱
        if 'evaluations' in result and len(result['evaluations']) > 0:
            evaluation = result['evaluations'][0]
            return {
                "symbol": request.symbol,
                "qualityScore": evaluation.get('quality_score', 0),
                "entryQuality": evaluation.get('quality_score', 0) * 0.9,  # 추정값
                "exitTiming": evaluation.get('quality_score', 0) * 1.1,    # 추정값
                "resultQuality": evaluation.get('quality_score', 0) * 0.95, # 추정값
                "feedback": f"거래 품질 등급: {evaluation.get('grade', 'N/A')}",
                "grade": evaluation.get('grade', 'N/A'),
                "details": {
                    "return_pct": evaluation.get('return_pct', 0),
                    "holding_days": evaluation.get('holding_days', 0),
                    "raw_quality_score": evaluation.get('quality_score', 0)
                },
                "status": "success"
            }
        else:
            raise HTTPException(status_code=500, detail="AI 모델이 평가 결과를 반환하지 않았습니다.")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"포괄적 거래 품질 평가 중 오류 발생: {str(e)}")

@router.post("/evaluate-trade-quality")
async def evaluate_trade_quality(request: TradeQualityRequest):
    """완료된 거래의 품질을 AI로 평가"""
    try:
        if not models_initialized:
            raise HTTPException(status_code=503, detail="AI 모델이 초기화되지 않았습니다.")
        
        # 거래 데이터 준비
        trade_data = {
            'symbol': request.symbol,
            'entry_price': request.entryPrice,
            'exit_price': request.exitPrice,
            'entry_date': request.entryDate,
            'exit_date': request.exitDate,
            'quantity': request.quantity,
            'return_pct': request.returnPct,
            'holding_period_days': request.holdingDays
        }
        
        # AI 거래 품질 평가 실행 (단일 거래)
        result = trading_service.evaluate_single_trade_quality(trade_data)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=f"거래 품질 평가 실패: {result['error']}")
        
        return {
            "symbol": request.symbol,
            "qualityScore": result.get('quality_score', 0),
            "entryQuality": result.get('entry_quality', 0),
            "exitTiming": result.get('exit_timing', 0), 
            "resultQuality": result.get('result_quality', 0),
            "feedback": result.get('feedback', '거래 품질 평가가 완료되었습니다.'),
            "grade": result.get('grade', 'N/A'),
            "details": result.get('details', {}),
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"거래 품질 평가 중 오류 발생: {str(e)}")

@router.post("/trade-feedback")
async def generate_trade_feedback(request: TradeFeedbackRequest):
    """완료된 거래에 대한 AI 기반 데이터 분석 피드백 생성"""
    try:
        # 피드백 서비스 초기화 확인
        if not feedback_service_initialized:
            # 자동 초기화 시도
            if not models_initialized:
                init_result = await initialize_ai_models()
                if init_result["status"] != "success":
                    raise HTTPException(status_code=503, detail="AI 모델이 초기화되지 않았습니다.")
        
        # 거래 데이터 준비
        trade_data = request.dict()
        
        print(f"🔍 거래 피드백 분석 시작: {trade_data['symbol']} ({trade_data['entry_date']} ~ {trade_data['exit_date']})")
        
        # AI 데이터 기반 분석 실행
        try:
            print(f"📊 AI 피드백 분석 시작 - 모델 로드 상태: {feedback_service.models_loaded}")
            
            if feedback_service.models_loaded:
                # 실제 AI 모델을 사용한 분석
                feedback_result = feedback_service.analyze_trade(trade_data, verbose=True)
                print(f"✅ AI 피드백 분석 완료: {type(feedback_result)}")
                print(f"📋 AI 피드백 결과 미리보기: {list(feedback_result.keys()) if isinstance(feedback_result, dict) else 'dict가 아님'}")
                if isinstance(feedback_result, dict) and 'error' in feedback_result:
                    print(f"❗ AI 분석 오류 내용: {feedback_result['error']}")
            else:
                # 모델이 로드되지 않은 경우 기본 피드백
                print("⚠️ AI 모델이 로드되지 않음, 기본 피드백 사용")
                feedback_result = _generate_basic_feedback(trade_data)
                
        except Exception as e:
            # 오류 시 기본 피드백 생성
            print(f"⚠️ AI 피드백 서비스 오류: {str(e)}")
            print(f"   오류 타입: {type(e).__name__}")
            feedback_result = _generate_basic_feedback(trade_data)
        
        if 'error' in feedback_result:
            # 마지막 시도: 매우 기본적인 피드백
            print("❌ AI 분석 실패, 기본 피드백으로 전환")
            feedback_result = _generate_basic_feedback(trade_data)
        
        # 응답 데이터 구성 (numpy 타입 변환 포함)
        response = _convert_numpy_types({
            "symbol": request.symbol,
            "trade_period": f"{request.entry_date} ~ {request.exit_date}",
            "return_pct": request.return_pct,
            "holding_days": request.holding_period_days,
            
            # AI 모델 예측 결과
            "model_predictions": feedback_result.get('model_predictions', {}),
            
            # 데이터 기반 평가
            "data_driven_evaluation": feedback_result.get('data_driven_evaluation', {}),
            
            # 적응적 인사이트
            "adaptive_insights": feedback_result.get('adaptive_insights', []),
            
            # 학습 기회
            "learning_opportunities": feedback_result.get('learning_opportunities', []),
            
            # SHAP 분석 (상위 기여 요인)
            "shap_analysis": feedback_result.get('shap_analysis', {}),
            
            # 피드백 요약
            "feedback_summary": _generate_feedback_summary(feedback_result),
            
            "timestamp": feedback_result.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            "status": "success"
        })
        
        print(f"✅ 거래 피드백 생성 완료: {request.symbol}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 거래 피드백 생성 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"거래 피드백 생성 중 오류 발생: {str(e)}")

def _convert_numpy_types(obj):
    """numpy 타입을 Python 네이티브 타입으로 변환"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj

def _generate_feedback_summary(feedback_result):
    """피드백 결과를 요약하여 한국어로 생성"""
    summary = {
        "overall_assessment": "분석 완료",
        "key_insights": [],
        "recommendations": []
    }
    
    try:
        # 적응적 인사이트에서 핵심 내용 추출
        if 'adaptive_insights' in feedback_result:
            insights = feedback_result['adaptive_insights']
            for insight in insights[:3]:  # 상위 3개만
                summary["key_insights"].append(insight['message'])
        
        # 학습 기회에서 추천사항 추출
        if 'learning_opportunities' in feedback_result:
            opportunities = feedback_result['learning_opportunities']
            for opp in opportunities[:2]:  # 상위 2개만
                summary["recommendations"].append(opp['learning'])
        
        # 전체 평가
        if 'data_driven_evaluation' in feedback_result:
            eval_data = feedback_result['data_driven_evaluation']
            if 'performance_ranking' in eval_data:
                perf = eval_data['performance_ranking']
                summary["overall_assessment"] = f"실제 성과가 {perf['rank_description']}을 기록했습니다."
    
    except Exception as e:
        print(f"피드백 요약 생성 중 오류: {str(e)}")
    
    return summary

def _generate_basic_feedback(trade_data):
    """기본적인 거래 피드백 생성 (과거 데이터 없이도 작동)"""
    try:
        return_pct = trade_data.get('return_pct', 0)
        holding_days = trade_data.get('holding_period_days', 0)
        symbol = trade_data.get('symbol', 'N/A')
        
        # 기본 인사이트 생성
        insights = []
        
        # 수익률 기반 인사이트
        if return_pct > 20:
            insights.append({
                'type': 'strength',
                'message': f'{return_pct:.1f}% 수익률로 매우 우수한 성과를 거두었습니다.',
                'data_basis': '20% 이상의 높은 수익률'
            })
        elif return_pct > 10:
            insights.append({
                'type': 'strength', 
                'message': f'{return_pct:.1f}% 수익률로 좋은 성과를 거두었습니다.',
                'data_basis': '10% 이상의 양호한 수익률'
            })
        elif return_pct > 0:
            insights.append({
                'type': 'performance',
                'message': f'{return_pct:.1f}% 수익률로 수익을 실현했습니다.',
                'data_basis': '양의 수익률 달성'
            })
        else:
            insights.append({
                'type': 'weakness',
                'message': f'{return_pct:.1f}% 손실이 발생했습니다.',
                'data_basis': '음의 수익률'
            })
        
        # 보유 기간 기반 인사이트
        if holding_days < 7:
            insights.append({
                'type': 'performance',
                'message': f'{holding_days}일의 단기 거래였습니다.',
                'data_basis': '1주 미만 보유'
            })
        elif holding_days > 90:
            insights.append({
                'type': 'performance',
                'message': f'{holding_days}일의 장기 보유 거래였습니다.',
                'data_basis': '3개월 이상 보유'
            })
        
        # 학습 기회
        opportunities = []
        
        if return_pct < 0:
            opportunities.append({
                'area': '손실 관리',
                'learning': '손절 타이밍과 리스크 관리 전략을 재검토해보세요.',
                'confidence': 'rule_based'
            })
        elif return_pct > 15:
            opportunities.append({
                'area': '성공 패턴',
                'learning': '이런 높은 수익률 패턴을 분석하여 재현해보세요.',
                'confidence': 'rule_based'
            })
        
        if holding_days < 3:
            opportunities.append({
                'area': '보유 기간',
                'learning': '너무 짧은 보유 기간이 수익률에 미친 영향을 고려해보세요.',
                'confidence': 'rule_based'
            })
        
        return {
            'trade_info': {
                'symbol': symbol,
                'return_pct': return_pct,
                'holding_days': holding_days,
                'entry_date': trade_data.get('entry_date', 'N/A'),
                'exit_date': trade_data.get('exit_date', 'N/A')
            },
            'adaptive_insights': insights,
            'learning_opportunities': opportunities,
            'feedback_summary': {
                'overall_assessment': f'{symbol} 거래 기본 분석이 완료되었습니다.',
                'key_insights': [insight['message'] for insight in insights[:2]],
                'recommendations': [opp['learning'] for opp in opportunities[:1]]
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_type': 'basic_fallback'
        }
        
    except Exception as e:
        return {
            'error': f'기본 피드백 생성 실패: {str(e)}',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }