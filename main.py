<<<<<<< HEAD
# main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
import uvicorn
import asyncio
from advanced_trading_ai import AdvancedTradingAI

# FastAPI 앱 초기화
app = FastAPI(
    title="매매패턴 AI API",
    description="사용자의 매매 패턴을 분석하고 투자 스타일을 예측하는 AI API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정 (프론트엔드와 연동시 필요)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 AI 모델 인스턴스
ai_model = AdvancedTradingAI()

# Pydantic 모델 정의
class TradeData(BaseModel):
    user_id: str = Field(..., description="사용자 ID")
    profit_rate: float = Field(..., description="수익률 (0.05 = 5%)")
    holding_days: int = Field(..., description="보유일수")
    market_volatility: float = Field(default=0.02, description="시장 변동성")
    market_trend: int = Field(default=0, description="시장 트렌드 (-1: 하락, 0: 횡보, 1: 상승)")
    is_profit_taking: int = Field(default=0, description="수익실현 여부")
    is_loss_cutting: int = Field(default=0, description="손절 여부")
    is_panic_sell: int = Field(default=0, description="패닉 매도 여부")
    is_diamond_hands: int = Field(default=0, description="다이아몬드 핸즈 여부")
    risk_tolerance: float = Field(default=0.5, description="위험 허용도 (0~1)")

class UserTradingHistory(BaseModel):
    user_id: str = Field(..., description="사용자 ID")
    trades: List[TradeData] = Field(..., description="거래 내역 리스트")

class SellPredictionRequest(BaseModel):
    user_id: str = Field(..., description="사용자 ID")
    current_profit_rate: float = Field(..., description="현재 수익률")
    holding_days: int = Field(..., description="현재 보유일수")
    market_volatility: float = Field(default=0.02, description="시장 변동성")
    user_style_probs: Optional[Dict[str, float]] = Field(default=None, description="사용자 스타일 확률")

class SimpleTradeInput(BaseModel):
    user_id: str
    trades: List[Dict[str, float]] = Field(
        ...,
        description="간단한 거래 데이터 [{'profit_rate': 0.05, 'holding_days': 7}, ...]",
        example=[
            {"profit_rate": 0.05, "holding_days": 7},
            {"profit_rate": -0.03, "holding_days": 15},
            {"profit_rate": 0.12, "holding_days": 3}
        ]
    )

class RealtimePredictionRequest(BaseModel):
    ticker: str = Field(..., description="종목 코드")
    stock_name: str = Field(..., description="종목명")
    current_profit_rate: float = Field(..., description="현재 수익률")
    holding_days: int = Field(..., description="보유일수")
    current_time: str = Field(..., description="현재 시간 (HH:MM)")
    sector: str = Field(..., description="섹터")
    market_cap: str = Field(..., description="시가총액 구분")
    daily_volatility: float = Field(default=0.02, description="일일 변동성")
    market_condition: str = Field(default="횡보장", description="시장 상황")

# API 엔드포인트
@app.on_event("startup")
async def startup_event():
    """서버 시작시 AI 모델 훈련"""
    print("🚀 매매패턴 AI API 서버 시작")
    print("🤖 AI 모델 훈련 중... (약 30초 소요)")

    try:
        # 백그라운드에서 모델 훈련 (메서드명 변경: train_model -> train_models)
        result = ai_model.train_models()
        print(f"✅ AI 모델 훈련 완료")
    except Exception as e:
        print(f"❌ 모델 훈련 실패: {e}")

@app.get("/")
async def root():
    """API 기본 정보"""
    return {
        "message": "고급 매매패턴 AI API",
        "version": "2.0.0",
        "status": "running",
        "model_trained": ai_model.is_trained,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "train": "/train",
            "predict_realtime": "/predict/realtime (NEW)",
            "predict_sell": "/predict/sell-probability",
            "quick_analysis": "/quick-analysis",
            "demo": "/demo"
        },
        "features": [
            "실시간 매매 의사결정",
            "종목별 특성 분석",
            "시간대별 패턴 학습",
            "과거 손실 패턴 경고",
            "수익률 구간별 행동 예측"
        ]
    }

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "model_trained": ai_model.is_trained,
        "model_performance": ai_model.model_performance if ai_model.is_trained else None,
        "timestamp": datetime.now().isoformat(),
        "server_uptime": "running"
    }

@app.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    """AI 모델 재훈련"""
    try:
        print("🤖 모델 재훈련 시작...")
        result = ai_model.train_models()  # 메서드명 변경
        return {
            "success": True,
            "message": "모델 훈련 완료",
            "performance": {
                "models_trained": ["sell_probability", "profit_zone", "loss_pattern"],
                "status": "completed"
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 훈련 실패: {str(e)}")


@app.post("/quick-analysis")
async def quick_analysis(trade_input: SimpleTradeInput):
    """간단한 거래 데이터로 빠른 분석"""
    if not ai_model.is_trained:
        raise HTTPException(status_code=400, detail="AI 모델이 훈련되지 않았습니다.")

    try:
        # 간단한 데이터를 상세 형식으로 변환
        detailed_trades = []
        for i, trade in enumerate(trade_input.trades):
            detailed_trade = {
                'user_id': trade_input.user_id,
                'profit_rate': trade['profit_rate'],
                'holding_days': trade['holding_days'],
                'market_volatility': 0.02,  # 기본값
                'market_trend': 1 if trade['profit_rate'] > 0 else -1,
                'is_profit_taking': 1 if trade['profit_rate'] > 0.03 else 0,
                'is_loss_cutting': 1 if trade['profit_rate'] < -0.03 else 0,
                'is_panic_sell': 1 if (trade['profit_rate'] < -0.05 and trade['holding_days'] < 3) else 0,
                'is_diamond_hands': 1 if (trade['profit_rate'] < -0.05 and trade['holding_days'] > 20) else 0,
                'risk_tolerance': 0.7 if trade['profit_rate'] > 0.05 else 0.3
            }
            detailed_trades.append(detailed_trade)

        # AI 예측
        result = ai_model.predict_trading_style(detailed_trades)

        # 간단한 요약 제공
        summary = {
            "투자_스타일": result["predicted_style"],
            "신뢰도": f"{result['confidence']:.1%}",
            "주요_특징": result["style_description"],
            "분석된_거래수": len(trade_input.trades)
        }

        return {
            "user_id": trade_input.user_id,
            "summary": summary,
            "detailed_analysis": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 실패: {str(e)}")

@app.post("/predict/realtime")
async def predict_realtime(request: RealtimePredictionRequest):
    """실시간 매매 의사결정 예측 (새로운 AI 모델)"""
    try:
        # 실시간 예측
        result = ai_model.predict_realtime(
            ticker=request.ticker,
            stock_name=request.stock_name,
            current_profit_rate=request.current_profit_rate,
            holding_days=request.holding_days,
            current_time=request.current_time,
            market_data={
                'sector': request.sector,
                'market_cap': request.market_cap,
                'daily_volatility': request.daily_volatility,
                'market_condition': request.market_condition
            }
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 실패: {str(e)}")

@app.post("/predict/sell-probability")
async def predict_sell_probability(request: SellPredictionRequest):
    """현재 상황에서 매도 확률 예측 (레거시 API - 호환성 유지)"""
    # 기존 API를 새로운 모델로 연결
    try:
        result = ai_model.predict_realtime(
            ticker="UNKNOWN",
            stock_name="Unknown",
            current_profit_rate=request.current_profit_rate,
            holding_days=request.holding_days,
            current_time="10:00",  # 기본값
            market_data={
                'sector': '기타',
                'market_cap': '중형주',
                'daily_volatility': request.market_volatility,
                'market_condition': '횡보장'
            }
        )

        # 기존 응답 형식으로 변환
        sell_prob = float(result['analysis']['sell_probability'].rstrip('%')) / 100
        
        return {
            "user_id": request.user_id,
            "sell_probability": sell_prob,
            "recommendation": result['recommendation']['action'],
            "analysis": {
                "current_situation": {
                    "profit_rate": f"{request.current_profit_rate:.1%}",
                    "holding_days": f"{request.holding_days}일",
                    "market_volatility": f"{request.market_volatility:.1%}"
                },
                "decision_factors": result['recommendation']['reasons'],
                "ai_recommendation": result['recommendation']['summary']
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 실패: {str(e)}")

@app.get("/model/stats")
async def get_model_stats():
    """AI 모델 통계 정보"""
    if not ai_model.is_trained:
        raise HTTPException(status_code=400, detail="AI 모델이 훈련되지 않았습니다.")

    return {
        "models": {
            "sell_probability_model": "XGBoost Classifier",
            "profit_zone_model": "XGBoost Classifier", 
            "loss_pattern_model": "XGBoost Classifier"
        },
        "feature_count": len(ai_model.feature_names) if ai_model.feature_names else 0,
        "feature_names": ai_model.feature_names[:10] if ai_model.feature_names else [],
        "capabilities": [
            "실시간 매도 확률 예측",
            "수익률 구간별 행동 분석",
            "과거 손실 패턴 감지",
            "시간대별 매매 패턴 분석"
        ]
    }

@app.post("/demo")
async def demo_prediction():
    """데모용 예측 (새로운 실시간 예측)"""
    if not ai_model.is_trained:
        raise HTTPException(status_code=400, detail="AI 모델이 훈련되지 않았습니다.")

    try:
        # 데모: 삼성전자 +6.8% 시나리오
        demo_result = ai_model.predict_realtime(
            ticker="005930",
            stock_name="삼성전자",
            current_profit_rate=0.068,
            holding_days=8,
            current_time="14:30",
            market_data={
                'sector': '전자',
                'market_cap': '대형주',
                'daily_volatility': 0.021,
                'market_condition': '상승장'
            }
        )

        return {
            "demo_scenario": {
                "종목": "삼성전자",
                "현재_수익률": "+6.8%",
                "보유일수": "8일",
                "시간": "14:30 (장마감 1시간 전)"
            },
            "ai_analysis": demo_result,
            "message": "실시간 매매 의사결정 AI 데모"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데모 실행 실패: {str(e)}")

# 서버 실행 (개발용)
if __name__ == "__main__":
    print("🚀 매매패턴 AI API 서버 시작...")
    print("📖 API 문서: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("🎯 Demo: http://localhost:8000/demo")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
=======
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import sys

# 모듈 임포트
from data.data_loader import DataLoader
from patterns.pattern_analyzer import PatternAnalyzer
from simulation.trading_simulator import TradingSimulator
from utils.visualization import PatternVisualizer
from config import *

def setup_environment():
    """환경 설정 및 출력 디렉토리 생성"""
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 로깅 설정
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    plt.style.use('seaborn-v0_8')
    
    print(f"""
    ==========================================
       Trading Pattern Analysis for {SYMBOL}
       Period: {PERIOD}
       Output Directory: {os.path.abspath(OUTPUT_DIR)}
    ==========================================
    """)

def main():
    # 환경 설정
    setup_environment()
    
    try:
        # 1. 데이터 로드
        print("\n[1/4] 데이터 로딩 중...")
        data_loader = DataLoader(SYMBOL, PERIOD)
        df = data_loader.load_data()
        
        if df is None or df.empty:
            raise ValueError("데이터 로딩 실패!")
        
        # 2. 기술적 지표 계산
        print("\n[2/4] 기술적 지표 계산 중...")
        df = data_loader.calculate_technical_indicators()
        
        # 3. 트레이딩 시뮬레이션 실행 (패턴 분석기는 시뮬레이터 내부에서 생성됨)
        print("\n[3/4] 트레이딩 시뮬레이션 실행 중...")
        simulator = TradingSimulator(df)
        
        # 모든 프로필에 대해 시뮬레이션 실행
        results = simulator.simulate_all_profiles()
        
        if not results:
            raise ValueError("시뮬레이션 결과가 없습니다!")
        
        # 4. 결과 분석 및 시각화
        print("\n[4/4] 결과 분석 및 저장 중...")
        visualizer = PatternVisualizer(results)
        
        # 데이터셋 요약 정보 출력
        dataset = visualizer.create_dataset_summary()
        
        # 시각화 생성
        visualizer.create_pattern_analysis_chart(OUTPUT_CHART)
        visualizer.create_correlation_heatmap()
        
        # 데이터셋 저장 (ML 분석용)
        visualizer.save_dataset(OUTPUT_CSV, ml_format=True)
        
        print("\n✅ 시뮬레이션이 성공적으로 완료되었습니다!")
        print(f"- 결과 파일: {os.path.abspath(OUTPUT_CSV)}")
        print(f"- 차트 파일: {os.path.abspath(OUTPUT_CHART)}")
        
        return 0
        
    except Exception as e:
        print(f"\nX 오류 발생: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
>>>>>>> 0291587d95dbf44f12dfdf304db759a6b632df10
