from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import sys
import os

# 백엔드 서비스 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.news_sentiment_service import news_sentiment_service

router = APIRouter()

class NewsAnalysisRequest(BaseModel):
    stock_name: str
    news_limit: Optional[int] = 10
    max_days: Optional[int] = 14

class SingleNewsRequest(BaseModel):
    original_stock: str
    news_date: str
    content: str
    positive: Optional[float] = 0.5
    negative: Optional[float] = 0.3
    neutral: Optional[float] = 0.2
    sentiment_score: Optional[float] = 0.2

@router.post("/api/news/sentiment/analyze-stock")
async def analyze_stock_sentiment(request: NewsAnalysisRequest):
    """특정 종목의 뉴스 감정 분석 및 종합 점수 계산"""
    try:
        result = news_sentiment_service.analyze_stock_news_sentiment(
            stock_name=request.stock_name,
            news_limit=request.news_limit,
            max_days=request.max_days
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return {
            "success": True,
            "data": result
        }
    
    except Exception as e:
        print(f"종목 뉴스 감정 분석 API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

@router.post("/api/news/sentiment/analyze-single")
async def analyze_single_news(request: SingleNewsRequest):
    """개별 뉴스의 감정 분석"""
    try:
        news_data = request.dict()
        result = news_sentiment_service.analyze_single_news(news_data)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return {
            "success": True,
            "data": result
        }
    
    except Exception as e:
        print(f"개별 뉴스 감정 분석 API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

@router.get("/api/news/sentiment/stock-news/{stock_name}")
async def get_stock_news(stock_name: str, limit: int = 10):
    """특정 종목의 뉴스 데이터 가져오기"""
    try:
        news_list = news_sentiment_service.get_stock_news(stock_name, limit)
        
        return {
            "success": True,
            "data": {
                "stock_name": stock_name,
                "news_count": len(news_list),
                "news_list": news_list
            }
        }
    
    except Exception as e:
        print(f"종목 뉴스 조회 API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

@router.get("/api/news/sentiment/health")
async def health_check():
    """뉴스 감정 분석 서비스 상태 확인"""
    try:
        # 모델 로드 상태 확인
        if not news_sentiment_service.model_loaded:
            model_status = news_sentiment_service.load_model()
        else:
            model_status = True
        
        # 뉴스 데이터 로드 상태 확인
        if news_sentiment_service.news_data_cache is None:
            data_status = news_sentiment_service.load_news_data()
        else:
            data_status = True
        
        return {
            "success": True,
            "data": {
                "model_loaded": model_status,
                "news_data_loaded": data_status,
                "news_data_count": len(news_sentiment_service.news_data_cache) if news_sentiment_service.news_data_cache is not None else 0
            }
        }
    
    except Exception as e:
        print(f"뉴스 감정 분석 헬스체크 API 오류: {e}")
        return {
            "success": False,
            "error": str(e)
        }