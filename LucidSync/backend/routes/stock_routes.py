from fastapi import APIRouter, Query, HTTPException
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from services.stock_service import get_stocks_list
from schemas.stock_schema import StockItem
import yfinance as yf
import pandas as pd

router = APIRouter(prefix="/api/stock", tags=["Stocks"])

@router.get("/list", response_model=List[StockItem])
def list_stocks(symbols: List[str] = Query(..., description="주식 심볼 리스트")):
    return get_stocks_list(symbols)

@router.get("/historical-data")
async def get_historical_data(symbol: str, date: str):
    """특정 날짜의 역사적 시장 데이터 수집"""
    try:
        # yfinance를 사용해 해당 날짜 주변의 데이터 수집
        ticker = yf.Ticker(symbol)
        
        # 날짜 파싱
        target_date = datetime.strptime(date, '%Y-%m-%d')
        
        # 해당 날짜 전후 60일간의 데이터를 수집 (기술적 지표 계산을 위해)
        start_date = (target_date - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
        end_date = (target_date + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        
        # 주가 데이터
        hist = ticker.history(start=start_date, end=end_date)
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
            
        # 기업 정보
        info = ticker.info
        
        # 해당 날짜에 가장 가까운 데이터 찾기
        target_datetime = pd.to_datetime(date).tz_localize('America/New_York') if pd.to_datetime(date).tz is None else pd.to_datetime(date)
        target_data = hist[hist.index <= target_datetime]
        if target_data.empty:
            target_data = hist.iloc[:1]  # 첫 번째 데이터 사용
        else:
            target_data = target_data.iloc[-1:]  # 마지막 데이터 사용
            
        current_price = float(target_data['Close'].iloc[0])
        
        # 기술적 지표 계산
        hist['MA_5'] = hist['Close'].rolling(window=5).mean()
        hist['MA_20'] = hist['Close'].rolling(window=20).mean()
        hist['MA_60'] = hist['Close'].rolling(window=60).mean()
        
        # 변동성 계산 (20일)
        hist['Returns'] = hist['Close'].pct_change()
        hist['Volatility_20d'] = hist['Returns'].rolling(window=20).std() * (252**0.5)
        
        # 모멘텀 계산
        hist['Momentum_5d'] = hist['Close'].pct_change(5)
        hist['Momentum_20d'] = hist['Close'].pct_change(20)
        hist['Momentum_60d'] = hist['Close'].pct_change(60)
        
        # 52주 최고가 대비 비율
        hist['52W_High'] = hist['High'].rolling(window=252).max()
        hist['Ratio_52W_High'] = hist['Close'] / hist['52W_High']
        
        # 해당 날짜의 지표들
        target_idx = target_data.index[0]
        
        # VIX와 TNX 데이터 수집 (시장 지수)
        try:
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(start=start_date, end=end_date)
            vix_data = vix_hist[vix_hist.index <= target_datetime]
            current_vix = float(vix_data['Close'].iloc[-1]) if not vix_data.empty else 20.0
        except:
            current_vix = 20.0  # 기본값
            
        try:
            tnx = yf.Ticker("^TNX")
            tnx_hist = tnx.history(start=start_date, end=end_date)
            tnx_data = tnx_hist[tnx_hist.index <= target_datetime]
            current_tnx = float(tnx_data['Close'].iloc[-1]) if not tnx_data.empty else 2.5
        except:
            current_tnx = 2.5  # 기본값
        
        # 응답 데이터 구성
        response_data = {
            "symbol": symbol,
            "date": date,
            "current_price": current_price,
            "ma_5d": float(hist.loc[target_idx, 'MA_5']) if pd.notna(hist.loc[target_idx, 'MA_5']) else current_price,
            "ma_20d": float(hist.loc[target_idx, 'MA_20']) if pd.notna(hist.loc[target_idx, 'MA_20']) else current_price,
            "ma_60d": float(hist.loc[target_idx, 'MA_60']) if pd.notna(hist.loc[target_idx, 'MA_60']) else current_price,
            "ma_dev_5d": ((current_price - float(hist.loc[target_idx, 'MA_5'])) / float(hist.loc[target_idx, 'MA_5']) * 100) if pd.notna(hist.loc[target_idx, 'MA_5']) else 0,
            "ma_dev_20d": ((current_price - float(hist.loc[target_idx, 'MA_20'])) / float(hist.loc[target_idx, 'MA_20']) * 100) if pd.notna(hist.loc[target_idx, 'MA_20']) else 0,
            "ma_dev_60d": ((current_price - float(hist.loc[target_idx, 'MA_60'])) / float(hist.loc[target_idx, 'MA_60']) * 100) if pd.notna(hist.loc[target_idx, 'MA_60']) else 0,
            "volatility_5d": float(hist['Returns'].rolling(window=5).std().loc[target_idx] * (252**0.5)) if pd.notna(hist['Returns'].rolling(window=5).std().loc[target_idx]) else 0.2,
            "volatility_20d": float(hist.loc[target_idx, 'Volatility_20d']) if pd.notna(hist.loc[target_idx, 'Volatility_20d']) else 0.2,
            "volatility_60d": float(hist['Returns'].rolling(window=60).std().loc[target_idx] * (252**0.5)) if pd.notna(hist['Returns'].rolling(window=60).std().loc[target_idx]) else 0.2,
            "momentum_5d": float(hist.loc[target_idx, 'Momentum_5d']) if pd.notna(hist.loc[target_idx, 'Momentum_5d']) else 0,
            "momentum_20d": float(hist.loc[target_idx, 'Momentum_20d']) if pd.notna(hist.loc[target_idx, 'Momentum_20d']) else 0,
            "momentum_60d": float(hist.loc[target_idx, 'Momentum_60d']) if pd.notna(hist.loc[target_idx, 'Momentum_60d']) else 0,
            "ratio_52w_high": float(hist.loc[target_idx, 'Ratio_52W_High']) if pd.notna(hist.loc[target_idx, 'Ratio_52W_High']) else 0.5,
            "pe_ratio": info.get('trailingPE', 15.0),
            "pb_ratio": info.get('priceToBook', 2.0),
            "roe": info.get('returnOnEquity', 0.15),
            "earnings_growth": info.get('earningsGrowth', 0.05),
            "vix": current_vix,
            "tnx_yield": current_tnx,
            "volume": float(target_data['Volume'].iloc[0]),
            "volume_avg_20d": float(hist['Volume'].rolling(window=20).mean().loc[target_idx]) if pd.notna(hist['Volume'].rolling(window=20).mean().loc[target_idx]) else float(target_data['Volume'].iloc[0]),
            "timestamp": datetime.now().isoformat()
        }
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch historical data: {str(e)}")

@router.get("/market-returns")
async def get_market_returns(start_date: str, end_date: str):
    """특정 기간의 시장 수익률 계산 (S&P 500 기준)"""
    try:
        # S&P 500 지수 데이터 수집
        sp500 = yf.Ticker("^GSPC")
        
        # 날짜 파싱
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 약간의 버퍼를 둬서 데이터 수집
        buffer_start = (start - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        buffer_end = (end + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        
        hist = sp500.history(start=buffer_start, end=buffer_end)
        
        if hist.empty:
            raise HTTPException(status_code=404, detail="No market data found for the specified period")
        
        # 시작일과 종료일에 가장 가까운 데이터 찾기
        start_datetime = pd.to_datetime(start_date).tz_localize('America/New_York') if pd.to_datetime(start_date).tz is None else pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date).tz_localize('America/New_York') if pd.to_datetime(end_date).tz is None else pd.to_datetime(end_date)
        start_data = hist[hist.index <= start_datetime]
        end_data = hist[hist.index <= end_datetime]
        
        if start_data.empty or end_data.empty:
            raise HTTPException(status_code=404, detail="Insufficient market data for the specified period")
        
        start_price = float(start_data['Close'].iloc[-1])
        end_price = float(end_data['Close'].iloc[-1])
        
        # 시장 수익률 계산
        market_return = ((end_price - start_price) / start_price) * 100
        
        # 기간 중 변동성 계산
        period_data = hist[(hist.index >= start_datetime) & (hist.index <= end_datetime)]
        returns = period_data['Close'].pct_change().dropna()
        volatility = float(returns.std() * (252**0.5)) if len(returns) > 1 else 0.15
        
        # VIX 평균 (시장 공포 지수)
        try:
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(start=buffer_start, end=buffer_end)
            vix_period = vix_hist[(vix_hist.index >= start_datetime) & (vix_hist.index <= end_datetime)]
            avg_vix = float(vix_period['Close'].mean()) if not vix_period.empty else 20.0
        except:
            avg_vix = 20.0
        
        return {
            "start_date": start_date,
            "end_date": end_date,
            "market_return_pct": market_return,
            "market_volatility": volatility,
            "avg_vix": avg_vix,
            "start_price": start_price,
            "end_price": end_price,
            "trading_days": len(period_data),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate market returns: {str(e)}")