"""
Market Data Service - Yahoo Finance API를 통한 실시간 데이터 수집
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

class MarketDataService:
    """Yahoo Finance를 통한 실시간 시장 데이터 수집 서비스"""
    
    def __init__(self):
        self.cache = {}  # 캐시 저장소
        self.cache_duration = timedelta(minutes=5)  # 5분 캐시
        
    def get_buy_signal_data(self, ticker: str, position_size_pct: float = 5.0) -> Dict[str, Any]:
        """
        매수 신호 예측에 필요한 모든 데이터 수집
        
        Returns:
            buy_signal_predictor가 필요로 하는 모든 feature 데이터
        """
        try:
            # Yahoo Finance에서 데이터 가져오기
            stock = yf.Ticker(ticker)
            
            # 과거 100일치 가격 데이터
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                raise ValueError(f"No data available for {ticker}")
            
            # 기본 정보
            info = stock.info
            current_price = hist['Close'][-1]
            
            # 1. 모멘텀 계산 (5일, 20일, 60일)
            momentum_5d = (hist['Close'][-1] / hist['Close'][-6] - 1) if len(hist) >= 6 else 0
            momentum_20d = (hist['Close'][-1] / hist['Close'][-21] - 1) if len(hist) >= 21 else 0
            momentum_60d = (hist['Close'][-1] / hist['Close'][-61] - 1) if len(hist) >= 61 else 0
            
            # 2. 이동평균 편차
            ma_5 = hist['Close'].rolling(window=5).mean()
            ma_20 = hist['Close'].rolling(window=20).mean()
            ma_60 = hist['Close'].rolling(window=60).mean()
            
            ma_dev_5d = ((current_price - ma_5.iloc[-1]) / ma_5.iloc[-1]) if not ma_5.empty else 0
            ma_dev_20d = ((current_price - ma_20.iloc[-1]) / ma_20.iloc[-1]) if not ma_20.empty else 0
            ma_dev_60d = ((current_price - ma_60.iloc[-1]) / ma_60.iloc[-1]) if not ma_60.empty else 0
            
            # 3. 변동성 (표준편차)
            returns = hist['Close'].pct_change()
            volatility_5d = returns.tail(5).std() * np.sqrt(252) if len(returns) >= 5 else 0
            volatility_20d = returns.tail(20).std() * np.sqrt(252) if len(returns) >= 20 else 0
            volatility_60d = returns.tail(60).std() * np.sqrt(252) if len(returns) >= 60 else 0
            
            # 4. 거래량 변화율
            avg_vol_5d = hist['Volume'].tail(5).mean() if len(hist) >= 5 else hist['Volume'].mean()
            avg_vol_20d = hist['Volume'].tail(20).mean() if len(hist) >= 20 else hist['Volume'].mean()
            avg_vol_60d = hist['Volume'].tail(60).mean() if len(hist) >= 60 else hist['Volume'].mean()
            avg_vol_prev = hist['Volume'].tail(90).head(30).mean() if len(hist) >= 90 else hist['Volume'].mean()
            
            vol_change_5d = (avg_vol_5d - avg_vol_prev) / avg_vol_prev if avg_vol_prev > 0 else 0
            vol_change_20d = (avg_vol_20d - avg_vol_prev) / avg_vol_prev if avg_vol_prev > 0 else 0
            vol_change_60d = (avg_vol_60d - avg_vol_prev) / avg_vol_prev if avg_vol_prev > 0 else 0
            
            # 5. 펀더멘털 지표
            pe_ratio = info.get('trailingPE', info.get('forwardPE', 20))
            pb_ratio = info.get('priceToBook', 3)
            roe = info.get('returnOnEquity', 0.15) if info.get('returnOnEquity') else 0.15
            operating_margin = info.get('operatingMargins', 0.2) if info.get('operatingMargins') else 0.2
            debt_equity = info.get('debtToEquity', 50) / 100 if info.get('debtToEquity') else 0.5
            earnings_growth = info.get('earningsGrowth', 0.1) if info.get('earningsGrowth') else 0.1
            
            # 6. 52주 최고가 대비 비율
            fifty_two_week_high = info.get('fiftyTwoWeekHigh', current_price * 1.1)
            ratio_52w_high = current_price / fifty_two_week_high if fifty_two_week_high > 0 else 0.9
            
            # 7. VIX 가져오기
            vix_ticker = yf.Ticker("^VIX")
            vix_hist = vix_ticker.history(period="1d")
            current_vix = vix_hist['Close'][-1] if not vix_hist.empty else 20
            
            # 8. 10년 국채 수익률 (TNX)
            tnx_ticker = yf.Ticker("^TNX")
            tnx_hist = tnx_ticker.history(period="1d")
            tnx_yield = tnx_hist['Close'][-1] / 10 if not tnx_hist.empty else 3.5  # TNX는 %*10으로 표시됨
            
            # 9. 시장(S&P 500) 데이터
            spy_ticker = yf.Ticker("SPY")
            spy_hist = spy_ticker.history(start=start_date, end=end_date)
            
            if not spy_hist.empty:
                spy_returns = spy_hist['Close'].pct_change()
                market_ma_return_5d = spy_hist['Close'].tail(5).mean() / spy_hist['Close'].tail(10).head(5).mean() - 1 if len(spy_hist) >= 10 else 0
                market_ma_return_20d = spy_hist['Close'].tail(20).mean() / spy_hist['Close'].tail(40).head(20).mean() - 1 if len(spy_hist) >= 40 else 0
                market_cum_return_5d = (spy_hist['Close'][-1] / spy_hist['Close'][-6] - 1) if len(spy_hist) >= 6 else 0
                market_cum_return_20d = (spy_hist['Close'][-1] / spy_hist['Close'][-21] - 1) if len(spy_hist) >= 21 else 0
                market_volatility_20d = spy_returns.tail(20).std() * np.sqrt(252) if len(spy_returns) >= 20 else 0.15
            else:
                market_ma_return_5d = 0
                market_ma_return_20d = 0
                market_cum_return_5d = 0
                market_cum_return_20d = 0
                market_volatility_20d = 0.15
            
            # 모든 feature를 딕셔너리로 반환
            return {
                'symbol': ticker,
                'current_price': float(current_price),
                'volume': float(hist['Volume'][-1]),
                
                # 기술적 지표
                'entry_momentum_5d': float(momentum_5d),
                'entry_momentum_20d': float(momentum_20d),
                'entry_momentum_60d': float(momentum_60d),
                
                'entry_ma_dev_5d': float(ma_dev_5d),
                'entry_ma_dev_20d': float(ma_dev_20d),
                'entry_ma_dev_60d': float(ma_dev_60d),
                
                'entry_volatility_5d': float(volatility_5d),
                'entry_volatility_20d': float(volatility_20d),
                'entry_volatility_60d': float(volatility_60d),
                
                'entry_vol_change_5d': float(vol_change_5d),
                'entry_vol_change_20d': float(vol_change_20d),
                'entry_vol_change_60d': float(vol_change_60d),
                
                # 펀더멘털
                'entry_pe_ratio': float(pe_ratio) if pe_ratio else 20,
                'entry_pb_ratio': float(pb_ratio) if pb_ratio else 3,
                'entry_roe': float(roe),
                'entry_earnings_growth': float(earnings_growth),
                'entry_operating_margin': float(operating_margin),
                'entry_debt_equity_ratio': float(debt_equity),
                
                # 52주 최고가 대비
                'entry_ratio_52w_high': float(ratio_52w_high),
                
                # 시장 지표
                'entry_vix': float(current_vix),
                'entry_tnx_yield': float(tnx_yield),
                
                # 시장 환경
                'market_entry_ma_return_5d': float(market_ma_return_5d),
                'market_entry_ma_return_20d': float(market_ma_return_20d),
                'market_entry_cum_return_5d': float(market_cum_return_5d),
                'market_entry_cum_return_20d': float(market_cum_return_20d),
                'market_entry_volatility_20d': float(market_volatility_20d),
                
                # 포지션 크기
                'position_size_pct': float(position_size_pct),
                
                # 메타데이터
                'timestamp': datetime.now().isoformat(),
                'data_source': 'yahoo_finance'
            }
            
        except Exception as e:
            print(f"Error fetching buy signal data for {ticker}: {str(e)}")
            raise
    
    def get_sell_signal_data(self, ticker: str, entry_price: float, 
                           entry_date: str, position_size: float = 100) -> Dict[str, Any]:
        """
        매도 신호 예측에 필요한 모든 데이터 수집
        
        Args:
            ticker: 종목 코드
            entry_price: 매수 가격
            entry_date: 매수 날짜 (YYYY-MM-DD)
            position_size: 포지션 크기
            
        Returns:
            sell_signal_predictor가 필요로 하는 모든 feature 데이터
        """
        try:
            # 현재 매수 신호 데이터를 가져와서 exit 데이터로 사용
            current_data = self.get_buy_signal_data(ticker, position_size)
            
            # entry_date를 datetime으로 변환 (타임존 제거)
            entry_dt = pd.to_datetime(entry_date).tz_localize(None)
            current_dt = datetime.now()
            holding_days = (current_dt - entry_dt).days
            
            # Yahoo Finance에서 과거 데이터 가져오기
            stock = yf.Ticker(ticker)
            hist = stock.history(start=entry_dt - timedelta(days=10), end=entry_dt + timedelta(days=10))
            
            # 진입 시점 데이터 (entry_date 근처)
            entry_data = {}
            if not hist.empty:
                # 타임존 제거
                hist.index = hist.index.tz_localize(None)
                # entry_date에 가장 가까운 날짜 찾기
                closest_date = hist.index[hist.index.get_indexer([entry_dt], method='nearest')[0]]
                entry_idx = hist.index.get_loc(closest_date)
                
                # 진입 시점의 지표들 계산
                if entry_idx >= 20:
                    entry_hist = hist.iloc[:entry_idx+1]
                    entry_returns = entry_hist['Close'].pct_change()
                    
                    entry_data = {
                        'momentum_5d': (entry_hist['Close'][-1] / entry_hist['Close'][-6] - 1) if len(entry_hist) >= 6 else 0,
                        'momentum_20d': (entry_hist['Close'][-1] / entry_hist['Close'][-21] - 1) if len(entry_hist) >= 21 else 0,
                        'momentum_60d': (entry_hist['Close'][-1] / entry_hist['Close'][-61] - 1) if len(entry_hist) >= 61 else 0,
                        'volatility_5d': entry_returns.tail(5).std() * np.sqrt(252) if len(entry_returns) >= 5 else 0.2,
                        'volatility_20d': entry_returns.tail(20).std() * np.sqrt(252) if len(entry_returns) >= 20 else 0.2,
                        'volatility_60d': entry_returns.tail(60).std() * np.sqrt(252) if len(entry_returns) >= 60 else 0.2,
                    }
                else:
                    entry_data = {
                        'momentum_5d': 0,
                        'momentum_20d': 0,
                        'momentum_60d': 0,
                        'volatility_5d': 0.2,
                        'volatility_20d': 0.2,
                        'volatility_60d': 0.2,
                    }
            
            # 현재(exit) 시점 데이터는 current_data에서 가져옴
            exit_data = {
                'exit_momentum_5d': current_data['entry_momentum_5d'],
                'exit_momentum_20d': current_data['entry_momentum_20d'],
                'exit_momentum_60d': current_data['entry_momentum_60d'],
                'exit_ma_dev_5d': current_data['entry_ma_dev_5d'],
                'exit_ma_dev_20d': current_data['entry_ma_dev_20d'],
                'exit_ma_dev_60d': current_data['entry_ma_dev_60d'],
                'exit_volatility_5d': current_data['entry_volatility_5d'],
                'exit_volatility_20d': current_data['entry_volatility_20d'],
                'exit_volatility_60d': current_data['entry_volatility_60d'],
                'exit_vix': current_data['entry_vix'],
                'exit_tnx_yield': current_data['entry_tnx_yield'],
                'exit_ratio_52w_high': current_data['entry_ratio_52w_high'],
            }
            
            # 변화량 계산 (exit - entry)
            change_data = {}
            for key in ['momentum_5d', 'momentum_20d', 'momentum_60d', 
                       'volatility_5d', 'volatility_20d', 'volatility_60d']:
                change_key = f'change_{key}'
                exit_key = f'exit_{key}'
                entry_key = key  # entry_data에는 prefix 없이 저장됨
                
                if exit_key in exit_data and entry_key in entry_data:
                    change_data[change_key] = exit_data[exit_key] - entry_data[entry_key]
                elif exit_key in exit_data:
                    change_data[change_key] = exit_data[exit_key]
            
            # 추가 변화 지표
            change_data.update({
                'change_ma_dev_5d': exit_data['exit_ma_dev_5d'] - current_data.get('entry_ma_dev_5d', 0),
                'change_ma_dev_20d': exit_data['exit_ma_dev_20d'] - current_data.get('entry_ma_dev_20d', 0),
                'change_ma_dev_60d': exit_data['exit_ma_dev_60d'] - current_data.get('entry_ma_dev_60d', 0),
                'change_vix': exit_data['exit_vix'] - current_data.get('entry_vix', 20),
                'change_tnx_yield': exit_data['exit_tnx_yield'] - current_data.get('entry_tnx_yield', 3.5),
                'change_ratio_52w_high': exit_data['exit_ratio_52w_high'] - current_data.get('entry_ratio_52w_high', 0.9),
            })
            
            # 시장 수익률
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(start=entry_dt, end=current_dt)
            if not spy_hist.empty and len(spy_hist) > 1:
                market_return = (spy_hist['Close'][-1] / spy_hist['Close'][0] - 1) * 100
            else:
                market_return = 0
            
            # 현재 수익률
            current_return = ((current_data['current_price'] - entry_price) / entry_price) * 100
            excess_return = current_return - market_return
            
            # 모든 feature 결합
            result = {
                'symbol': ticker,
                'entry_price': float(entry_price),
                'current_price': float(current_data['current_price']),
                'holding_days': int(holding_days),
                'position_size_pct': float(position_size / 100),
                'current_return': float(current_return),
                'return_pct': float(current_return),
                
                # Entry 시점 지표들
                **{f'entry_{k}': float(v) for k, v in entry_data.items()},
                'entry_ma_dev_5d': float(current_data.get('entry_ma_dev_5d', 0)),
                'entry_ma_dev_20d': float(current_data.get('entry_ma_dev_20d', 0)),
                'entry_ma_dev_60d': float(current_data.get('entry_ma_dev_60d', 0)),
                'entry_vol_change_5d': float(current_data.get('entry_vol_change_5d', 0)),
                'entry_vol_change_20d': float(current_data.get('entry_vol_change_20d', 0)),
                'entry_vol_change_60d': float(current_data.get('entry_vol_change_60d', 0)),
                'entry_vix': float(current_data.get('entry_vix', 20)),
                'entry_tnx_yield': float(current_data.get('entry_tnx_yield', 3.5)),
                'entry_ratio_52w_high': float(current_data.get('entry_ratio_52w_high', 0.9)),
                
                # Exit 시점 지표들
                **{k: float(v) for k, v in exit_data.items()},
                
                # 변화량 지표들
                **{k: float(v) for k, v in change_data.items()},
                
                # 시장 대비 성과
                'market_return_during_holding': float(market_return),
                'excess_return': float(excess_return),
                
                # 메타데이터
                'timestamp': datetime.now().isoformat(),
                'data_source': 'yahoo_finance'
            }
            
            return result
            
        except Exception as e:
            print(f"Error fetching sell signal data for {ticker}: {str(e)}")
            raise
    
    def get_current_price(self, ticker: str) -> float:
        """현재 가격만 빠르게 가져오기"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if not hist.empty:
                return float(hist['Close'][-1])
            else:
                # fallback to info
                info = stock.info
                return float(info.get('currentPrice', info.get('regularMarketPrice', 0)))
        except:
            return 0.0

# 싱글톤 인스턴스
market_data_service = MarketDataService()