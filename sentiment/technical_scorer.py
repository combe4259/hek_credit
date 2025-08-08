"""
토스뱅크 스타일 기술적 점수화 시스템 (yfinance 기반, 룰 기반 강화) - 수정 완료 버전
- 뉴스 감성분석 점수는 별도 파일에서 산출 후 추후 합산
- 메서드명 통일, 점수 스케일링 일관성 확보
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class TechnicalScorer:
    """토스뱅크 스타일 기술적 분석 점수화"""

    def __init__(self):
        self.cached_data = {}
        self.rate_limit_delay = 0.3
        
        # 티커 매핑 (클래스 B 우선)
        self.ticker_mapping = {
            'BRK': 'BRK-B',  # 버크셔 해서웨이 클래스B (일반 투자자용)
            'BF': 'BF-B'     # 브라운 포만 클래스B (실제 거래량 높음)
        }
        
        # 룰셋 가중치 (투자자 동향 → PER/PBR로 대체)
        self.weights = {
            'analyst_opinion': 0.35,        # 35% (애널리스트 의견 + 목표주가)
            'price_momentum': 0.30,         # 30% (수익률 분석)
            'volume_surge': 0.20,          # 20% (거래량 분석)
            'valuation_metrics': 0.15      # 15% (PER/PBR 밸류에이션) - 투자자 동향 대체
        }
        
        print("🏦 룰셋 기반 기술적 분석 시스템")
        print("✅ yfinance 실제 데이터만 사용")
        print("✅ 투자자 동향 → PER/PBR 밸류에이션으로 대체")
        print("✅ BRK→BRK-B, BF→BF-B 티커 매핑 지원")

    def _map_ticker(self, ticker: str) -> str:
        """티커 매핑 (BRK → BRK-B, BF → BF-B)"""
        mapped = self.ticker_mapping.get(ticker, ticker)
        if mapped != ticker:
            print(f"📈 티커 매핑: {ticker} → {mapped}")
        return mapped

    def _normalize_score(self, raw_score: float) -> float:
        """점수를 0-100 스케일로 정규화"""
        # -100~100 → 0~100 변환
        normalized = (raw_score + 100) / 2
        return max(0, min(100, normalized))

    def analyze_analyst_opinion(self, ticker: str) -> Tuple[float, Dict]:
        """애널리스트 의견 + 목표주가 분석 (yfinance 실제 데이터만)"""
        
        try:
            mapped_ticker = self._map_ticker(ticker)
            stock = yf.Ticker(mapped_ticker)
            info = stock.info
            
            score = 0
            details = {'factors': [], 'data_sources': []}
            
            # 1. 현재 애널리스트 정보 (yfinance에서 실제 제공)
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            target_mean = info.get('targetMeanPrice', 0)
            target_high = info.get('targetHighPrice', 0)
            target_low = info.get('targetLowPrice', 0)
            recommendation_mean = info.get('recommendationMean', 3.0)
            num_analysts = info.get('numberOfAnalystOpinions', 0)
            
            details['data_sources'].append(f"애널리스트 {num_analysts}명 의견")
            
            # 값 유효성 체크
            if not current_price or current_price <= 0:
                details['factors'].append("현재가 정보 없음")
                current_price = None
            if not target_mean or target_mean <= 0:
                details['factors'].append("목표주가 정보 없음")
                target_mean = None
            if not recommendation_mean or not (1 <= recommendation_mean <= 5):
                details['factors'].append("투자의견 정보 없음")
                recommendation_mean = None

            # 2. 목표가 기반 분석
            if target_mean and current_price:
                upside = (target_mean - current_price) / current_price
                
                # 🟢 호재: 목표주가 상승여력
                if upside >= 0.30:  # 30% 이상
                    score += 50
                    details['factors'].append(f"목표가 강력 상승여력 {upside:.1%} (+50점)")
                elif upside >= 0.20:  # 20% 이상
                    score += 35
                    details['factors'].append(f"목표가 상승여력 {upside:.1%} (+35점)")
                elif upside >= 0.10:  # 10% 이상
                    score += 20
                    details['factors'].append(f"목표가 긍정 {upside:.1%} (+20점)")
                
                # 🔴 악재: 목표주가 하락위험
                elif upside <= -0.15:  # 15% 이상 하락위험
                    score -= 40
                    details['factors'].append(f"목표가 강력 하락위험 {upside:.1%} (-40점)")
                elif upside <= -0.05:  # 5% 이상 하락위험
                    score -= 25
                    details['factors'].append(f"목표가 하락위험 {upside:.1%} (-25점)")
                
                # 목표가 컨센서스 신뢰도
                if target_high and target_low:
                    price_range = (target_high - target_low) / target_mean
                    if price_range < 0.25:  # 25% 미만 = 높은 컨센서스
                        score += 15
                        details['factors'].append(f"목표가 컨센서스 높음 (+15점)")
                    elif price_range > 0.80:  # 80% 이상 = 낮은 컨센서스
                        score -= 10
                        details['factors'].append(f"목표가 컨센서스 낮음 (-10점)")
            
            # 3. 투자의견 분석 (구매/판매 의견)
            if recommendation_mean and num_analysts >= 3:  # 최소 3명 이상
                # 🟢 호재: '구매' 의견
                if recommendation_mean <= 1.8:  # Strong Buy
                    score += 40
                    details['factors'].append(f"강력 매수 추천 {recommendation_mean:.1f} (+40점)")
                elif recommendation_mean <= 2.2:  # Buy
                    score += 25
                    details['factors'].append(f"매수 추천 {recommendation_mean:.1f} (+25점)")
                elif recommendation_mean <= 2.8:  # Hold에서 Buy 쪽
                    score += 10
                    details['factors'].append(f"긍정적 추천 {recommendation_mean:.1f} (+10점)")
                
                # 🔴 악재: '판매' 의견
                elif recommendation_mean >= 4.2:  # Strong Sell
                    score -= 40
                    details['factors'].append(f"강력 매도 추천 {recommendation_mean:.1f} (-40점)")
                elif recommendation_mean >= 3.8:  # Sell
                    score -= 25
                    details['factors'].append(f"매도 추천 {recommendation_mean:.1f} (-25점)")
                elif recommendation_mean >= 3.2:  # Hold에서 Sell 쪽
                    score -= 10
                    details['factors'].append(f"부정적 추천 {recommendation_mean:.1f} (-10점)")
            
            # 4. 최근 투자의견 변화 감지
            try:
                recommendations = stock.recommendations
                if recommendations is not None and len(recommendations) > 0:
                    recent_30d = recommendations[recommendations.index >= (datetime.now() - timedelta(days=30))]
                    
                    if len(recent_30d) > 0:
                        details['data_sources'].append(f"최근 30일 추천변화 {len(recent_30d)}건")
                        
                        upgrades = 0
                        downgrades = 0
                        
                        for _, row in recent_30d.iterrows():
                            from_grade = str(row.get('From Grade', '')).lower()
                            to_grade = str(row.get('To Grade', '')).lower()
                            
                            # 🟢 호재: 상향
                            if from_grade and from_grade != 'nan':
                                if ('buy' in to_grade and 'buy' not in from_grade) or \
                                   ('outperform' in to_grade and 'outperform' not in from_grade):
                                    upgrades += 1
                                # 🔴 악재: 하향
                                elif ('sell' in to_grade and 'sell' not in from_grade) or \
                                     ('underperform' in to_grade and 'underperform' not in from_grade):
                                    downgrades += 1
                        
                        if upgrades > 0:
                            bonus = min(upgrades * 25, 50)
                            score += bonus
                            details['factors'].append(f"최근 투자의견 상향 {upgrades}건 (+{bonus}점)")
                        
                        if downgrades > 0:
                            penalty = min(downgrades * 25, 50)
                            score -= penalty
                            details['factors'].append(f"최근 투자의견 하향 {downgrades}건 (-{penalty}점)")
            
            except Exception:
                details['data_sources'].append("투자의견 변화 데이터 없음")
            
            # 0-100 스케일로 정규화
            normalized_score = self._normalize_score(score)
            return normalized_score, details
            
        except Exception as e:
            print(f"⚠️ {ticker} 애널리스트 분석 실패: {e}")
            return 50, {'error': str(e), 'data_sources': ['데이터 수집 실패']}
    
    def analyze_price_momentum(self, ticker: str) -> Tuple[float, Dict]:
        """수익률 분석 (한국식 기준 적용)"""
        
        try:
            mapped_ticker = self._map_ticker(ticker)
            stock = yf.Ticker(mapped_ticker)
            daily_hist = stock.history(period="1y")
            
            if len(daily_hist) < 10:
                return 50, {'error': '주가 데이터 부족'}
            
            score = 0
            details = {'factors': [], 'data_sources': [f"{len(daily_hist)}일 주가 데이터"]}
            
            current_price = daily_hist['Close'].iloc[-1]
            
            # 1. 일주일 수익률
            if len(daily_hist) >= 7:
                week_ago_price = daily_hist['Close'].iloc[-8]
                week_return = (current_price - week_ago_price) / week_ago_price
                
                # 🟢 호재: +30% 이상
                if week_return >= 0.30:
                    score += 60
                    details['factors'].append(f"1주일 +30%↑ 대박 {week_return:.1%} (+60점)")
                elif week_return >= 0.20:
                    score += 40
                    details['factors'].append(f"1주일 강한 상승 {week_return:.1%} (+40점)")
                elif week_return >= 0.10:
                    score += 20
                    details['factors'].append(f"1주일 상승 {week_return:.1%} (+20점)")
                
                # 🔴 악재: -30% 이하
                elif week_return <= -0.30:
                    score -= 60
                    details['factors'].append(f"1주일 -30%↓ 폭락 {week_return:.1%} (-60점)")
                elif week_return <= -0.20:
                    score -= 40
                    details['factors'].append(f"1주일 강한 하락 {week_return:.1%} (-40점)")
                elif week_return <= -0.10:
                    score -= 20
                    details['factors'].append(f"1주일 하락 {week_return:.1%} (-20점)")
            
            # 2. 3개월 수익률
            if len(daily_hist) >= 63:
                quarter_ago_price = daily_hist['Close'].iloc[-64]
                quarter_return = (current_price - quarter_ago_price) / quarter_ago_price
                
                # 🟢 호재: +100% 이상
                if quarter_return >= 1.0:
                    score += 70
                    details['factors'].append(f"3개월 +100%↑ 대박 {quarter_return:.1%} (+70점)")
                elif quarter_return >= 0.50:
                    score += 40
                    details['factors'].append(f"3개월 강한 상승 {quarter_return:.1%} (+40점)")
                elif quarter_return >= 0.30:
                    score += 25
                    details['factors'].append(f"3개월 상승 {quarter_return:.1%} (+25점)")
                
                # 🔴 악재: -50% 이하
                elif quarter_return <= -0.50:
                    score -= 60
                    details['factors'].append(f"3개월 -50%↓ 폭락 {quarter_return:.1%} (-60점)")
                elif quarter_return <= -0.30:
                    score -= 35
                    details['factors'].append(f"3개월 하락 {quarter_return:.1%} (-35점)")
            
            # 3. 1년 수익률
            if len(daily_hist) >= 252:
                year_ago_price = daily_hist['Close'].iloc[-253]
                year_return = (current_price - year_ago_price) / year_ago_price
                
                # 🟢 호재: +100% 이상
                if year_return >= 1.0:
                    score += 50
                    details['factors'].append(f"1년 +100%↑ 대박 {year_return:.1%} (+50점)")
                elif year_return >= 0.50:
                    score += 30
                    details['factors'].append(f"1년 강한 상승 {year_return:.1%} (+30점)")
                
                # 🔴 악재: -50% 이하
                elif year_return <= -0.50:
                    score -= 50
                    details['factors'].append(f"1년 -50%↓ 폭락 {year_return:.1%} (-50점)")
            
            # 4. 당일 수익률
            if len(daily_hist) >= 2:
                daily_return = (current_price - daily_hist['Close'].iloc[-2]) / daily_hist['Close'].iloc[-2]
                
                # 🟢 호재: +5% 이상
                if daily_return >= 0.05:
                    score += 30
                    details['factors'].append(f"당일 +5%↑ 급등 {daily_return:.1%} (+30점)")
                elif daily_return >= 0.03:
                    score += 15
                    details['factors'].append(f"당일 상승 {daily_return:.1%} (+15점)")
                
                # 🔴 악재: -5% 이상 하락
                elif daily_return <= -0.05:
                    score -= 30
                    details['factors'].append(f"당일 -5%↓ 급락 {daily_return:.1%} (-30점)")
                elif daily_return <= -0.03:
                    score -= 15
                    details['factors'].append(f"당일 하락 {daily_return:.1%} (-15점)")
            
            # TODO 항목 추가
            details['factors'].append("TODO: 장시작/마감 1시간 세밀 분석")
            
            normalized_score = self._normalize_score(score)
            return normalized_score, details
            
        except Exception as e:
            print(f"⚠️ {ticker} 수익률 분석 실패: {e}")
            return 50, {'error': str(e)}
    
    def analyze_volume_surge(self, ticker: str) -> Tuple[float, Dict]:
        """거래량 분석 (한국식 기준 적용)"""
        
        try:
            mapped_ticker = self._map_ticker(ticker)
            stock = yf.Ticker(mapped_ticker)
            hist = stock.history(period="2mo")
            
            if len(hist) < 20:
                return 50, {'error': '거래량 데이터 부족'}
            
            score = 0
            details = {'factors': [], 'data_sources': [f"{len(hist)}일 거래량 데이터"]}
            
            current_volume = hist['Volume'].iloc[-1]
            current_price = hist['Close'].iloc[-1]
            
            # 1. 전일 대비 거래량
            if len(hist) >= 2:
                prev_volume = hist['Volume'].iloc[-2]
                if prev_volume > 0:
                    volume_ratio = current_volume / prev_volume
                    
                    # 🟢 호재: 2배 이상 증가
                    if volume_ratio >= 3.0:
                        score += 50
                        details['factors'].append(f"거래량 폭증 {volume_ratio:.1f}배 (+50점)")
                    elif volume_ratio >= 2.0:
                        score += 35
                        details['factors'].append(f"거래량 급증 {volume_ratio:.1f}배 (+35점)")
                    elif volume_ratio >= 1.5:
                        score += 20
                        details['factors'].append(f"거래량 증가 {volume_ratio:.1f}배 (+20점)")
                    
                    # 거래량 급감
                    elif volume_ratio <= 0.5:
                        score -= 15
                        details['factors'].append(f"거래량 급감 {volume_ratio:.1f}배 (-15점)")
            
            # 2. 거래대금 기반 TOP20 추정
            trading_value = current_volume * current_price
            
            if len(hist) >= 20:
                avg_trading_value = (hist['Volume'].iloc[-20:] * hist['Close'].iloc[-20:]).mean()
                
                if avg_trading_value > 0:
                    trading_value_ratio = trading_value / avg_trading_value

                    if trading_value_ratio >= 5.0:
                        score += 15
                        details['factors'].append(f"거래대금 급증 (20일 평균 대비 {trading_value_ratio:.1f}배) (+15점)")
                    elif trading_value_ratio >= 3.0:
                        score += 10
                        details['factors'].append(f"거래대금 증가 (20일 평균 대비 {trading_value_ratio:.1f}배) (+10점)")
            
            # 3. 20일 평균 대비 거래량
            if len(hist) >= 20:
                avg_volume_20d = hist['Volume'].iloc[-20:].mean()
                if avg_volume_20d > 0:
                    avg_ratio = current_volume / avg_volume_20d
                    
                    if avg_ratio >= 3.0:
                        score += 40
                        details['factors'].append(f"20일 평균 대비 급증 {avg_ratio:.1f}배 (+40점)")
                    elif avg_ratio >= 2.0:
                        score += 25
                        details['factors'].append(f"20일 평균 대비 증가 {avg_ratio:.1f}배 (+25점)")
            
            # 4. 연속 고거래량 패턴
            if len(hist) >= 25:
                recent_3d_avg = hist['Volume'].iloc[-3:].mean()
                prev_20d_avg = hist['Volume'].iloc[-23:-3].mean()
                
                if prev_20d_avg > 0 and recent_3d_avg / prev_20d_avg > 1.5:
                    score += 20
                    details['factors'].append(f"3일 연속 고거래량 {recent_3d_avg/prev_20d_avg:.1f}배 (+20점)")
            
            # TODO 항목 추가
            details['factors'].append("TODO: S&P 500 전체 거래량 TOP20 순위 연동")
            
            normalized_score = self._normalize_score(score)
            return normalized_score, details
            
        except Exception as e:
            print(f"⚠️ {ticker} 거래량 분석 실패: {e}")
            return 50, {'error': str(e)}
    
    def analyze_valuation_metrics(self, ticker: str) -> Tuple[float, Dict]:
        """PER/PBR 밸류에이션 분석 (투자자 동향 대체)"""
        
        try:
            mapped_ticker = self._map_ticker(ticker)
            stock = yf.Ticker(mapped_ticker)
            info = stock.info
            
            score = 0
            details = {'factors': [], 'data_sources': ['yfinance 재무 데이터']}
            
            # 1. PER 분석
            per = info.get('trailingPE') or info.get('forwardPE', 0)
            if per and per > 0:
                # 🟢 호재: 저PER (저평가)
                if per <= 10:
                    score += 25
                    details['factors'].append(f"초저PER {per:.1f} 저평가 (+25점)")
                elif per <= 15:
                    score += 15
                    details['factors'].append(f"저PER {per:.1f} 저평가 (+15점)")
                elif per <= 20:
                    score += 5
                    details['factors'].append(f"적정PER {per:.1f} (+5점)")
                
                # 🔴 악재: 고PER (고평가)
                elif per >= 50:
                    score -= 25
                    details['factors'].append(f"초고PER {per:.1f} 고평가 (-25점)")
                elif per >= 30:
                    score -= 15
                    details['factors'].append(f"고PER {per:.1f} 고평가 (-15점)")
                elif per >= 25:
                    score -= 5
                    details['factors'].append(f"높은PER {per:.1f} (-5점)")
            
            # 2. PBR 분석
            pbr = info.get('priceToBook', 0)
            if pbr and pbr > 0:
                # 🟢 호재: 저PBR (저평가)
                if pbr <= 0.8:
                    score += 20
                    details['factors'].append(f"초저PBR {pbr:.1f} 저평가 (+20점)")
                elif pbr <= 1.2:
                    score += 10
                    details['factors'].append(f"저PBR {pbr:.1f} 저평가 (+10점)")
                elif pbr <= 2.0:
                    score += 5
                    details['factors'].append(f"적정PBR {pbr:.1f} (+5점)")
                
                # 🔴 악재: 고PBR (고평가)
                elif pbr >= 5.0:
                    score -= 20
                    details['factors'].append(f"초고PBR {pbr:.1f} 고평가 (-20점)")
                elif pbr >= 3.0:
                    score -= 10
                    details['factors'].append(f"고PBR {pbr:.1f} 고평가 (-10점)")
            
            # 3. 배당수익률
            dividend_yield = info.get('dividendYield', 0)
            if dividend_yield and dividend_yield > 0:
                dividend_pct = dividend_yield * 100
                
                if dividend_pct >= 4.0:
                    score += 15
                    details['factors'].append(f"고배당 {dividend_pct:.1f}% (+15점)")
                elif dividend_pct >= 2.0:
                    score += 8
                    details['factors'].append(f"배당 {dividend_pct:.1f}% (+8점)")
            
            # 4. 부채비율
            debt_to_equity = info.get('debtToEquity', 0)
            if debt_to_equity:
                if debt_to_equity >= 200:
                    score -= 15
                    details['factors'].append(f"고부채비율 {debt_to_equity:.0f}% (-15점)")
                elif debt_to_equity >= 100:
                    score -= 8
                    details['factors'].append(f"높은부채 {debt_to_equity:.0f}% (-8점)")
                elif debt_to_equity <= 30:
                    score += 10
                    details['factors'].append(f"건전부채 {debt_to_equity:.0f}% (+10점)")
            
            # TODO 항목 추가
            details['factors'].append("TODO: 동종업계 평균 PER/PBR 대비 상대 밸류에이션")
            
            normalized_score = self._normalize_score(score)
            return normalized_score, details
            
        except Exception as e:
            print(f"⚠️ {ticker} 밸류에이션 분석 실패: {e}")
            return 50, {'error': str(e), 'data_sources': ['데이터 수집 실패']}
    
    def calculate_total_score(self, ticker: str) -> Dict:
        """단일 티커 룰셋 기반 종합 점수 (메서드명 통일)"""
        
        print(f"🏦 룰셋 기반 기술적 분석: {ticker}")
        
        # 각 영역별 분석
        analyst_score, analyst_details = self.analyze_analyst_opinion(ticker)
        momentum_score, momentum_details = self.analyze_price_momentum(ticker)
        volume_score, volume_details = self.analyze_volume_surge(ticker)
        valuation_score, valuation_details = self.analyze_valuation_metrics(ticker)
        
        # 가중 평균 계산
        total_score = (
            analyst_score * self.weights['analyst_opinion'] +
            momentum_score * self.weights['price_momentum'] +
            volume_score * self.weights['volume_surge'] +
            valuation_score * self.weights['valuation_metrics']
        )
        
        # 카테고리 분류
        if total_score >= 80:
            category = "강력호재"
        elif total_score >= 60:
            category = "호재"
        elif total_score >= 40:
            category = "중립"
        elif total_score >= 20:
            category = "악재"
        else:
            category = "강력악재"
        
        return {
            'ticker': ticker,
            'total_score': round(total_score, 1),
            'category': category,
            'component_scores': {
                'analyst_opinion': round(analyst_score, 1),    # 35%
                'price_momentum': round(momentum_score, 1),    # 30%
                'volume_surge': round(volume_score, 1),        # 20%
                'valuation_metrics': round(valuation_score, 1) # 15%
            },
            'details': {
                'analyst': analyst_details,
                'momentum': momentum_details,
                'volume': volume_details,
                'valuation': valuation_details
            },
            'weights_used': self.weights,
            'note': 'KB국민은행 해커톤용 - yfinance 실제 데이터만 사용',
            'calculated_at': datetime.now().isoformat()
        }
    
    def process_sentiment_data(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """다수 뉴스 데이터에 대해 티커별 룰 적용"""
        print(f"S&P 500 룰 적용: {len(sentiment_df)}개 뉴스")
        
        results = []
        unique_tickers = sentiment_df['ticker'].unique() if 'ticker' in sentiment_df.columns else []
        
        for ticker in unique_tickers:
            if ticker == 'UNKNOWN' or not ticker or pd.isna(ticker):
                continue
            
            try:
                # 메서드명 수정: calculate_total_score 사용
                score_data = self.calculate_total_score(ticker)
                
                # 해당 티커의 모든 뉴스에 적용
                ticker_news = sentiment_df[sentiment_df['ticker'] == ticker]
                
                for _, row in ticker_news.iterrows():
                    result = row.to_dict()
                    result.update({
                        'rule_score': score_data['total_score'],
                        'rule_category': score_data['category'],
                        'analyst_score': score_data['component_scores']['analyst_opinion'],
                        'momentum_score': score_data['component_scores']['price_momentum'], 
                        'volume_score': score_data['component_scores']['volume_surge'],
                        'valuation_score': score_data['component_scores']['valuation_metrics'],
                        'data_quality': 'yfinance_only'
                    })
                    results.append(result)
                
                print(f"✅ {ticker}: {score_data['total_score']:.1f}점 ({score_data['category']})")
                
            except Exception as e:
                print(f"⚠️ {ticker} 분석 실패: {e}")
                
                # 실패한 경우 기본값
                ticker_news = sentiment_df[sentiment_df['ticker'] == ticker]
                for _, row in ticker_news.iterrows():
                    result = row.to_dict()
                    result.update({
                        'rule_score': 50,
                        'rule_category': 'neutral',
                        'analyst_score': 50,
                        'momentum_score': 50,
                        'volume_score': 50,
                        'valuation_score': 50,
                        'data_quality': 'failed'
                    })
                    results.append(result)
        
        df = pd.DataFrame(results)
        print(f"✅ S&P 500 룰 적용 완료: {len(df)}개")
        
        return df
    
    def get_detailed_analysis_report(self, ticker: str) -> Dict:
        """종목별 상세 분석 리포트"""
        score_data = self.calculate_total_score(ticker)  # 메서드명 수정
        
        # 상세 리포트 생성
        report = {
            'ticker': ticker,
            'overall_assessment': {
                'score': score_data['total_score'],
                'category': score_data['category'],
                'recommendation': self._get_recommendation(score_data['total_score'])
            },
            'component_analysis': {
                'analyst_opinion': {
                    'score': score_data['component_scores']['analyst_opinion'],
                    'weight': f"{self.weights['analyst_opinion']*100:.0f}%",
                    'factors': score_data['details']['analyst'].get('factors', [])
                },
                'price_momentum': {
                    'score': score_data['component_scores']['price_momentum'],
                    'weight': f"{self.weights['price_momentum']*100:.0f}%",
                    'factors': score_data['details']['momentum'].get('factors', [])
                },
                'volume_surge': {
                    'score': score_data['component_scores']['volume_surge'],
                    'weight': f"{self.weights['volume_surge']*100:.0f}%",
                    'factors': score_data['details']['volume'].get('factors', [])
                },
                'valuation_metrics': {
                    'score': score_data['component_scores']['valuation_metrics'],
                    'weight': f"{self.weights['valuation_metrics']*100:.0f}%",
                    'factors': score_data['details']['valuation'].get('factors', [])
                }
            },
            'key_insights': self._generate_key_insights(score_data),
            'limitations': [
                "투자자 동향 데이터 부재 (해외 시장 한계)",
                "실시간 뉴스 감정 분석 미포함",
                "동종업계 상대 밸류에이션 비교 제한적",
                "장시작/마감 1시간 세밀 분석 개선 필요"
            ],
            'data_sources': list(set([
                source for detail in score_data['details'].values() 
                for source in detail.get('data_sources', [])
            ])),
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def _get_recommendation(self, score: float) -> str:
        """점수 기반 투자 추천"""
        if score >= 80:
            return "강력 매수 추천"
        elif score >= 60:
            return "매수 고려"
        elif score >= 40:
            return "중립적 관찰"
        elif score >= 20:
            return "매도 고려"
        else:
            return "강력 매도 추천"
    
    def _generate_key_insights(self, score_data: Dict) -> List[str]:
        """핵심 인사이트 생성"""
        insights = []
        scores = score_data['component_scores']
        
        # 가장 강한/약한 요인 식별
        max_score = max(scores.values())
        min_score = min(scores.values())
        
        max_factor = [k for k, v in scores.items() if v == max_score][0]
        min_factor = [k for k, v in scores.items() if v == min_score][0]
        
        factor_names = {
            'analyst_opinion': '애널리스트 의견',
            'price_momentum': '주가 모멘텀',
            'volume_surge': '거래량 급증',
            'valuation_metrics': 'PER/PBR 밸류에이션'
        }
        
        if max_score > 70:
            insights.append(f"✅ 주요 호재: {factor_names[max_factor]} ({max_score:.0f}점)")
        
        if min_score < 30:
            insights.append(f"⚠️ 주요 악재: {factor_names[min_factor]} ({min_score:.0f}점)")
        
        # 밸런스 체크
        positive_factors = sum(1 for score in scores.values() if score > 60)
        negative_factors = sum(1 for score in scores.values() if score < 40)
        
        if positive_factors >= 3:
            insights.append("📈 다면적 호재 요인 존재")
        elif negative_factors >= 3:
            insights.append("📉 다면적 악재 요인 존재")
        else:
            insights.append("⚖️ 호재/악재 요인 혼재")
        
        return insights

# 사용 예시
if __name__ == "__main__":
    scorer = TechnicalScorer()
    
    # 개별 종목 분석 테스트
    print("=" * 60)
    print("🔍 개별 종목 기술적 분석 테스트")
    print("=" * 60)
    
    test_tickers = ['AAPL', 'NVDA', 'MSFT']
    
    for ticker in test_tickers:
        try:
            result = scorer.calculate_total_score(ticker)
            
            print(f"\n📊 {ticker} 분석 결과:")
            print(f"종합 점수: {result['total_score']:.1f}점")
            print(f"평가: {result['category']}")
            
            print(f"\n📈 구성 요소별 점수:")
            for component, score in result['component_scores'].items():
                weight = scorer.weights[component] * 100
                print(f"  {component}: {score:.1f}점 (가중치: {weight:.0f}%)")
            
        except Exception as e:
            print(f"❌ {ticker} 분석 실패: {e}")
        
        print("-" * 40)
    
    print(f"\n{'=' * 60}")
    print("✅ 룰셋 기반 점수화 시스템 테스트 완료")