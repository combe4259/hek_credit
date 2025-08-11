"""
2단계: BERT 임베딩이 추가된 파일에 yfinance 피처 추가 (완전 수정 버전)
- FlexibleTechnicalScorer 사용으로 데이터 부족 문제 해결
- 시간대 문제 완전 해결
- 매핑된 티커 캐시 누락 문제 해결
- 상세 디버깅 로그로 문제점 정확히 파악
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from tqdm import tqdm
import time
import sys
import os
import joblib
from pandas.tseries.offsets import BDay

# Google Drive 마운트 및 경로 설정
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# FlexibleTechnicalScorer 클래스 (시간대 문제 완전 해결)
class FlexibleTechnicalScorer:
    def __init__(self, data_cache=None):
        self.cached_data = data_cache if data_cache is not None else {}
        self.ticker_mapping = {'BRK': 'BRK-B', 'BF': 'BF-B'}
        self.weights = {'price_momentum': 0.60, 'volume_surge': 0.40}
        print("🏦 유연한 기술적 분석 시스템 (v2.5 - 시간대 문제 해결) 로딩 완료")

    def _map_ticker(self, ticker: str) -> str:
        return self.ticker_mapping.get(ticker, ticker)

    def _get_stock_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """시간대 문제를 완전히 해결한 데이터 추출"""
        if ticker in self.cached_data:
            hist = self.cached_data[ticker]
            try:
                # 🔧 강력한 시간대 처리

                # 1. 캐시 데이터 시간대 처리
                if hist.index.tz is not None:
                    # 캐시 데이터가 tz-aware면 tz-naive로 변환
                    hist_index = hist.index.tz_convert('UTC').tz_localize(None)
                    hist_clean = hist.copy()
                    hist_clean.index = hist_index
                else:
                    hist_clean = hist

                # 2. 입력 날짜도 tz-naive로 통일
                start_clean = start_date
                end_clean = end_date

                if start_date.tzinfo is not None:
                    start_clean = start_date.tz_convert('UTC').tz_localize(None)
                if end_date.tzinfo is not None:
                    end_clean = end_date.tz_convert('UTC').tz_localize(None)

                # 3. 안전한 필터링 (boolean indexing 사용)
                mask = (hist_clean.index >= start_clean) & (hist_clean.index <= end_clean)
                filtered_data = hist_clean[mask]

                return filtered_data

            except Exception as e:
                print(f"⚠️ 데이터 필터링 실패 ({ticker}): {e}")
                # 최후의 수단: 최근 데이터 반환
                try:
                    return hist.tail(min(252, len(hist)))  # 최대 1년치
                except:
                    return pd.DataFrame()

        return pd.DataFrame()

    def _normalize_score(self, raw_score: float) -> float:
        return max(0, min(100, (raw_score + 100) / 2))

    def analyze_price_momentum(self, ticker: str, reference_date=None):
        try:
            mapped_ticker = self._map_ticker(ticker)
            end_date = reference_date or datetime.now()
            start_date = end_date - timedelta(days=365)

            daily_hist = self._get_stock_data(mapped_ticker, start_date, end_date)

            if len(daily_hist) < 10:
                return 50, {'error': f'데이터 너무 부족 ({mapped_ticker}, {len(daily_hist)}일)', 'data_days': len(daily_hist)}

            score = 0
            analysis_details = {'factors': [], 'data_days': len(daily_hist)}

            current_price = daily_hist['Close'].iloc[-1]
            if current_price <= 0:
                return 50, {'error': '현재가가 0 이하'}

            # 단계별 분석 (데이터 양에 따라 적응적으로)

            # 1주일 수익률 (최소 5일 필요)
            if len(daily_hist) >= 5:
                week_start_idx = max(0, len(daily_hist) - 6)
                week_ago_price = daily_hist['Close'].iloc[week_start_idx]
                if week_ago_price > 0:
                    week_return = (current_price - week_ago_price) / week_ago_price
                    if week_return >= 0.05:
                        score += 15
                        analysis_details['factors'].append(f"주간 상승 {week_return:.1%}")
                    elif week_return <= -0.05:
                        score -= 15
                        analysis_details['factors'].append(f"주간 하락 {week_return:.1%}")

            # 1개월 수익률 (최소 15일 필요)
            if len(daily_hist) >= 15:
                month_start_idx = max(0, len(daily_hist) - 21)
                month_ago_price = daily_hist['Close'].iloc[month_start_idx]
                if month_ago_price > 0:
                    month_return = (current_price - month_ago_price) / month_ago_price
                    if month_return >= 0.15:
                        score += 25
                        analysis_details['factors'].append(f"월간 강한 상승 {month_return:.1%}")
                    elif month_return >= 0.05:
                        score += 10
                        analysis_details['factors'].append(f"월간 상승 {month_return:.1%}")
                    elif month_return <= -0.15:
                        score -= 25
                        analysis_details['factors'].append(f"월간 강한 하락 {month_return:.1%}")
                    elif month_return <= -0.05:
                        score -= 10
                        analysis_details['factors'].append(f"월간 하락 {month_return:.1%}")

            # 3개월 수익률 (최소 40일 필요)
            if len(daily_hist) >= 40:
                quarter_start_idx = max(0, len(daily_hist) - 63)
                quarter_ago_price = daily_hist['Close'].iloc[quarter_start_idx]
                if quarter_ago_price > 0:
                    quarter_return = (current_price - quarter_ago_price) / quarter_ago_price
                    if quarter_return >= 0.30:
                        score += 30
                        analysis_details['factors'].append(f"분기 대폭 상승 {quarter_return:.1%}")
                    elif quarter_return <= -0.30:
                        score -= 30
                        analysis_details['factors'].append(f"분기 대폭 하락 {quarter_return:.1%}")

            # 장기 수익률 (최소 100일 필요)
            if len(daily_hist) >= 100:
                long_start_idx = max(0, len(daily_hist) - 252)
                long_ago_price = daily_hist['Close'].iloc[long_start_idx]
                if long_ago_price > 0:
                    long_return = (current_price - long_ago_price) / long_ago_price
                    actual_days = len(daily_hist) - long_start_idx
                    analysis_details['factors'].append(f"{actual_days}일간 수익률 {long_return:.1%}")

                    if long_return >= 0.50:
                        score += 30
                    elif long_return >= 0.20:
                        score += 15
                    elif long_return <= -0.50:
                        score -= 30
                    elif long_return <= -0.20:
                        score -= 15

            # 데이터 부족 표시 (페널티 없음)
            if len(daily_hist) < 50:
                analysis_details['factors'].append(f"제한된 데이터 ({len(daily_hist)}일)")

            normalized_score = self._normalize_score(score)
            return normalized_score, analysis_details

        except Exception as e:
            return 50, {'error': str(e)}

    def analyze_volume_surge(self, ticker: str, reference_date=None):
        try:
            mapped_ticker = self._map_ticker(ticker)
            end_date = reference_date or datetime.now()
            start_date = end_date - timedelta(days=60)

            hist = self._get_stock_data(mapped_ticker, start_date, end_date)

            if len(hist) < 5:
                return 50, {'error': f'거래량 데이터 부족 ({len(hist)}일)', 'data_days': len(hist)}

            analysis_details = {'factors': [], 'data_days': len(hist)}
            current_volume = hist['Volume'].iloc[-1]

            if current_volume <= 0:
                return 50, {'error': '현재 거래량이 0'}

            score = 0

            # 단기 거래량 비교 (최소 5일)
            if len(hist) >= 5:
                recent_avg = hist['Volume'].iloc[-5:].mean()
                if recent_avg > 0:
                    short_ratio = current_volume / recent_avg
                    if short_ratio >= 2.0:
                        score += 25
                        analysis_details['factors'].append(f"5일 평균 대비 {short_ratio:.1f}배")

            # 중기 거래량 비교 (최소 15일)
            if len(hist) >= 15:
                mid_avg = hist['Volume'].iloc[-15:].mean()
                if mid_avg > 0:
                    mid_ratio = current_volume / mid_avg
                    if mid_ratio >= 3.0:
                        score += 35
                        analysis_details['factors'].append(f"15일 평균 대비 {mid_ratio:.1f}배")
                    elif mid_ratio >= 1.5:
                        score += 15
                        analysis_details['factors'].append(f"15일 평균 대비 {mid_ratio:.1f}배")

            # 장기 거래량 비교 (최소 30일)
            if len(hist) >= 30:
                long_avg = hist['Volume'].iloc[-30:].mean()
                if long_avg > 0:
                    long_ratio = current_volume / long_avg
                    if long_ratio >= 5.0:
                        score += 40
                        analysis_details['factors'].append(f"30일 평균 대비 {long_ratio:.1f}배 급증")

            # 거래량 급감 감지
            if len(hist) >= 10:
                recent_avg = hist['Volume'].iloc[-10:].mean()
                if recent_avg > 0 and current_volume < recent_avg * 0.3:
                    score -= 10
                    analysis_details['factors'].append("거래량 급감")

            normalized_score = self._normalize_score(score)
            return normalized_score, analysis_details

        except Exception as e:
            return 50, {'error': str(e)}

    def calculate_total_score(self, ticker: str, reference_date=None):
        """총점 계산 및 상세 정보 제공"""
        momentum_score, momentum_details = self.analyze_price_momentum(ticker, reference_date)
        volume_score, volume_details = self.analyze_volume_surge(ticker, reference_date)

        total_score = (
            momentum_score * self.weights['price_momentum'] +
            volume_score * self.weights['volume_surge']
        )

        # 카테고리 분류
        if total_score >= 80: category = "강력호재"
        elif total_score >= 60: category = "호재"
        elif total_score >= 40: category = "중립"
        elif total_score >= 20: category = "악재"
        else: category = "강력악재"

        return {
            'ticker': ticker,
            'total_score': round(total_score, 1),
            'category': category,
            'component_scores': {
                'price_momentum': round(momentum_score, 1),
                'volume_surge': round(volume_score, 1),
            },
            'details': {
                'momentum': momentum_details,
                'volume': volume_details,
            }
        }

# --- 메인 함수들 ---

def update_cache_with_mappings():
    """캐시에 매핑된 티커 추가 (BRK → BRK-B, BF → BF-B)"""
    cache_path = "/content/drive/MyDrive/yfinance_data_cache.joblib"

    if not os.path.exists(cache_path):
        print("❌ 캐시 파일이 없습니다.")
        return

    try:
        cache = joblib.load(cache_path)
        mappings = {'BRK': 'BRK-B', 'BF': 'BF-B'}

        updated = False
        for original, mapped in mappings.items():
            if original in cache and mapped not in cache:
                cache[mapped] = cache[original].copy()  # 데이터 복사
                print(f"✅ 매핑 추가: {original} → {mapped}")
                updated = True

        if updated:
            joblib.dump(cache, cache_path)
            print(f"💾 캐시 업데이트 완료: {len(cache)}개 티커")
        else:
            print("ℹ️ 매핑할 티커가 없거나 이미 존재합니다.")

    except Exception as e:
        print(f"❌ 캐시 업데이트 실패: {e}")

def fetch_and_cache_yfinance_data(min_date, max_date, return_horizons):
    """캐시 파일을 우선 로드하고, SPY 시장 데이터는 항상 최신으로 가져옵니다."""
    print("📈 yfinance 데이터 캐싱 및 로드 시작...")

    cache_path = "/content/drive/MyDrive/yfinance_data_cache.joblib"

    # 캐시 파일 존재 여부 상세 확인
    print(f"🔍 캐시 파일 경로 확인: {cache_path}")
    print(f"   파일 존재 여부: {os.path.exists(cache_path)}")

    if os.path.exists(cache_path):
        try:
            print("⏳ 캐시 파일 로딩 중...")
            ticker_data_cache = joblib.load(cache_path)
            print(f"✅ 캐시 파일에서 yfinance 데이터 로드 완료! ({len(ticker_data_cache)}개 티커)")

            # 캐시 상태 상세 분석
            if ticker_data_cache:
                print(f"\n📊 캐시 데이터 상세 정보:")

                # 샘플 티커 정보
                sample_ticker = list(ticker_data_cache.keys())[0]
                sample_data = ticker_data_cache[sample_ticker]
                print(f"   📈 샘플 티커 {sample_ticker}:")
                print(f"      - 데이터 행 수: {len(sample_data)}")
                print(f"      - 데이터 기간: {sample_data.index[0].date()} ~ {sample_data.index[-1].date()}")
                print(f"      - 컬럼: {list(sample_data.columns)}")

                # 전체 티커 목록 샘플
                all_tickers = list(ticker_data_cache.keys())
                print(f"   🎯 캐시된 티커 샘플 (처음 10개): {all_tickers[:10]}")

                # 데이터 기간 통계
                date_ranges = []
                for ticker, data in list(ticker_data_cache.items())[:5]:  # 처음 5개만 체크
                    if not data.empty:
                        date_ranges.append((data.index[0], data.index[-1]))

                if date_ranges:
                    earliest_start = min([dr[0] for dr in date_ranges])
                    latest_end = max([dr[1] for dr in date_ranges])
                    print(f"   📅 전체 데이터 기간: {earliest_start.date()} ~ {latest_end.date()}")

                # 뉴스 날짜와의 비교
                print(f"   🗞️  뉴스 날짜 범위: {min_date.date()} ~ {max_date.date()}")

                # 날짜 범위 호환성 체크 (시간대 문제 해결)
                sample_start = sample_data.index[0]
                sample_end = sample_data.index[-1]

                # 시간대 정보 확인 및 통일
                print(f"   🕐 시간대 정보:")
                print(f"      - 캐시 데이터 시간대: {sample_data.index.tz}")
                print(f"      - 뉴스 날짜 시간대: {min_date.tzinfo}")

                # 시간대 통일하여 비교
                try:
                    if sample_data.index.tz is not None:
                        # 캐시가 tz-aware인 경우, 뉴스 날짜를 같은 시간대로 변환
                        min_date_tz = min_date.tz_localize(sample_data.index.tz) if min_date.tzinfo is None else min_date.tz_convert(sample_data.index.tz)
                        max_date_tz = max_date.tz_localize(sample_data.index.tz) if max_date.tzinfo is None else max_date.tz_convert(sample_data.index.tz)
                    else:
                        # 캐시가 tz-naive인 경우, 그대로 비교
                        min_date_tz = min_date.tz_localize(None) if min_date.tzinfo is not None else min_date
                        max_date_tz = max_date.tz_localize(None) if max_date.tzinfo is not None else max_date

                    news_in_range = sample_start <= max_date_tz and sample_end >= min_date_tz
                    print(f"   ✅ 날짜 범위 호환성: {news_in_range}")

                except Exception as tz_error:
                    print(f"   ⚠️ 시간대 비교 실패: {tz_error}")
                    print(f"   📅 캐시 범위: {sample_start} ~ {sample_end}")
                    print(f"   📅 뉴스 범위: {min_date} ~ {max_date}")
                    # 에러가 있어도 계속 진행

            else:
                print("❌ 캐시가 비어있습니다!")
                return None, None

        except Exception as e:
            print(f"❌ 캐시 파일 로드 실패: {e}")
            return None, None
    else:
        print("❌ 캐시 파일이 존재하지 않습니다.")
        print("   data_fetcher.py를 먼저 실행하여 캐시를 생성하세요.")
        return None, None

    # SPY 시장 데이터 로드
    market_hist = None
    market_ticker = 'SPY'
    try:
        hist_start_date = min_date - timedelta(days=400)
        market_hist = yf.Ticker(market_ticker).history(
            start=hist_start_date,
            end=max_date + timedelta(days=max(return_horizons) + 15)
        )
        print("✅ SPY 시장 데이터 로드 완료")
    except Exception as e:
        print(f"❌ SPY 데이터 로드 실패: {e}")
        market_hist = pd.DataFrame()

    return ticker_data_cache, market_hist

def calculate_technical_scores_with_debug(df, ticker_data_cache, debug_mode=True):
    """디버깅 강화된 기술적 분석 점수 계산"""
    print("📈 기술적 분석 점수 계산 중 (FlexibleTechnicalScorer + 디버깅)...")

    scorer = FlexibleTechnicalScorer(data_cache=ticker_data_cache)

    # 디버깅 정보 수집
    results = []
    success_count = 0
    fail_count = 0
    fail_reasons = {}
    score_distribution = {'50점': 0, '50점_외': 0}

    # 캐시 정보 출력
    print(f"📦 캐시 정보: {len(ticker_data_cache)}개 티커")
    print(f"📰 뉴스 정보: {len(df)}개, 날짜 범위: {df['news_date'].min().date()} ~ {df['news_date'].max().date()}")
    print(f"🎯 고유 티커: {df['ticker'].nunique()}개")

    # 처음 5개 항목만 상세 디버깅
    debug_sample_size = 5 if debug_mode else 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="기술적 분석"):
        result = row.to_dict()
        ticker = row['ticker']
        news_date = row['news_date']

        is_debug_sample = idx < debug_sample_size

        try:
            # 날짜 보정
            adjusted_date = news_date - BDay(0)

            if is_debug_sample:
                print(f"\n🔍 디버깅 #{idx+1}: {ticker} ({news_date.date()})")
                print(f"   보정된 날짜: {adjusted_date.date()}")

            # 티커 매핑 확인
            mapped_ticker = scorer._map_ticker(ticker)
            cache_exists = mapped_ticker in ticker_data_cache

            if is_debug_sample:
                print(f"   매핑: {ticker} → {mapped_ticker}")
                print(f"   캐시 존재: {cache_exists}")

            if not cache_exists:
                raise ValueError(f"티커 {mapped_ticker}가 캐시에 없음")

            # 데이터 기간 확인
            hist_data = ticker_data_cache[mapped_ticker]
            hist_start = hist_data.index[0]
            hist_end = hist_data.index[-1]

            if is_debug_sample:
                print(f"   히스토리: {hist_start.date()} ~ {hist_end.date()} ({len(hist_data)}행)")
                # 시간대 안전 비교는 스킵 (FlexibleTechnicalScorer가 처리)
                print(f"   데이터 충분: {len(hist_data) >= 10}")

            # 스코어 계산
            score_data = scorer.calculate_total_score(ticker, reference_date=adjusted_date)

            if is_debug_sample:
                print(f"   결과: rule={score_data['total_score']:.1f}, momentum={score_data['component_scores']['price_momentum']:.1f}, volume={score_data['component_scores']['volume_surge']:.1f}")
                if 'details' in score_data:
                    if 'momentum' in score_data['details'] and 'factors' in score_data['details']['momentum']:
                        print(f"   모멘텀 요인: {score_data['details']['momentum']['factors']}")
                    if 'volume' in score_data['details'] and 'factors' in score_data['details']['volume']:
                        print(f"   거래량 요인: {score_data['details']['volume']['factors']}")

            # 결과 저장
            result.update({
                'rule_score': score_data['total_score'],
                'momentum_score': score_data['component_scores']['price_momentum'],
                'volume_score': score_data['component_scores']['volume_surge']
            })

            # 점수 분포 체크
            if abs(score_data['total_score'] - 50.0) < 0.1:  # 50점에 가까운 경우
                score_distribution['50점'] += 1
            else:
                score_distribution['50점_외'] += 1

            success_count += 1

        except Exception as e:
            error_msg = str(e)
            fail_count += 1

            if error_msg not in fail_reasons:
                fail_reasons[error_msg] = 0
            fail_reasons[error_msg] += 1

            if is_debug_sample:
                print(f"   ❌ 실패: {error_msg}")

            # NaN으로 처리
            result.update({
                'rule_score': np.nan,
                'momentum_score': np.nan,
                'volume_score': np.nan
            })
            score_distribution['50점'] += 1

        results.append(result)

    # 결과 요약
    print(f"\n📊 기술적 분석 결과 요약:")
    print(f"   성공: {success_count}개")
    print(f"   실패: {fail_count}개")
    print(f"   성공률: {success_count/(success_count+fail_count)*100:.1f}%")
    print(f"\n📈 점수 분포:")
    print(f"   50점 (기본값): {score_distribution['50점']}개")
    print(f"   기타 점수: {score_distribution['50점_외']}개")

    if fail_reasons:
        print(f"\n❌ 주요 실패 원인:")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {reason}: {count}건")

    # ✅ score_distribution을 반환하도록 수정
    return pd.DataFrame(results), score_distribution

def calculate_composite_signal_cached(df, ticker_data_cache, market_hist, return_horizons):
    """복합 신호 계산 (시간대 문제 해결)"""
    print("🎯 캐싱된 데이터로 복합 신호 계산 중...")
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="복합 신호 계산"):
        result = row.to_dict()
        ticker = row['ticker']
        news_date = row['news_date']

        adjusted_date = news_date - BDay(0)
        mapped_ticker = {'BRK': 'BRK-B', 'BF': 'BF-B'}.get(ticker, ticker)

        if mapped_ticker in ticker_data_cache and not market_hist.empty:
            hist = ticker_data_cache[mapped_ticker]
            try:
                # 시간대 통일
                target_date = adjusted_date

                # 캐시 데이터 시간대 처리
                if hist.index.tz is not None:
                    if target_date.tzinfo is None:
                        target_date = target_date.tz_localize(hist.index.tz)
                    else:
                        target_date = target_date.tz_convert(hist.index.tz)

                # SPY 데이터 시간대 처리
                market_target_date = target_date
                if market_hist.index.tz is not None:
                    if market_target_date.tzinfo is None:
                        market_target_date = market_target_date.tz_localize(market_hist.index.tz)
                    else:
                        market_target_date = market_target_date.tz_convert(market_hist.index.tz)

                # asof를 사용한 안전한 인덱싱
                hist_slice_start_index = hist.index.asof(target_date)
                market_slice_start_index = market_hist.index.asof(market_target_date)

                if pd.isna(hist_slice_start_index) or pd.isna(market_slice_start_index):
                    raise IndexError("Cannot find matching date in historical data")

                relevant_hist = hist.loc[hist_slice_start_index:]
                market_relevant_hist = market_hist.loc[market_slice_start_index:]

                if len(relevant_hist) > max(return_horizons) and len(market_relevant_hist) > max(return_horizons):
                    news_day_close = relevant_hist.iloc[0]['Close']
                    market_news_day_close = market_relevant_hist.iloc[0]['Close']

                    returns = {}
                    for horizon in return_horizons:
                        if len(relevant_hist) > horizon and len(market_relevant_hist) > horizon:
                            future_close = relevant_hist.iloc[horizon]['Close']
                            market_future_close = market_relevant_hist.iloc[horizon]['Close']
                            if news_day_close > 0 and market_news_day_close > 0:
                                stock_return = (future_close - news_day_close) / news_day_close
                                market_return = (market_future_close - market_news_day_close) / market_news_day_close
                                returns[f'relative_return_{horizon}d'] = stock_return - market_return

                    if returns:
                        result['composite_signal'] = np.mean(list(returns.values()))
                    else:
                        result['composite_signal'] = np.nan
                else:
                    result['composite_signal'] = np.nan
            except Exception:
                result['composite_signal'] = np.nan
        else:
            result['composite_signal'] = np.nan
        results.append(result)

    return pd.DataFrame(results)

def main():
    print("🚀 2단계: FlexibleTechnicalScorer 기반 yfinance 피처 생성 (완전 수정 버전)!")

    # 0. 캐시에 매핑된 티커 추가 (한 번만 실행)
    print("🔧 캐시 매핑 업데이트 중...")
    update_cache_with_mappings()

    # 1. 데이터 로드
    DRIVE_DATA_PATH = "/content/drive/MyDrive/news_with_embeddings.csv"
    try:
        df = pd.read_csv(DRIVE_DATA_PATH, engine='python')
        df['news_date'] = pd.to_datetime(df['news_date'])
        print(f"✅ 뉴스 데이터 로드: {len(df)}개")
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return

    # 2. 캐시 및 시장 데이터 준비
    min_date = df['news_date'].min()
    max_date = df['news_date'].max()
    return_horizons = [1, 3, 7]

    ticker_data_cache, market_hist = fetch_and_cache_yfinance_data(min_date, max_date, return_horizons)
    if ticker_data_cache is None:
        return

    # 3. 기술적 분석 점수 계산 (디버깅 모드)
    print("\n" + "="*60)
    print("📈 기술적 분석 점수 계산 시작")
    print("="*60)

    # ✅ score_distribution을 받도록 수정
    df_scores, score_distribution = calculate_technical_scores_with_debug(df.copy(), ticker_data_cache, debug_mode=True)
    df_with_scores = pd.merge(df, df_scores[['rule_score', 'momentum_score', 'volume_score']], left_index=True, right_index=True)

    # 4. 실패 데이터 제거
    original_count = len(df_with_scores)
    df_processed = df_with_scores.dropna(subset=['rule_score']).reset_index(drop=True)
    print(f"\n✅ 기술적 분석 성공 데이터: {len(df_processed)}개 (실패: {original_count - len(df_processed)}개)")

    if df_processed.empty:
        print("❌ 처리할 유효 데이터가 없습니다.")
        return

    # 5. 점수 분포 상세 분석
    print(f"\n📊 점수 분포 상세 분석:")

    rule_scores = df_processed['rule_score'].dropna()
    momentum_scores = df_processed['momentum_score'].dropna()
    volume_scores = df_processed['volume_score'].dropna()

    print(f"📈 rule_score 분포:")
    print(f"   평균: {rule_scores.mean():.1f}, 표준편차: {rule_scores.std():.1f}")
    print(f"   범위: {rule_scores.min():.1f} ~ {rule_scores.max():.1f}")
    print(f"   50점 정확히: {(rule_scores == 50.0).sum()}개")
    print(f"   50점 근처(49~51): {((rule_scores >= 49) & (rule_scores <= 51)).sum()}개")

    print(f"\n📈 momentum_score 분포:")
    print(f"   평균: {momentum_scores.mean():.1f}, 표준편차: {momentum_scores.std():.1f}")
    print(f"   범위: {momentum_scores.min():.1f} ~ {momentum_scores.max():.1f}")

    print(f"\n📈 volume_score 분포:")
    print(f"   평균: {volume_scores.mean():.1f}, 표준편차: {volume_scores.std():.1f}")
    print(f"   범위: {volume_scores.min():.1f} ~ {volume_scores.max():.1f}")

    # 6. 섹터 및 시가총액 정보 수집
    print("\n🏢 섹터 및 시가총액 정보 수집 중...")
    unique_tickers_list = df_processed['ticker'].unique().tolist()
    sector_map = {}
    market_cap_map = {}

    for t in tqdm(unique_tickers_list, desc="티커 정보 수집"):
        try:
            # 매핑된 티커 사용
            mapped_t = {'BRK': 'BRK-B', 'BF': 'BF-B'}.get(t, t)
            info = yf.Ticker(mapped_t).info
            sector_map[t] = info.get('sector', 'Unknown')
            market_cap_map[t] = info.get('marketCap', None)
        except Exception:
            sector_map[t] = 'Unknown'
            market_cap_map[t] = None
        time.sleep(0.5)

    df_processed['sector'] = df_processed['ticker'].map(sector_map)
    df_processed['market_cap'] = df_processed['ticker'].map(market_cap_map)

    # 7. 복합 신호 계산
    print("\n" + "="*60)
    print("🎯 복합 신호 계산 시작")
    print("="*60)

    df_composite = calculate_composite_signal_cached(df_processed.copy(), ticker_data_cache, market_hist, return_horizons)
    df_processed['composite_signal'] = df_composite['composite_signal']

    # 복합 신호 성공 데이터만 필터링
    df_final = df_processed.dropna(subset=['composite_signal']).reset_index(drop=True)
    print(f"✅ 복합 신호 계산 성공: {len(df_final)}개")

    # 8. 최종 점수 계산 및 분포 확인
    df_final['final_score_10'] = ((df_final['composite_signal'] + 1) * 5).clip(0, 10)

    print(f"\n📊 최종 데이터 품질 확인:")
    print(f"   composite_signal 분포:")
    print(f"     평균: {df_final['composite_signal'].mean():.3f}")
    print(f"     표준편차: {df_final['composite_signal'].std():.3f}")
    print(f"     범위: {df_final['composite_signal'].min():.3f} ~ {df_final['composite_signal'].max():.3f}")

    print(f"   final_score_10 분포:")
    print(f"     평균: {df_final['final_score_10'].mean():.1f}")
    print(f"     표준편차: {df_final['final_score_10'].std():.1f}")
    print(f"     범위: {df_final['final_score_10'].min():.1f} ~ {df_final['final_score_10'].max():.1f}")

    # 9. 최종 파일 저장
    output_file = "/content/drive/MyDrive/news_full_features_robust.csv"
    df_final.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n🎉 최종 파일 저장 완료: {output_file}")
    print(f"📊 최종 컬럼 수: {len(df_final.columns)}")
    print(f"📊 최종 컬럼: {list(df_final.columns)}")
    print(f"📈 최종 데이터: {len(df_final)}개")

    # 10. 성과 요약
    print(f"\n" + "="*60)
    print("🎉 2단계 완료 - 성과 요약")
    print("="*60)
    print(f"✅ 원본 뉴스: {len(df)}개")
    print(f"✅ 기술적 분석 성공: {len(df_processed)}개 ({len(df_processed)/len(df)*100:.1f}%)")
    print(f"✅ 복합 신호 성공: {len(df_final)}개 ({len(df_final)/len(df)*100:.1f}%)")
    print(f"✅ 50점 외 점수 비율: {score_distribution.get('50점_외', 0)/(score_distribution.get('50점', 1) + score_distribution.get('50점_외', 0))*100:.1f}%")
    print(f"🚀 머신러닝 모델 훈련 준비 완료!")

if __name__ == "__main__":
    main()