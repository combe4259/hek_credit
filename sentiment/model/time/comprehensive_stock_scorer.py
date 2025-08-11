# =============================================================================
# 종합 종목 호재/악재 점수화 시스템
# 뉴스 신선도 가중치 + 감성분석 + 기술적지표 + 뉴스 빈도
# =============================================================================
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import sys
import os

warnings.filterwarnings('ignore')

class ComprehensiveStockScorer:
    """
    🎯 종목별 종합 호재/악재 점수화 시스템 (0~100점)
    
    구성 요소:
    1. 감성 점수 (40%) - sentiment_score, positive, negative, neutral 활용
    2. 기술적 지표 (30%) - momentum_score, volume_score 활용
    3. 뉴스 빈도 (20%) - 최근 뉴스 활동성 평가
    4. BERT 감성 강도 (10%) - BERT 임베딩 기반 감성 강도
    """

    def __init__(self):
        self.stock_scores = {}
        print("✅ ComprehensiveStockScorer 초기화 완료!")

    def calculate_freshness_weight(self, df_stock):
        """
        각 종목 내에서 상대적 신선도로 가중치 계산
        최신 뉴스 = 1.0, 가장 오래된 뉴스 = 0.0
        """
        df_stock = df_stock.copy()
        df_stock['news_date'] = pd.to_datetime(df_stock['news_date'])
        df_stock = df_stock.sort_values('news_date', ascending=False)  # 최신순 정렬
        
        # 각 종목 내에서 상대적 신선도 계산
        df_stock['days_from_latest'] = (df_stock['news_date'].max() - df_stock['news_date']).dt.days
        max_days = df_stock['days_from_latest'].max()
        
        if max_days == 0:  # 모든 뉴스가 같은 날인 경우
            df_stock['freshness_weight'] = 1.0
        else:
            df_stock['freshness_weight'] = 1 - (df_stock['days_from_latest'] / max_days)
        
        return df_stock

    def calculate_sentiment_component(self, df_stock, verbose=False):
        """
        감성 점수 컴포넌트 계산 (0~100점)
        - sentiment_score, positive, negative 가중 평균
        - 신선도 가중치 적용
        """
        if verbose:
            print("    📊 감성 점수 계산 중...")
        
        weights = df_stock['freshness_weight'].values
        
        # 1. 기본 감성 점수 (sentiment_score)
        sentiment_scores = df_stock['sentiment_score'].fillna(0).values
        weighted_sentiment = np.average(sentiment_scores, weights=weights)
        
        # 2. 감성 비율 점수 (positive - negative)
        positive_scores = df_stock['positive'].fillna(0).values
        negative_scores = df_stock['negative'].fillna(0).values
        neutral_scores = df_stock['neutral'].fillna(0).values
        
        # 각 뉴스별 감성 균형 계산
        emotion_balance = []
        for i in range(len(df_stock)):
            total = positive_scores[i] + negative_scores[i] + neutral_scores[i]
            if total > 0:
                balance = (positive_scores[i] - negative_scores[i]) / total
            else:
                balance = 0
            emotion_balance.append(balance)
        
        weighted_balance = np.average(emotion_balance, weights=weights)
        
        # 3. 감성 강도 (positive + negative)
        emotion_intensity = []
        for i in range(len(df_stock)):
            total = positive_scores[i] + negative_scores[i] + neutral_scores[i]
            if total > 0:
                intensity = (positive_scores[i] + negative_scores[i]) / total
            else:
                intensity = 0
            emotion_intensity.append(intensity)
        
        weighted_intensity = np.average(emotion_intensity, weights=weights)
        
        # 최종 감성 점수 (0~100)
        # sentiment_score: -1~1 범위를 0~100으로 변환
        sentiment_part = (weighted_sentiment + 1) * 50
        # emotion_balance: -1~1 범위를 0~100으로 변환  
        balance_part = (weighted_balance + 1) * 50
        # emotion_intensity: 0~1 범위를 0~100으로 변환
        intensity_part = weighted_intensity * 100
        
        # 가중 평균
        final_sentiment = (
            sentiment_part * 0.5 +    # 기본 감성 점수 50%
            balance_part * 0.4 +      # 감성 균형 40%
            intensity_part * 0.1      # 감성 강도 10%
        )
        
        final_sentiment = np.clip(final_sentiment, 0, 100)
        
        if verbose:
            print(f"      감성 점수: {final_sentiment:.1f}")
            print(f"        기본 감성: {sentiment_part:.1f}")
            print(f"        감성 균형: {balance_part:.1f}")
            print(f"        감성 강도: {intensity_part:.1f}")
        
        return final_sentiment

    def calculate_technical_component(self, df_stock, verbose=False):
        """
        기술적 지표 컴포넌트 계산 (0~100점)
        - momentum_score, volume_score 가중 평균
        - 신선도 가중치 적용
        """
        if verbose:
            print("    📈 기술적 지표 계산 중...")
        
        weights = df_stock['freshness_weight'].values
        
        # momentum_score와 volume_score 가중 평균
        momentum_scores = df_stock['momentum_score'].fillna(50).values  # 기본값 50 (중립)
        volume_scores = df_stock['volume_score'].fillna(50).values      # 기본값 50 (중립)
        
        weighted_momentum = np.average(momentum_scores, weights=weights)
        weighted_volume = np.average(volume_scores, weights=weights)
        
        # 최종 기술적 지표 (0~100)
        final_technical = (
            weighted_momentum * 0.6 +  # 모멘텀 60%
            weighted_volume * 0.4      # 거래량 40%
        )
        
        final_technical = np.clip(final_technical, 0, 100)
        
        if verbose:
            print(f"      기술적 지표: {final_technical:.1f}")
            print(f"        모멘텀: {weighted_momentum:.1f}")
            print(f"        거래량: {weighted_volume:.1f}")
        
        return final_technical

    def calculate_frequency_component(self, df_stock, total_stocks_news_counts, verbose=False):
        """
        뉴스 빈도 컴포넌트 계산 (0~100점)
        - 해당 종목의 뉴스 개수를 전체 종목들과 비교하여 상대적 활성도 평가
        - 뉴스 많음 = 관심도/활성도 높음 = 높은 점수
        """
        if verbose:
            print("    📰 뉴스 빈도 계산 중...")
        
        news_count = len(df_stock)
        
        # 전체 종목들의 뉴스 개수 분포에서 현재 종목의 순위 계산
        rank_percentile = np.sum(np.array(total_stocks_news_counts) <= news_count) / len(total_stocks_news_counts)
        
        # 순위를 0~100점으로 변환
        frequency_score = rank_percentile * 100
        
        # 최신성 가중치 추가 (최근 뉴스 비율이 높으면 보너스)
        recent_ratio = np.sum(df_stock['freshness_weight'] > 0.7) / len(df_stock)
        freshness_bonus = recent_ratio * 10  # 최대 10점 보너스
        
        final_frequency = min(frequency_score + freshness_bonus, 100)
        
        if verbose:
            print(f"      뉴스 빈도: {final_frequency:.1f}")
            print(f"        뉴스 개수: {news_count}개")
            print(f"        상대 순위: {rank_percentile:.1%}")
            print(f"        최신성 보너스: {freshness_bonus:.1f}")
        
        return final_frequency

    def calculate_bert_component(self, df_stock, verbose=False):
        """
        BERT 감성 강도 컴포넌트 계산 (0~100점)
        - BERT 임베딩의 분산을 통해 감성의 일관성/강도 측정
        - 신선도 가중치 적용
        """
        if verbose:
            print("    🤖 BERT 감성 강도 계산 중...")
        
        # BERT 임베딩 컬럼들 추출
        bert_cols = [f'finbert_{i}' for i in range(768) if f'finbert_{i}' in df_stock.columns]
        
        if not bert_cols:
            if verbose:
                print("      BERT 임베딩 없음, 기본값 50 사용")
            return 50.0
        
        weights = df_stock['freshness_weight'].values
        bert_data = df_stock[bert_cols].fillna(0).values
        
        # 가중 평균 BERT 임베딩 계산
        weighted_bert = np.average(bert_data, axis=0, weights=weights)
        
        # BERT 임베딩의 강도 계산 (L2 norm)
        bert_intensity = np.linalg.norm(weighted_bert)
        
        # 정규화 (일반적인 BERT 임베딩 norm 범위: 0~30)
        normalized_intensity = min(bert_intensity / 30 * 100, 100)
        
        if verbose:
            print(f"      BERT 감성 강도: {normalized_intensity:.1f}")
            print(f"        원시 강도: {bert_intensity:.3f}")
        
        return normalized_intensity

    def calculate_stock_score(self, df_stock, stock_name, total_stocks_news_counts, verbose=False):
        """
        종목별 종합 점수 계산 (0~100점)
        """
        if verbose:
            print(f"  🎯 {stock_name} 종합 점수 계산")
        
        # 신선도 가중치 계산
        df_stock = self.calculate_freshness_weight(df_stock)
        
        if verbose:
            print(f"    뉴스 개수: {len(df_stock)}개")
            print(f"    기간: {df_stock['news_date'].min().date()} ~ {df_stock['news_date'].max().date()}")
            print(f"    평균 신선도: {df_stock['freshness_weight'].mean():.3f}")
        
        # 각 컴포넌트 계산
        sentiment_score = self.calculate_sentiment_component(df_stock, verbose)
        technical_score = self.calculate_technical_component(df_stock, verbose)
        frequency_score = self.calculate_frequency_component(df_stock, total_stocks_news_counts, verbose)
        bert_score = self.calculate_bert_component(df_stock, verbose)
        
        # 최종 점수 계산
        final_score = (
            sentiment_score * 0.4 +   # 감성 점수 40%
            technical_score * 0.3 +   # 기술적 지표 30%
            frequency_score * 0.2 +   # 뉴스 빈도 20%
            bert_score * 0.1          # BERT 감성 강도 10%
        )
        
        final_score = np.clip(final_score, 0, 100)
        
        if verbose:
            print(f"    🏆 최종 점수: {final_score:.1f}/100")
            print(f"      감성(40%): {sentiment_score:.1f}")
            print(f"      기술(30%): {technical_score:.1f}")
            print(f"      빈도(20%): {frequency_score:.1f}")
            print(f"      BERT(10%): {bert_score:.1f}")
            print()
        
        return {
            'stock_name': stock_name,
            'final_score': final_score,
            'sentiment_score': sentiment_score,
            'technical_score': technical_score,
            'frequency_score': frequency_score,
            'bert_score': bert_score,
            'news_count': len(df_stock),
            'date_range': f"{df_stock['news_date'].min().date()} ~ {df_stock['news_date'].max().date()}",
            'avg_freshness': df_stock['freshness_weight'].mean()
        }

    def score_all_stocks(self, df, verbose=True):
        """
        모든 종목에 대해 종합 점수 계산
        """
        if verbose:
            print("🎯 전체 종목 호재/악재 점수화 시작")
            print("="*60)
        
        # 날짜 컬럼 처리
        df['news_date'] = pd.to_datetime(df['news_date'])
        
        # 종목별로 그룹화
        stocks = df.groupby('original_stock')
        
        # 전체 종목의 뉴스 개수 분포 계산 (빈도 점수 계산용)
        total_stocks_news_counts = [len(group) for _, group in stocks]
        
        results = []
        
        for stock_name, stock_df in stocks:
            try:
                stock_result = self.calculate_stock_score(
                    stock_df, stock_name, total_stocks_news_counts, verbose
                )
                results.append(stock_result)
                self.stock_scores[stock_name] = stock_result
                
            except Exception as e:
                if verbose:
                    print(f"  ❌ {stock_name} 점수 계산 실패: {e}")
        
        # 결과 DataFrame 생성
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('final_score', ascending=False)
        
        if verbose:
            print("="*60)
            print("📊 전체 종목 점수 요약 (상위 20개)")
            print("="*60)
            print(results_df[['stock_name', 'final_score', 'news_count']].head(20).to_string(index=False))
            
            # 점수 분포 분석
            print(f"\n📈 점수 분포:")
            print(f"  평균: {results_df['final_score'].mean():.1f}")
            print(f"  표준편차: {results_df['final_score'].std():.1f}")
            print(f"  최고점: {results_df['final_score'].max():.1f}")
            print(f"  최저점: {results_df['final_score'].min():.1f}")
            
            # 등급별 분류
            high_scores = (results_df['final_score'] >= 70).sum()
            good_scores = ((results_df['final_score'] >= 60) & (results_df['final_score'] < 70)).sum()
            neutral_scores = ((results_df['final_score'] >= 40) & (results_df['final_score'] < 60)).sum()
            bad_scores = (results_df['final_score'] < 40).sum()
            
            print(f"\n🏆 등급별 분포:")
            print(f"  강력 호재 (70+): {high_scores}개 ({high_scores/len(results_df)*100:.1f}%)")
            print(f"  호재 (60-69): {good_scores}개 ({good_scores/len(results_df)*100:.1f}%)")
            print(f"  중립 (40-59): {neutral_scores}개 ({neutral_scores/len(results_df)*100:.1f}%)")
            print(f"  악재 (0-39): {bad_scores}개 ({bad_scores/len(results_df)*100:.1f}%)")
        
        return results_df

    def get_stock_rating(self, score):
        """점수를 등급으로 변환"""
        if score >= 80:
            return "🔥 매우 강한 호재"
        elif score >= 70:
            return "📈 강한 호재"
        elif score >= 60:
            return "✅ 호재"
        elif score >= 50:
            return "😐 약간 호재"
        elif score >= 40:
            return "😐 중립"
        elif score >= 30:
            return "⚠️ 약간 악재"
        elif score >= 20:
            return "📉 악재"
        else:
            return "💥 강한 악재"

    def save_results(self, results_df, file_path):
        """결과를 CSV 파일로 저장"""
        # 등급 컬럼 추가
        results_df['rating'] = results_df['final_score'].apply(self.get_stock_rating)
        
        # 저장
        results_df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"💾 결과 저장 완료: {file_path}")

# =============================================================================
# 실행부
# =============================================================================
if __name__ == "__main__":
    try:
        # 데이터 로드
        data_path = "/Users/inter4259/Desktop/news_full_features_robust.csv"
        print(f"📁 데이터 로드: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")
        
        df_news = pd.read_csv(data_path)
        print(f"✅ 데이터 로드 완료: {len(df_news):,}개 뉴스")
        print(f"📊 종목 수: {df_news['original_stock'].nunique()}개")
        
        # 점수화 시스템 실행
        scorer = ComprehensiveStockScorer()
        results_df = scorer.score_all_stocks(df_news, verbose=True)
        
        # 결과 저장
        output_path = "/Users/inter4259/Desktop/Programming/hek_credit/sentiment/model/time/stock_comprehensive_scores.csv"
        scorer.save_results(results_df, output_path)
        
        print("\n✅ 종합 종목 점수화 완료!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()