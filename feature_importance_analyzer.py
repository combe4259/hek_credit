#!/usr/bin/env python3
"""
Feature Importance Top 5 Analyzer
XGBoost 모델의 상위 5개 중요 피처와 영향도를 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'

# KB 컬러 팔레트
KB_COLORS = {
    'primary': '#FFB800',
    'secondary': '#1E3A8A', 
    'accent': '#059669',
    'danger': '#DC2626',
    'text': '#1F2937'
}

class FeatureImportanceAnalyzer:
    """XGBoost 모델의 피처 중요도 분석기"""
    
    def __init__(self):
        self.models = {}
        self.feature_importances = {}
        self.feature_impacts = {}
        
    def generate_synthetic_trading_data(self, n_samples=10000):
        """트레이딩 데이터 시뮬레이션 (실제 패턴 기반)"""
        np.random.seed(42)
        
        # 기본 시장 지표들
        market_trend = np.random.uniform(-0.05, 0.05, n_samples)
        volatility = np.random.exponential(0.02, n_samples)
        volume_ratio = np.random.lognormal(0, 0.5, n_samples)
        
        # 기술적 지표들
        rsi = np.random.uniform(20, 80, n_samples)
        macd = np.random.normal(0, 0.01, n_samples)
        bollinger_position = np.random.uniform(-1, 1, n_samples)
        moving_avg_20 = np.random.uniform(-0.1, 0.1, n_samples)
        moving_avg_60 = np.random.uniform(-0.1, 0.1, n_samples)
        
        # 펀더멘털 지표들
        pe_ratio = np.random.lognormal(2.5, 0.5, n_samples)
        eps_growth = np.random.normal(0.1, 0.3, n_samples)
        debt_ratio = np.random.uniform(0.2, 0.8, n_samples)
        roe = np.random.uniform(0.05, 0.25, n_samples)
        
        # 거시경제 지표들
        interest_rate = np.random.uniform(1.5, 4.5, n_samples)
        inflation_rate = np.random.uniform(1.0, 5.0, n_samples)
        gdp_growth = np.random.uniform(-2.0, 6.0, n_samples)
        
        # 시장 감정 지표들
        fear_greed_index = np.random.uniform(20, 80, n_samples)
        news_sentiment = np.random.normal(0, 1, n_samples)
        analyst_recommendations = np.random.uniform(1, 5, n_samples)
        
        # 섹터/산업 지표들
        sector_performance = np.random.normal(0, 0.03, n_samples)
        industry_beta = np.random.uniform(0.5, 1.5, n_samples)
        
        # 리스크 지표들
        value_at_risk = np.random.uniform(0.01, 0.05, n_samples)
        sharpe_ratio = np.random.normal(1, 0.5, n_samples)
        
        # 유동성 지표들
        bid_ask_spread = np.random.exponential(0.001, n_samples)
        turnover_rate = np.random.uniform(0.01, 0.2, n_samples)
        
        # 모멘텀 지표들
        price_momentum_5d = np.random.normal(0, 0.02, n_samples)
        price_momentum_20d = np.random.normal(0, 0.05, n_samples)
        earnings_momentum = np.random.normal(0, 0.1, n_samples)
        
        # 데이터프레임 생성
        features_df = pd.DataFrame({
            # 기본 시장 지표 (높은 중요도)
            'market_trend': market_trend,
            'volatility': volatility,
            'volume_ratio': volume_ratio,
            
            # 기술적 지표 (중간 중요도)
            'rsi': rsi,
            'macd': macd,
            'bollinger_position': bollinger_position,
            'moving_avg_20': moving_avg_20,
            'moving_avg_60': moving_avg_60,
            
            # 펀더멘털 지표 (높은 중요도)
            'pe_ratio': pe_ratio,
            'eps_growth': eps_growth,
            'debt_ratio': debt_ratio,
            'roe': roe,
            
            # 거시경제 지표 (중간 중요도)
            'interest_rate': interest_rate,
            'inflation_rate': inflation_rate,
            'gdp_growth': gdp_growth,
            
            # 시장 감정 지표 (중간 중요도)
            'fear_greed_index': fear_greed_index,
            'news_sentiment': news_sentiment,
            'analyst_recommendations': analyst_recommendations,
            
            # 섹터/산업 지표 (낮은 중요도)
            'sector_performance': sector_performance,
            'industry_beta': industry_beta,
            
            # 리스크 지표 (중간 중요도)
            'value_at_risk': value_at_risk,
            'sharpe_ratio': sharpe_ratio,
            
            # 유동성 지표 (낮은 중요도)
            'bid_ask_spread': bid_ask_spread,
            'turnover_rate': turnover_rate,
            
            # 모멘텀 지표 (높은 중요도)
            'price_momentum_5d': price_momentum_5d,
            'price_momentum_20d': price_momentum_20d,
            'earnings_momentum': earnings_momentum
        })
        
        return features_df
    
    def create_target_variables(self, features_df):
        """타겟 변수 생성 (3개 모델용)"""
        n_samples = len(features_df)
        
        # Buy Signal: 펀더멘털 + 모멘텀 중심
        buy_signal = (
            features_df['eps_growth'] * 0.3 +
            features_df['roe'] * 0.25 +
            features_df['price_momentum_20d'] * 0.2 +
            features_df['market_trend'] * 0.15 +
            features_df['news_sentiment'] * 0.1 +
            np.random.normal(0, 0.1, n_samples)
        )
        
        # Sell Signal: 기술적 지표 + 리스크 중심
        sell_signal = (
            features_df['rsi'] * 0.01 +  # RSI가 높으면 매도 신호
            features_df['volatility'] * 5 +
            features_df['value_at_risk'] * 10 +
            features_df['price_momentum_5d'] * 2 +
            features_df['bollinger_position'] * 0.3 +
            np.random.normal(0, 0.1, n_samples)
        )
        
        # Trade Quality: 전체적 균형
        trade_quality = (
            features_df['sharpe_ratio'] * 0.3 +
            features_df['volume_ratio'] * 0.2 +
            features_df['market_trend'] * 0.2 +
            features_df['eps_growth'] * 0.15 +
            features_df['analyst_recommendations'] * 0.15 +
            np.random.normal(0, 0.1, n_samples)
        )
        
        targets = {
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'trade_quality': trade_quality
        }
        
        return targets
    
    def train_models(self, features_df, targets):
        """3개 모델 학습"""
        print("🔥 3개 XGBoost 모델 학습 중...")
        
        for model_name, y in targets.items():
            print(f"   📊 {model_name} 모델 학습...")
            
            # 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, y, test_size=0.2, random_state=42
            )
            
            # XGBoost 모델 생성 및 학습
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # 성능 평가
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            print(f"      ✅ R² - Train: {train_r2:.4f}, Test: {test_r2:.4f}")
            
            # 모델 저장
            self.models[model_name] = model
            
            # Feature Importance 추출
            importance = model.feature_importances_
            feature_names = features_df.columns.tolist()
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            self.feature_importances[model_name] = importance_df
            
            # 실제 영향도 계산 (예측값 변화량 기준)
            self._calculate_feature_impact(model, X_test, model_name)
        
        print("✅ 모든 모델 학습 완료!")
    
    def _calculate_feature_impact(self, model, X_test, model_name):
        """피처별 실제 영향도 계산 (예측값 변화량 기준)"""
        base_pred = model.predict(X_test)
        impacts = {}
        
        for feature in X_test.columns:
            # 피처 값을 평균으로 변경했을 때 예측값 변화
            X_modified = X_test.copy()
            X_modified[feature] = X_modified[feature].mean()
            
            modified_pred = model.predict(X_modified)
            impact = np.mean(np.abs(base_pred - modified_pred))
            impacts[feature] = impact
        
        # 정규화 (0-100% 범위)
        max_impact = max(impacts.values())
        if max_impact > 0:
            impacts = {k: (v/max_impact)*100 for k, v in impacts.items()}
        
        impact_df = pd.DataFrame(list(impacts.items()), 
                               columns=['feature', 'impact_percent'])
        impact_df = impact_df.sort_values('impact_percent', ascending=False)
        
        self.feature_impacts[model_name] = impact_df
    
    def get_top5_features(self):
        """상위 5개 피처 추출"""
        results = {}
        
        for model_name in self.models.keys():
            importance_df = self.feature_importances[model_name]
            impact_df = self.feature_impacts[model_name]
            
            # 상위 5개 피처
            top5_importance = importance_df.head(5)
            top5_impact = impact_df.head(5)
            
            # 결합된 정보
            combined_info = []
            for _, row in top5_importance.iterrows():
                feature = row['feature']
                importance = row['importance']
                
                # 해당 피처의 영향도 찾기
                impact_row = impact_df[impact_df['feature'] == feature]
                impact_percent = impact_row['impact_percent'].iloc[0] if len(impact_row) > 0 else 0
                
                # 피처 설명 추가
                feature_desc = self._get_feature_description(feature)
                
                combined_info.append({
                    'feature': feature,
                    'description': feature_desc,
                    'importance_score': importance,
                    'impact_percent': impact_percent,
                    'rank': len(combined_info) + 1
                })
            
            results[model_name] = combined_info
            
        return results
    
    def _get_feature_description(self, feature):
        """피처 설명 매핑"""
        descriptions = {
            'market_trend': '시장 트렌드 (전체 시장 방향성)',
            'volatility': '변동성 (가격 변동 폭)',
            'volume_ratio': '거래량 비율 (평균 대비)',
            'rsi': 'RSI 지표 (과매수/과매도)',
            'macd': 'MACD 지표 (모멘텀)',
            'bollinger_position': '볼린저 밴드 위치',
            'moving_avg_20': '20일 이동평균 대비',
            'moving_avg_60': '60일 이동평균 대비',
            'pe_ratio': 'PER (주가수익비율)',
            'eps_growth': 'EPS 성장률',
            'debt_ratio': '부채비율',
            'roe': 'ROE (자기자본이익률)',
            'interest_rate': '기준금리',
            'inflation_rate': '인플레이션율',
            'gdp_growth': 'GDP 성장률',
            'fear_greed_index': '공포탐욕지수',
            'news_sentiment': '뉴스 감정점수',
            'analyst_recommendations': '애널리스트 추천점수',
            'sector_performance': '섹터 성과',
            'industry_beta': '산업 베타',
            'value_at_risk': 'VaR (위험가치)',
            'sharpe_ratio': '샤프 비율',
            'bid_ask_spread': '호가 스프레드',
            'turnover_rate': '회전율',
            'price_momentum_5d': '5일 가격 모멘텀',
            'price_momentum_20d': '20일 가격 모멘텀',
            'earnings_momentum': '수익 모멘텀'
        }
        return descriptions.get(feature, feature)
    
    def visualize_top5_features(self, save_path=None):
        """상위 5개 피처 시각화"""
        top5_results = self.get_top5_features()
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.patch.set_facecolor('white')
        
        model_info = [
            ('buy_signal', 'Buy Signal Model', KB_COLORS['primary']),
            ('sell_signal', 'Sell Signal Model', KB_COLORS['danger']),
            ('trade_quality', 'Trade Quality Model', KB_COLORS['accent'])
        ]
        
        for idx, (model_name, title, color) in enumerate(model_info):
            ax = axes[idx]
            data = top5_results[model_name]
            
            # 데이터 준비
            features = [d['description'][:20] + '...' if len(d['description']) > 20 
                       else d['description'] for d in data]
            importances = [d['importance_score'] for d in data]
            impacts = [d['impact_percent'] for d in data]
            
            # 이중 막대 그래프
            x = np.arange(len(features))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, importances, width, 
                          label='Feature Importance', color=color, alpha=0.7)
            bars2 = ax.bar(x + width/2, impacts, width,
                          label='Impact (%)', color=color, alpha=0.4)
            
            # 값 표시
            for i, (imp, impact) in enumerate(zip(importances, impacts)):
                ax.text(i - width/2, imp + max(importances)*0.01, f'{imp:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
                ax.text(i + width/2, impact + max(impacts)*0.01, f'{impact:.1f}%',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # 스타일링
            ax.set_title(title, fontsize=16, fontweight='bold', color=color, pad=20)
            ax.set_xlabel('Features', fontsize=12, color=KB_COLORS['text'])
            ax.set_ylabel('Score', fontsize=12, color=KB_COLORS['text'])
            ax.set_xticks(x)
            ax.set_xticklabels(features, rotation=45, ha='right', fontsize=10)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('white')
            
            # Y축 범위 조정
            max_val = max(max(importances), max(impacts))
            ax.set_ylim(0, max_val * 1.15)
        
        plt.suptitle('🔍 Top 5 Most Important Features Analysis', 
                    fontsize=20, fontweight='bold', y=0.98, color=KB_COLORS['text'])
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ 시각화 저장: {save_path}")
        
        plt.show()
        
        return top5_results
    
    def print_detailed_results(self):
        """상세 결과 출력"""
        top5_results = self.get_top5_features()
        
        print("\n" + "="*80)
        print("🔍 TOP 5 FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        for model_name, data in top5_results.items():
            print(f"\n🏆 {model_name.upper().replace('_', ' ')} MODEL")
            print("-" * 60)
            
            for item in data:
                print(f"#{item['rank']} {item['feature']}")
                print(f"   📋 설명: {item['description']}")
                print(f"   📊 중요도 점수: {item['importance_score']:.4f}")
                print(f"   📈 실제 영향도: {item['impact_percent']:.1f}%")
                print()
        
        print("="*80)
        print("📝 분석 요약:")
        print("• Feature Importance: XGBoost 모델 내부 중요도 점수")
        print("• Impact %: 해당 피처 변경 시 예측값 변화 정도")
        print("• 높은 값일수록 모델 결정에 더 큰 영향을 미침")
        print("="*80)

def main():
    """메인 실행 함수"""
    print("🚀 Feature Importance Top 5 Analysis 시작")
    print("="*60)
    
    # 분석기 초기화
    analyzer = FeatureImportanceAnalyzer()
    
    # 데이터 생성
    print("📊 트레이딩 데이터 생성 중...")
    features_df = analyzer.generate_synthetic_trading_data(n_samples=15000)
    targets = analyzer.create_target_variables(features_df)
    
    print(f"✅ 데이터 생성 완료: {len(features_df)}개 샘플, {len(features_df.columns)}개 피처")
    
    # 모델 학습
    analyzer.train_models(features_df, targets)
    
    # 결과 출력
    analyzer.print_detailed_results()
    
    # 시각화
    print("\n📈 상위 5개 피처 시각화 생성 중...")
    save_path = '/Users/inter4259/Desktop/Programming/hek_credit/top5_features_analysis.png'
    top5_results = analyzer.visualize_top5_features(save_path)
    
    print("\n✅ 분석 완료!")
    
    return analyzer, top5_results

if __name__ == "__main__":
    analyzer, results = main()