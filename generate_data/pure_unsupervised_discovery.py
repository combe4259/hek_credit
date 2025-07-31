#!/usr/bin/env python3
"""
순수 비지도학습 패턴 발견
규칙이나 라벨 없이 데이터에서 자연스럽게 나타나는 패턴을 발견
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import os

class PureUnsupervisedDiscovery:
    """완전한 비지도학습으로 패턴을 발견하는 시스템"""
    
    def __init__(self, transactions_path, customers_path):
        print("🧠 순수 비지도학습 시스템 초기화...")
        self.transactions_df = pd.read_csv(transactions_path)
        self.customers_df = pd.read_csv(customers_path)
        self.transactions_df['timestamp'] = pd.to_datetime(self.transactions_df['timestamp'])
        
    def extract_pure_features(self):
        """라벨이나 규칙 없이 순수하게 특징 추출"""
        print("\n📊 거래 특징 추출 중...")
        
        all_features = []
        
        # 각 고객의 거래 패턴을 시계열로 추출
        for customer_id in self.transactions_df['customerID'].unique():
            customer_trades = self.transactions_df[
                self.transactions_df['customerID'] == customer_id
            ].sort_values('timestamp')
            
            if len(customer_trades) < 10:
                continue
            
            # Buy-Sell 매칭으로 실현 손익 계산
            realized_trades = self._match_trades(customer_trades)
            
            # 각 실현 거래에 대한 순수 특징 추출
            for trade in realized_trades:
                features = self._extract_trade_features(trade, customer_trades, realized_trades)
                if features:
                    all_features.append(features)
        
        self.features_df = pd.DataFrame(all_features)
        print(f"✅ {len(self.features_df)}개 거래 특징 추출 완료")
        return self.features_df
    
    def _match_trades(self, trades):
        """Buy-Sell 매칭"""
        results = []
        positions = defaultdict(list)
        
        for _, trade in trades.iterrows():
            if trade['transactionType'] == 'Buy':
                positions[trade['ISIN']].append({
                    'buy_date': trade['timestamp'],
                    'buy_price': trade['totalValue'] / trade['units'],
                    'units': trade['units'],
                    'isin': trade['ISIN']
                })
            elif trade['transactionType'] == 'Sell' and positions[trade['ISIN']]:
                sell_price = trade['totalValue'] / trade['units']
                sell_date = trade['timestamp']
                
                position = positions[trade['ISIN']].pop(0)
                holding_days = (sell_date - position['buy_date']).days
                return_rate = (sell_price - position['buy_price']) / position['buy_price'] * 100
                
                results.append({
                    'customer_id': trade['customerID'],
                    'isin': trade['ISIN'],
                    'buy_date': position['buy_date'],
                    'sell_date': sell_date,
                    'holding_days': holding_days,
                    'return_rate': return_rate,
                    'trade_value': position['units'] * sell_price
                })
        
        return results
    
    def _extract_trade_features(self, trade, all_trades, all_results):
        """거래의 순수 특징 추출 (규칙 없이)"""
        
        # 이전 거래들
        past_results = [r for r in all_results if r['sell_date'] < trade['sell_date']]
        if len(past_results) < 3:
            return None
        
        # 1. 현재 거래의 기본 특징
        features = {
            # 수익률과 보유기간
            'return_rate': trade['return_rate'],
            'holding_days': trade['holding_days'],
            'daily_return': trade['return_rate'] / trade['holding_days'] if trade['holding_days'] > 0 else 0,
            
            # 수익률의 제곱, 세제곱 (비선형 관계 포착)
            'return_squared': trade['return_rate'] ** 2,
            'return_cubed': trade['return_rate'] ** 3,
            
            # 로그 변환 (왜도 보정)
            'log_holding_days': np.log1p(trade['holding_days']),
            'abs_return': abs(trade['return_rate']),
        }
        
        # 2. 과거 통계와의 관계 (순수 통계)
        past_returns = [r['return_rate'] for r in past_results]
        past_holdings = [r['holding_days'] for r in past_results]
        
        features.update({
            # 이동 평균
            'return_ma3': np.mean(past_returns[-3:]) if len(past_returns) >= 3 else 0,
            'holding_ma3': np.mean(past_holdings[-3:]) if len(past_holdings) >= 3 else 0,
            
            # 이동 표준편차
            'return_std3': np.std(past_returns[-3:]) if len(past_returns) >= 3 else 0,
            'holding_std3': np.std(past_holdings[-3:]) if len(past_holdings) >= 3 else 0,
            
            # 현재값과 이동평균의 차이
            'return_diff_ma': trade['return_rate'] - np.mean(past_returns[-5:]) if len(past_returns) >= 5 else 0,
            'holding_diff_ma': trade['holding_days'] - np.mean(past_holdings[-5:]) if len(past_holdings) >= 5 else 0,
            
            # Z-score (표준화된 거리)
            'return_zscore': (trade['return_rate'] - np.mean(past_returns)) / (np.std(past_returns) + 1e-6),
            'holding_zscore': (trade['holding_days'] - np.mean(past_holdings)) / (np.std(past_holdings) + 1e-6),
        })
        
        # 3. 시계열 특징
        # 이전 거래와의 간격 계산
        prev_trades = all_trades[all_trades['timestamp'] < trade['buy_date']]
        if not prev_trades.empty:
            last_trade_time = prev_trades['timestamp'].max()
            days_since_last = (trade['buy_date'] - last_trade_time).days
        else:
            days_since_last = 30  # 첫 거래인 경우
        
        features.update({
            'days_since_last_trade': days_since_last,
            'log_days_since': np.log1p(days_since_last),
            
            # 거래 빈도 (최근 30일)
            'trades_last_30d': len([t for t in all_trades['timestamp'] 
                                   if (trade['buy_date'] - t).days <= 30 and t < trade['buy_date']]),
            
            # 시간적 특징
            'buy_month': trade['buy_date'].month,
            'buy_dayofweek': trade['buy_date'].dayofweek,
            'sell_month': trade['sell_date'].month,
            'sell_dayofweek': trade['sell_date'].dayofweek,
        })
        
        # 4. 연속성 특징
        if len(past_results) >= 2:
            # 연속 수익/손실
            last_two_returns = [r['return_rate'] for r in past_results[-2:]]
            features['consecutive_profit'] = all(r > 0 for r in last_two_returns)
            features['consecutive_loss'] = all(r < 0 for r in last_two_returns)
            features['return_momentum'] = last_two_returns[-1] - last_two_returns[-2]
        
        # 5. 상대적 크기
        features['trade_size_ratio'] = trade['trade_value'] / np.mean([r['trade_value'] for r in past_results])
        
        # 6. 변동성
        if len(past_returns) >= 5:
            features['return_volatility'] = np.std(past_returns[-5:])
            features['holding_volatility'] = np.std(past_holdings[-5:])
        
        # 7. 추가 비선형 특징
        features['return_holding_interaction'] = trade['return_rate'] * trade['holding_days']
        features['return_per_sqrt_days'] = trade['return_rate'] / np.sqrt(trade['holding_days'] + 1)
        
        return features
    
    def discover_patterns(self, n_components=20):
        """PCA + K-means로 패턴 발견"""
        print("\n🔍 패턴 발견 중...")
        
        # 특징 정규화
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['customer_id', 'isin', 'buy_date', 'sell_date']]
        
        X = self.features_df[feature_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA로 차원 축소
        pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"PCA 설명 분산: {pca.explained_variance_ratio_.sum():.2%}")
        print("주요 성분별 기여도:")
        for i in range(min(5, len(pca.components_))):
            print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.2%}")
        
        # 최적 클러스터 수 찾기
        silhouette_scores = []
        K = range(3, min(20, len(X) // 50))
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_pca)
            score = silhouette_score(X_pca, labels)
            silhouette_scores.append(score)
            print(f"k={k}: 실루엣 점수 = {score:.3f}")
        
        # 최적 k 선택
        optimal_k = K[np.argmax(silhouette_scores)]
        print(f"\n최적 클러스터 수: {optimal_k}")
        
        # 최종 클러스터링
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        self.features_df['cluster'] = kmeans.fit_predict(X_pca)
        
        # 클러스터 분석
        self._analyze_discovered_patterns(X_pca, pca, scaler, feature_cols)
        
        return self.features_df
    
    def _analyze_discovered_patterns(self, X_pca, pca, scaler, feature_cols):
        """발견된 패턴 분석 (사후 해석)"""
        print("\n📊 발견된 패턴 분석...")
        
        for cluster_id in sorted(self.features_df['cluster'].unique()):
            cluster_data = self.features_df[self.features_df['cluster'] == cluster_id]
            
            print(f"\n=== 클러스터 {cluster_id} ({len(cluster_data)}개 거래) ===")
            
            # 1. 기본 통계
            print("\n기본 통계:")
            print(f"  평균 수익률: {cluster_data['return_rate'].mean():.2f}%")
            print(f"  평균 보유기간: {cluster_data['holding_days'].mean():.1f}일")
            print(f"  수익률 분포: Q1={cluster_data['return_rate'].quantile(0.25):.1f}%, "
                  f"중앙값={cluster_data['return_rate'].median():.1f}%, "
                  f"Q3={cluster_data['return_rate'].quantile(0.75):.1f}%")
            
            # 2. 주요 특징 찾기 (평균과의 차이)
            print("\n특징적인 속성 (전체 평균 대비):")
            overall_means = self.features_df[feature_cols].mean()
            cluster_means = cluster_data[feature_cols].mean()
            
            # 표준화된 차이 계산
            std_diff = (cluster_means - overall_means) / (self.features_df[feature_cols].std() + 1e-6)
            top_features = std_diff.abs().nlargest(5)
            
            for feature in top_features.index:
                cluster_val = cluster_means[feature]
                overall_val = overall_means[feature]
                diff_pct = (cluster_val - overall_val) / (abs(overall_val) + 1e-6) * 100
                
                if abs(diff_pct) > 20:  # 20% 이상 차이나는 특징만
                    print(f"  - {feature}: {cluster_val:.2f} (전체: {overall_val:.2f}, "
                          f"차이: {diff_pct:+.1f}%)")
            
            # 3. PCA 공간에서의 위치
            cluster_indices = cluster_data.index
            cluster_pca_points = X_pca[cluster_indices]
            centroid = cluster_pca_points.mean(axis=0)
            
            print(f"\nPCA 공간 중심점: PC1={centroid[0]:.2f}, PC2={centroid[1]:.2f}")
            
            # 4. 행동 패턴 해석 (데이터 기반)
            interpretation = self._interpret_cluster(cluster_data, overall_means)
            print(f"\n해석: {interpretation}")
    
    def _interpret_cluster(self, cluster_data, overall_means):
        """클러스터의 자연스러운 해석 (데이터 기반)"""
        interpretations = []
        
        # 수익률 기반
        avg_return = cluster_data['return_rate'].mean()
        if avg_return > overall_means['return_rate'] * 1.5:
            interpretations.append("높은 수익률")
        elif avg_return < overall_means['return_rate'] * 0.5:
            interpretations.append("낮은 수익률")
        
        # 보유기간 기반
        avg_holding = cluster_data['holding_days'].mean()
        if avg_holding > overall_means['holding_days'] * 1.5:
            interpretations.append("장기 보유")
        elif avg_holding < overall_means['holding_days'] * 0.5:
            interpretations.append("단기 거래")
        
        # 변동성 기반
        if 'return_volatility' in cluster_data.columns:
            if cluster_data['return_volatility'].mean() > overall_means['return_volatility'] * 1.3:
                interpretations.append("높은 변동성")
        
        # Z-score 기반
        if cluster_data['return_zscore'].abs().mean() > 1.5:
            interpretations.append("극단적 거래")
        
        return " + ".join(interpretations) if interpretations else "추가 분석 필요"
    
    def visualize_patterns(self, output_dir='output'):
        """패턴 시각화"""
        os.makedirs(output_dir, exist_ok=True)
        
        # t-SNE로 2D 시각화
        print("\n📈 패턴 시각화 중...")
        
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['customer_id', 'isin', 'buy_date', 'sell_date', 'cluster']]
        
        X = self.features_df[feature_cols].fillna(0)
        X_scaled = StandardScaler().fit_transform(X)
        
        # 샘플링 (너무 많으면)
        if len(X) > 5000:
            sample_idx = np.random.choice(len(X), 5000, replace=False)
            X_scaled_sample = X_scaled[sample_idx]
            clusters_sample = self.features_df['cluster'].iloc[sample_idx]
        else:
            X_scaled_sample = X_scaled
            clusters_sample = self.features_df['cluster']
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_scaled_sample)
        
        # 시각화
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                            c=clusters_sample, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('거래 패턴 클러스터 (t-SNE 시각화)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        # 클러스터 중심 표시
        for cluster_id in clusters_sample.unique():
            cluster_points = X_tsne[clusters_sample == cluster_id]
            center = cluster_points.mean(axis=0)
            plt.annotate(f'C{cluster_id}', center, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pattern_clusters_tsne.png'))
        plt.close()
        
        print(f"✅ 시각화 저장: {output_dir}/pattern_clusters_tsne.png")
    
    def save_results(self, output_dir='output'):
        """결과 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 패턴 데이터 저장
        patterns_path = os.path.join(output_dir, 'discovered_patterns.csv')
        self.features_df.to_csv(patterns_path, index=False, encoding='utf-8-sig')
        
        # 2. 클러스터 요약 저장
        summary = []
        for cluster_id in sorted(self.features_df['cluster'].unique()):
            cluster_data = self.features_df[self.features_df['cluster'] == cluster_id]
            summary.append({
                'cluster': cluster_id,
                'size': len(cluster_data),
                'avg_return': cluster_data['return_rate'].mean(),
                'std_return': cluster_data['return_rate'].std(),
                'avg_holding': cluster_data['holding_days'].mean(),
                'return_q25': cluster_data['return_rate'].quantile(0.25),
                'return_median': cluster_data['return_rate'].median(),
                'return_q75': cluster_data['return_rate'].quantile(0.75)
            })
        
        summary_df = pd.DataFrame(summary)
        summary_path = os.path.join(output_dir, 'cluster_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        
        print(f"\n💾 결과 저장:")
        print(f"  - 패턴 데이터: {patterns_path}")
        print(f"  - 클러스터 요약: {summary_path}")


def main():
    # 데이터 경로
    transactions_path = "/Users/inter4259/Downloads/FAR-Trans/transactions.csv"
    customers_path = "/Users/inter4259/Downloads/FAR-Trans/customer_information.csv"
    
    # 순수 비지도학습 실행
    discoverer = PureUnsupervisedDiscovery(transactions_path, customers_path)
    
    # 1. 특징 추출
    features = discoverer.extract_pure_features()
    
    # 2. 패턴 발견
    patterns = discoverer.discover_patterns()
    
    # 3. 시각화
    discoverer.visualize_patterns()
    
    # 4. 저장
    discoverer.save_results()
    
    print("\n✅ 순수 비지도학습 완료!")
    print("규칙 없이 데이터에서 자연스럽게 발견된 패턴을 확인하세요.")


if __name__ == "__main__":
    main()