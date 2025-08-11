import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
from itertools import product, combinations
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MegaLabelingOptimizer:
    """
    초대규모 라벨링 방식 테스트 시스템
    
    목표: 수백 가지 라벨링 공식과 수천 가지 가중치 조합 테스트
    """
    
    def __init__(self, data_path=None):
        self.data_path = data_path or '../results/final/trading_episodes_with_rebuilt_market_component.csv'
        self.df = None
        self.test_results = []
        
    def load_data(self, sample_size=None):
        """데이터 로드"""
        print("📊 데이터 로드 중...")
        self.df = pd.read_csv(self.data_path)
        
        if sample_size and len(self.df) > sample_size:
            self.df = self.df.sample(sample_size, random_state=42)
        
        self.df = self.df.fillna(0)
        print(f"  총 데이터: {len(self.df):,}개")
        return self.df

    def create_mega_labeling_variants(self):
        """수백 가지 라벨링 방식들 생성"""
        
        df = self.df
        
        # 기본 성분들
        components = {
            # 수익률 변형들
            'return_raw': df['return_pct'],
            'return_log': np.log(1 + np.abs(df['return_pct'])/100) * np.sign(df['return_pct']),
            'return_sqrt': np.sqrt(np.abs(df['return_pct'])) * np.sign(df['return_pct']),
            'return_squared': df['return_pct'] ** 2 * np.sign(df['return_pct']),
            'return_cubed': df['return_pct'] ** 3,
            
            # 리스크 조정들
            'risk_adj_vol': df['return_pct'] / np.maximum(df['entry_volatility_20d'], 0.01),
            'risk_adj_vol2': df['return_pct'] / np.maximum(df['entry_volatility_20d']**2, 0.01),
            'risk_adj_vol_sqrt': df['return_pct'] / np.maximum(np.sqrt(df['entry_volatility_20d']), 0.01),
            'risk_adj_exit_vol': df['return_pct'] / np.maximum(df['exit_volatility_20d'], 0.01),
            'risk_adj_avg_vol': df['return_pct'] / np.maximum((df['entry_volatility_20d'] + df['exit_volatility_20d'])/2, 0.01),
            
            # 시간 효율성들
            'time_eff_linear': df['return_pct'] / np.maximum(df['holding_period_days'], 1),
            'time_eff_log': df['return_pct'] / np.maximum(np.log(df['holding_period_days'] + 1), 0.01),
            'time_eff_sqrt': df['return_pct'] / np.maximum(np.sqrt(df['holding_period_days']), 0.01),
            'time_eff_squared': df['return_pct'] / np.maximum(df['holding_period_days']**2, 0.01),
            'time_decay': df['return_pct'] * np.exp(-df['holding_period_days']/30),
            
            # 시장 상대성들
            'market_rel_simple': df['return_pct'] - df['market_return_during_holding'],
            'market_rel_ratio': df['return_pct'] / np.maximum(np.abs(df['market_return_during_holding']), 0.01),
            'market_beta_adj': df['return_pct'] - df['market_return_during_holding'] * 1.2,
            'market_outperf': np.maximum(df['return_pct'] - df['market_return_during_holding'], 0),
            'market_correlation': df['return_pct'] * (1 - np.abs(df['market_return_during_holding'])/10),
            
            # 포지션 및 가격 위치들
            'position_adj': df['return_pct'] * np.log(df['position_size_pct'] + 1),
            'price_pos_adj': df['return_pct'] * (100 - df['entry_ratio_52w_high']) / 100,
            'price_momentum_adj': df['return_pct'] * (1 + df['entry_momentum_20d']/100),
            'vix_opportunity': df['return_pct'] * (1 + np.clip(df['entry_vix'] - 20, 0, 30)/100),
            'vol_regime': df['return_pct'] * np.where(df['entry_volatility_20d'] > 30, 0.8, 1.2),
            
            # 고급 지표들
            'sharpe_like': df['return_pct'] / np.maximum(df['entry_volatility_20d'], 0.01),
            'sortino_like': df['return_pct'] / np.maximum(np.where(df['return_pct'] < 0, np.abs(df['return_pct']), 1), 0.01),
            'calmar_like': df['return_pct'] / np.maximum(np.abs(np.minimum(df['return_pct'], 0)), 0.01),
            'sterling_like': (df['return_pct'] + 10) / np.maximum(np.abs(np.minimum(df['return_pct'], 0)), 0.01),
            'martin_like': df['return_pct'] / np.maximum(np.sqrt(np.abs(np.minimum(df['return_pct'], 0))), 0.01),
            
            # 변동성 기반들
            'vol_scaled_return': df['return_pct'] * (20 / np.maximum(df['entry_volatility_20d'], 1)),
            'vol_momentum': df['return_pct'] * (df['entry_volatility_5d'] / np.maximum(df['entry_volatility_20d'], 0.01)),
            'vol_change_adj': df['return_pct'] * (1 - np.abs(df['change_volatility_5d'])/10),
            'vol_efficiency': df['return_pct'] / (df['entry_volatility_20d'] * df['holding_period_days']),
            
            # 모멘텀 기반들  
            'momentum_weighted': df['return_pct'] * (1 + df['entry_momentum_20d']/50),
            'momentum_contrarian': df['return_pct'] * (1 - df['entry_momentum_20d']/50),
            'momentum_change': df['return_pct'] * (1 + df['change_momentum_20d']/100),
            'momentum_strength': df['return_pct'] * np.abs(df['entry_momentum_20d'])/10,
            
            # 기술적 지표들
            'ma_position': df['return_pct'] * (1 + df['entry_ma_dev_20d']/10),
            'ma_trend_follow': df['return_pct'] * np.where(df['entry_ma_dev_20d'] > 0, 1.1, 0.9),
            'ma_mean_revert': df['return_pct'] * np.where(df['entry_ma_dev_20d'] < -5, 1.2, 0.8),
            'ma_volatility': df['return_pct'] / np.maximum(np.abs(df['entry_ma_dev_20d']), 0.01),
            
            # 시장 환경 조정들
            'vix_adjusted': df['return_pct'] * (30 / np.maximum(df['entry_vix'], 1)),
            'vix_contrarian': df['return_pct'] * (df['entry_vix'] / 20),
            'yield_adjusted': df['return_pct'] * (1 + df['entry_tnx_yield']/10),
            'macro_score': df['return_pct'] * (1 + (df['entry_vix'] - df['exit_vix'])/10),
            
            # 손실 페널티들
            'downside_penalty_light': np.where(df['return_pct'] < 0, df['return_pct'] * 1.2, df['return_pct']),
            'downside_penalty_medium': np.where(df['return_pct'] < 0, df['return_pct'] * 1.5, df['return_pct']),
            'downside_penalty_heavy': np.where(df['return_pct'] < 0, df['return_pct'] * 2.0, df['return_pct']),
            'asymmetric_utility': np.where(df['return_pct'] < 0, df['return_pct'] * 2.5, df['return_pct'] * 0.8),
            
            # 수익 가속들
            'upside_boost_light': np.where(df['return_pct'] > 0, df['return_pct'] * 1.1, df['return_pct']),
            'upside_boost_medium': np.where(df['return_pct'] > 0, df['return_pct'] * 1.3, df['return_pct']),
            'upside_boost_heavy': np.where(df['return_pct'] > 0, df['return_pct'] * 1.5, df['return_pct']),
            
            # 복합 리스크 지표들
            'comprehensive_risk': df['return_pct'] / (df['entry_volatility_20d'] * np.sqrt(df['holding_period_days']) * (df['entry_ratio_52w_high']/100 + 0.1)),
            'multi_factor_alpha': (df['return_pct'] - df['market_return_during_holding']) / np.maximum(df['entry_volatility_20d'], 1),
            'risk_parity_score': df['return_pct'] / np.maximum(df['entry_volatility_20d'] * df['position_size_pct']/100, 0.01),
            'kelly_approx': df['return_pct'] * (df['return_pct'] / np.maximum(df['entry_volatility_20d']**2, 0.01)),
            
            # 행동재무학 기반들
            'overconfidence_adj': df['return_pct'] * np.where(df['position_size_pct'] > 5, 0.9, 1.1),
            'disposition_effect': df['return_pct'] * np.where(df['holding_period_days'] < 5, 0.8, 1.0),
            'anchoring_bias': df['return_pct'] * (1 - df['entry_ratio_52w_high']/200),
            'herding_penalty': df['return_pct'] * np.where(df['entry_vix'] < 15, 0.9, 1.0),
            
            # 정보 비율들
            'info_ratio_simple': (df['return_pct'] - df['market_return_during_holding']) / np.maximum(np.std(df['return_pct'] - df['market_return_during_holding']), 0.01),
            'tracking_error_adj': df['return_pct'] / np.maximum(np.abs(df['return_pct'] - df['market_return_during_holding']), 0.01),
            'active_return': np.maximum(df['return_pct'] - df['market_return_during_holding'], 0) * 2,
        }
        
        # 수백 가지 복합 지표 생성
        labeling_methods = {}
        
        # 1. 단일 성분들
        for name, formula in components.items():
            labeling_methods[f"single_{name}"] = {
                'formula': lambda df, f=formula: f,
                'description': f'단일 성분: {name}',
                'components': [name],
                'weights': [1.0]
            }
        
        # 2. 2개 성분 조합들 (핵심 조합들만)
        key_components = [
            'return_raw', 'risk_adj_vol', 'time_eff_linear', 'market_rel_simple', 
            'price_pos_adj', 'sharpe_like', 'downside_penalty_medium', 'momentum_weighted'
        ]
        
        weight_combos_2 = [
            [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5],
            [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]
        ]
        
        for i, comp1 in enumerate(key_components):
            for j, comp2 in enumerate(key_components[i+1:], i+1):
                for weights in weight_combos_2:
                    name = f"combo2_{comp1[:8]}_{comp2[:8]}_{weights[0]:.1f}_{weights[1]:.1f}"
                    labeling_methods[name] = {
                        'formula': lambda df, c1=comp1, c2=comp2, w=weights: components[c1] * w[0] + components[c2] * w[1],
                        'description': f'2성분: {comp1}({weights[0]}) + {comp2}({weights[1]})',
                        'components': [comp1, comp2],
                        'weights': weights
                    }
        
        # 3. 3개 성분 조합들
        weight_combos_3 = [
            [0.33, 0.33, 0.34], [0.5, 0.3, 0.2], [0.6, 0.25, 0.15], [0.2, 0.3, 0.5],
            [0.4, 0.4, 0.2], [0.7, 0.2, 0.1], [0.1, 0.2, 0.7], [0.45, 0.35, 0.2]
        ]
        
        triple_combos = [
            ('return_raw', 'risk_adj_vol', 'time_eff_linear'),
            ('return_raw', 'market_rel_simple', 'price_pos_adj'),
            ('sharpe_like', 'time_eff_linear', 'momentum_weighted'),
            ('risk_adj_vol', 'downside_penalty_medium', 'vix_opportunity'),
            ('return_raw', 'vol_scaled_return', 'ma_position'),
            ('market_rel_simple', 'momentum_weighted', 'time_eff_log'),
            ('comprehensive_risk', 'multi_factor_alpha', 'kelly_approx'),
            ('sortino_like', 'calmar_like', 'sterling_like')
        ]
        
        for comp_triple in triple_combos:
            for weights in weight_combos_3:
                name = f"combo3_{'_'.join([c[:6] for c in comp_triple])}_{'-'.join([f'{w:.2f}' for w in weights])}"
                labeling_methods[name] = {
                    'formula': lambda df, comps=comp_triple, w=weights: sum(components[c] * w[i] for i, c in enumerate(comps)),
                    'description': f'3성분: {" + ".join([f"{c}({w:.2f})" for c, w in zip(comp_triple, weights)])}',
                    'components': list(comp_triple),
                    'weights': weights
                }
        
        # 4. 4개 성분 조합들 (사용자 요청!)
        weight_combos_4 = [
            [0.25, 0.25, 0.25, 0.25],  # 균등
            [0.4, 0.3, 0.2, 0.1],      # 피라미드
            [0.1, 0.2, 0.3, 0.4],      # 역피라미드  
            [0.35, 0.35, 0.15, 0.15],  # 2+2
            [0.6, 0.2, 0.1, 0.1],      # 극단집중
            [0.5, 0.3, 0.15, 0.05],    # 체감
            [0.2, 0.2, 0.3, 0.3],      # 후반집중
            [0.3, 0.25, 0.25, 0.2],    # 약간불균등
        ]
        
        quad_combos = [
            ('return_raw', 'risk_adj_vol', 'time_eff_linear', 'market_rel_simple'),
            ('sharpe_like', 'sortino_like', 'calmar_like', 'sterling_like'),
            ('return_raw', 'downside_penalty_medium', 'upside_boost_medium', 'time_eff_linear'),
            ('risk_adj_vol', 'momentum_weighted', 'vix_opportunity', 'price_pos_adj'),
            ('comprehensive_risk', 'multi_factor_alpha', 'kelly_approx', 'info_ratio_simple'),
            ('vol_scaled_return', 'time_decay', 'market_correlation', 'ma_position'),
            ('return_log', 'risk_adj_vol2', 'time_eff_log', 'momentum_contrarian'),
            ('asymmetric_utility', 'vol_regime', 'macro_score', 'anchoring_bias')
        ]
        
        for comp_quad in quad_combos:
            for weights in weight_combos_4:
                name = f"combo4_{'_'.join([c[:5] for c in comp_quad])}_{'-'.join([f'{w:.2f}' for w in weights])}"
                labeling_methods[name] = {
                    'formula': lambda df, comps=comp_quad, w=weights: sum(components[c] * w[i] for i, c in enumerate(comps)),
                    'description': f'4성분: {" + ".join([f"{c}({w:.2f})" for c, w in zip(comp_quad, weights)])}',
                    'components': list(comp_quad),
                    'weights': weights
                }
        
        # 5. 5개+ 성분 조합들
        weight_combos_5 = [
            [0.2, 0.2, 0.2, 0.2, 0.2],    # 균등
            [0.3, 0.25, 0.2, 0.15, 0.1],   # 체감
            [0.4, 0.25, 0.15, 0.1, 0.1],   # 집중
            [0.15, 0.15, 0.25, 0.25, 0.2], # 중후반집중
        ]
        
        penta_combos = [
            ('return_raw', 'risk_adj_vol', 'time_eff_linear', 'market_rel_simple', 'momentum_weighted'),
            ('sharpe_like', 'sortino_like', 'calmar_like', 'sterling_like', 'martin_like'),
            ('comprehensive_risk', 'multi_factor_alpha', 'kelly_approx', 'info_ratio_simple', 'tracking_error_adj'),
            ('downside_penalty_medium', 'upside_boost_medium', 'vol_regime', 'vix_contrarian', 'yield_adjusted'),
        ]
        
        for comp_penta in penta_combos:
            for weights in weight_combos_5:
                name = f"combo5_{'_'.join([c[:4] for c in comp_penta])}_{'-'.join([f'{w:.2f}' for w in weights])}"
                labeling_methods[name] = {
                    'formula': lambda df, comps=comp_penta, w=weights: sum(components[c] * w[i] for i, c in enumerate(comps)),
                    'description': f'5성분: {" + ".join([f"{c}({w:.2f})" for c, w in zip(comp_penta, weights)])}',
                    'components': list(comp_penta),
                    'weights': weights
                }
        
        # 6. 초복잡 수식들 (창의적 조합)
        creative_formulas = {
            'ultimate_risk_adj': lambda df: (df['return_pct'] / np.maximum(df['entry_volatility_20d'], 0.01)) * (1 + df['entry_vix']/50) * np.exp(-df['holding_period_days']/30),
            'momentum_risk_time': lambda df: (df['return_pct'] * (1 + df['entry_momentum_20d']/100)) / (np.sqrt(df['entry_volatility_20d']) * np.log(df['holding_period_days'] + 1)),
            'market_adaptive': lambda df: df['return_pct'] * np.where(df['entry_vix'] > 25, 1 + (df['entry_vix']-25)/100, 1 - (25-df['entry_vix'])/200),
            'behavioral_composite': lambda df: df['return_pct'] * (1 - df['entry_ratio_52w_high']/300) * np.where(df['holding_period_days'] > 10, 1.1, 0.9),
            'volatility_regime_adaptive': lambda df: df['return_pct'] / np.where(df['entry_volatility_20d'] > 30, df['entry_volatility_20d'], np.sqrt(df['entry_volatility_20d'])),
            'macro_momentum_fusion': lambda df: (df['return_pct'] - df['market_return_during_holding']) * (1 + df['change_vix']/20) * (1 + df['entry_momentum_20d']/50),
            'asymmetric_risk_reward': lambda df: np.where(df['return_pct'] > 0, df['return_pct'] / np.sqrt(df['entry_volatility_20d']), df['return_pct'] / df['entry_volatility_20d']),
            'time_weighted_alpha': lambda df: (df['return_pct'] - df['market_return_during_holding']) / np.sqrt(df['holding_period_days']) / np.maximum(df['entry_volatility_20d'], 1),
            'contrarian_momentum_blend': lambda df: df['return_pct'] * (0.7 * (1 - df['entry_momentum_20d']/100) + 0.3 * (1 + df['entry_momentum_20d']/100)),
            'kelly_sortino_hybrid': lambda df: (df['return_pct'] / np.maximum(np.where(df['return_pct'] < 0, np.abs(df['return_pct']), 1), 0.01)) * (df['return_pct'] / np.maximum(df['entry_volatility_20d']**2, 0.01)),
        }
        
        for name, formula in creative_formulas.items():
            labeling_methods[f"creative_{name}"] = {
                'formula': formula,
                'description': f'창의적 공식: {name}',
                'components': ['complex_formula'],
                'weights': [1.0]
            }
        
        print(f"🎯 생성된 라벨링 방식: {len(labeling_methods):,}개")
        return labeling_methods
    
    def evaluate_labeling_method(self, y_labels, method_name, n_folds=3):
        """라벨링 방식 평가"""
        
        # 진입 시점 피처들만 사용
        feature_cols = [
            'entry_momentum_5d', 'entry_momentum_20d', 'entry_momentum_60d',
            'entry_ma_dev_5d', 'entry_ma_dev_20d', 'entry_ma_dev_60d',
            'entry_volatility_5d', 'entry_volatility_60d', 
            'entry_vix', 'entry_tnx_yield', 'position_size_pct',
            'entry_vol_change_5d', 'entry_vol_change_20d'
        ]
        
        available_features = [col for col in feature_cols if col in self.df.columns]
        
        if len(available_features) < 5:
            return {'error': 'Insufficient features'}
        
        X = self.df[available_features].fillna(0)
        y = y_labels
        
        # NaN/무한값 제거
        mask = ~(np.isnan(y) | np.isinf(y) | (np.abs(y) > 1e6))
        X = X[mask]
        y = y[mask]
        
        if len(X) < 200:
            return {'error': 'Insufficient data'}
        
        try:
            # 빠른 모델로 평가 (속도 최적화)
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42,
                n_jobs=1  # 병렬처리로 메모리 절약
            )
            
            scores = cross_val_score(model, X, y, cv=n_folds, scoring='r2')
            
            return {
                'mean_r2': scores.mean(),
                'std_r2': scores.std(),
                'min_r2': scores.min(),
                'max_r2': scores.max(),
                'method_name': method_name,
                'n_samples': len(X),
                'n_features': len(available_features),
                'y_mean': float(np.mean(y)),
                'y_std': float(np.std(y)),
                'y_range': float(np.max(y) - np.min(y))
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def run_mega_optimization(self, sample_size=15000, max_methods=None):
        """메가 최적화 실행"""
        
        print("🚀 메가 라벨링 최적화 시작")
        print("=" * 60)
        
        # 데이터 로드
        self.load_data(sample_size)
        
        # 모든 라벨링 방식 생성
        print("\n🔧 라벨링 방식 생성 중...")
        labeling_methods = self.create_mega_labeling_variants()
        
        if max_methods:
            # 무작위로 일부만 선택 (테스트용)
            import random
            method_names = random.sample(list(labeling_methods.keys()), min(max_methods, len(labeling_methods)))
            labeling_methods = {k: labeling_methods[k] for k in method_names}
        
        print(f"테스트할 방식: {len(labeling_methods):,}개")
        
        # 배치별로 처리 (메모리 관리)
        batch_size = 100
        all_results = []
        method_names = list(labeling_methods.keys())
        
        for i in range(0, len(method_names), batch_size):
            batch_names = method_names[i:i+batch_size]
            batch_results = []
            
            print(f"\n📊 배치 {i//batch_size + 1}/{(len(method_names)-1)//batch_size + 1} 처리 중... ({len(batch_names)}개 방식)")
            
            for j, method_name in enumerate(batch_names):
                if j % 20 == 0:
                    print(f"  진행: {j+1}/{len(batch_names)}")
                
                try:
                    method_info = labeling_methods[method_name]
                    y_labels = method_info['formula'](self.df)
                    
                    result = self.evaluate_labeling_method(y_labels, method_name)
                    
                    if 'error' not in result:
                        result.update({
                            'description': method_info['description'],
                            'components': method_info.get('components', []),
                            'weights': method_info.get('weights', [])
                        })
                        batch_results.append(result)
                    
                except Exception as e:
                    continue
            
            all_results.extend(batch_results)
            print(f"  배치 완료: {len(batch_results)}개 성공")
        
        # 결과 정렬
        self.test_results = sorted(all_results, key=lambda x: x.get('mean_r2', -999), reverse=True)
        
        # 상위 결과 출력
        print(f"\n🏆 최고 성능 라벨링 방식들 (상위 20개)")
        print("=" * 100)
        
        for i, result in enumerate(self.test_results[:20]):
            components_str = ' + '.join([f"{c}({w:.2f})" for c, w in zip(result.get('components', []), result.get('weights', []))])
            if len(components_str) > 60:
                components_str = components_str[:57] + "..."
                
            print(f"{i+1:2d}. R²={result['mean_r2']:.4f}±{result['std_r2']:.4f} | {result['method_name'][:25]:25s} | {components_str}")
            
            if i < 5:  # 상위 5개는 상세 정보
                print(f"    📊 샘플: {result['n_samples']:,}, 피처: {result['n_features']}, Y범위: {result['y_range']:.2f}")
        
        print(f"\n✅ 총 테스트 완료: {len(self.test_results):,}개 방식")
        return self.test_results
    
    def save_mega_results(self, filename=None, top_n=100):
        """메가 결과 저장"""
        if not filename:
            filename = f"mega_labeling_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 상위 N개만 저장 (파일 크기 관리)
        top_results = self.test_results[:top_n]
        
        serializable_results = []
        for result in top_results:
            clean_result = {}
            for key, value in result.items():
                if isinstance(value, (list, str, int, float, bool)):
                    clean_result[key] = value
                elif isinstance(value, np.ndarray):
                    clean_result[key] = value.tolist()
                else:
                    clean_result[key] = str(value)
            serializable_results.append(clean_result)
        
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_tested': len(self.test_results),
                'saved_top_n': len(serializable_results),
                'best_r2': self.test_results[0]['mean_r2'] if self.test_results else 0,
                'results': serializable_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"💾 메가 결과 저장: {filename} (상위 {top_n}개)")
        return filename

def main():
    """메인 실행"""
    print("🚀 메가 라벨링 최적화 시스템")
    print("=" * 50)
    
    optimizer = MegaLabelingOptimizer()
    
    # 메가 최적화 실행
    results = optimizer.run_mega_optimization(
        sample_size=20000,  # 적당한 샘플 크기
        max_methods=1000    # 테스트할 최대 방식 수 (None = 전체)
    )
    
    # 결과 저장
    optimizer.save_mega_results(top_n=100)
    
    print("\n✅ 메가 라벨링 최적화 완료!")

if __name__ == "__main__":
    main()