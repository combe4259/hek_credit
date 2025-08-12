"""
Trading Feedback Service - 완전 데이터 드리븐 거래 피드백 시스템

하드코딩된 임계값 없이 실제 모델 예측 분포와 과거 성과 데이터를 기반으로 
동적이고 적응적인 거래 피드백을 제공합니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 학습된 모델들 import
from buy_signal_predictor import BuySignalPredictor
from sell_signal_predictor import SellSignalPredictor
from trade_quality_evaluator import TradeQualityEvaluator

class DataDrivenFeedbackService:
    """완전 데이터 기반 거래 피드백 서비스"""
    
    def __init__(self, historical_data_path=None):
        # 학습된 모델들
        self.buy_predictor = BuySignalPredictor()
        self.sell_predictor = SellSignalPredictor()
        self.quality_evaluator = TradeQualityEvaluator()
        
        self.models_loaded = False
        self.historical_data = None
        
        # 동적 평가 기준 (모델과 데이터 기반으로 생성)
        self.evaluation_benchmarks = {}
        
        if historical_data_path:
            self.load_historical_data(historical_data_path)
    
    def load_models_and_calibrate(self, model_paths, verbose=True):
        """모델 로드 및 동적 평가 기준 설정"""
        
        # 1. 모델 로드
        if not self._load_models(model_paths, verbose):
            return False
        
        # 2. 과거 데이터가 있다면 동적 평가 기준 생성
        if self.historical_data is not None:
            if verbose:
                print("과거 데이터 기반 동적 평가 기준 생성 중...")
            self._create_dynamic_benchmarks(verbose)
        
        if verbose:
            print("데이터 기반 피드백 시스템 준비 완료")
        
        return True
    
    def _load_models(self, model_paths, verbose):
        """모델 로드"""
        try:
            if 'buy' in model_paths:
                self.buy_predictor.load_model(model_paths['buy'])
                if verbose: print("Buy Signal 모델 로드")
            
            if 'sell' in model_paths:
                self.sell_predictor.load_model(model_paths['sell'])
                if verbose: print("Sell Signal 모델 로드")
            
            if 'quality' in model_paths:
                self.quality_evaluator.load_model(model_paths['quality'])
                if verbose: print("Trade Quality 모델 로드")
            
            self.models_loaded = True
            return True
            
        except Exception as e:
            print(f"모델 로드 실패: {str(e)}")
            return False
    
    def load_historical_data(self, data_path):
        """과거 거래 데이터 로드"""
        try:
            self.historical_data = pd.read_csv(data_path)
            # 펀더멘털 데이터 필터링
            self.historical_data = self.historical_data[
                self.historical_data['entry_pe_ratio'].notna() & 
                self.historical_data['entry_roe'].notna() & 
                self.historical_data['entry_earnings_growth'].notna() &
                self.historical_data['return_pct'].notna()
            ].copy()
            print(f"과거 거래 데이터: {len(self.historical_data):,}건 (필터링 후)")
        except Exception as e:
            print(f"데이터 로드 실패: {str(e)}")
    
    def _create_dynamic_benchmarks(self, verbose=True):
        """과거 데이터 기반 동적 평가 기준 생성"""
        if self.historical_data is None or not self.models_loaded:
            return
        
        benchmarks = {}
        
        try:
            # 1. 모델 예측 분포 계산
            if verbose:
                print("   과거 데이터로 모델 예측 분포 계산 중...")
            
            model_predictions = self._get_historical_predictions()
            
            # 2. 실제 성과 분포 계산
            actual_performance = self._analyze_actual_performance()
            
            # 3. 예측-성과 매핑 생성
            score_performance_mapping = self._create_score_performance_mapping(
                model_predictions, actual_performance
            )
            
            # 4. 동적 평가 기준 설정
            benchmarks.update({
                'model_distributions': model_predictions,
                'performance_distributions': actual_performance,
                'score_performance_mapping': score_performance_mapping,
                'percentile_thresholds': self._calculate_percentile_thresholds(actual_performance)
            })
            
            self.evaluation_benchmarks = benchmarks
            
            if verbose:
                print("   동적 평가 기준 생성 완료")
                self._print_benchmark_summary()
                
        except Exception as e:
            print(f"   동적 기준 생성 실패: {str(e)}")
    
    def _get_historical_predictions(self):
        """과거 데이터에 대한 모델 예측값 계산"""
        predictions = {}
        
        try:
            # Buy Signal 예측
            if self.buy_predictor.is_trained:
                buy_scores = []
                for idx in range(0, len(self.historical_data), 1000):  # 배치 처리
                    batch = self.historical_data.iloc[idx:idx+1000]
                    batch_scores = self.buy_predictor.predict_entry_signal(batch, verbose=False)
                    buy_scores.extend(batch_scores)
                predictions['buy_signals'] = np.array(buy_scores)
            
            # Sell Signal 예측
            if self.sell_predictor.is_trained:
                sell_scores = []
                for idx in range(0, len(self.historical_data), 1000):
                    batch = self.historical_data.iloc[idx:idx+1000]
                    batch_scores = self.sell_predictor.predict_exit_signal(batch, verbose=False)
                    sell_scores.extend(batch_scores)
                predictions['sell_signals'] = np.array(sell_scores)
            
            # Quality 예측
            if self.quality_evaluator.is_trained:
                quality_scores = []
                for idx in range(0, len(self.historical_data), 1000):
                    batch = self.historical_data.iloc[idx:idx+1000]
                    batch_scores = self.quality_evaluator.predict_quality(batch, verbose=False)
                    quality_scores.extend(batch_scores)
                predictions['quality_scores'] = np.array(quality_scores)
            
        except Exception as e:
            print(f"예측 계산 중 오류: {str(e)}")
        
        return predictions
    
    def _analyze_actual_performance(self):
        """실제 성과 분포 분석"""
        performance = {}
        
        # 수익률 분포
        returns = self.historical_data['return_pct'].dropna()
        performance['returns'] = {
            'data': returns.values,
            'percentiles': np.percentile(returns, [5, 10, 25, 50, 75, 90, 95]),
            'mean': returns.mean(),
            'std': returns.std()
        }
        
        # 보유기간 분포
        holding_days = self.historical_data['holding_period_days'].dropna()
        performance['holding_periods'] = {
            'data': holding_days.values,
            'percentiles': np.percentile(holding_days, [5, 10, 25, 50, 75, 90, 95]),
            'mean': holding_days.mean()
        }
        
        # 업종별 성과
        if 'industry' in self.historical_data.columns:
            industry_performance = {}
            for industry in self.historical_data['industry'].unique():
                if pd.notna(industry):
                    industry_returns = self.historical_data[
                        self.historical_data['industry'] == industry
                    ]['return_pct'].dropna()
                    if len(industry_returns) > 10:
                        industry_performance[industry] = {
                            'mean_return': industry_returns.mean(),
                            'count': len(industry_returns),
                            'success_rate': (industry_returns > 0).mean()
                        }
            performance['by_industry'] = industry_performance
        
        return performance
    
    def _create_score_performance_mapping(self, predictions, performance):
        """모델 점수와 실제 성과 매핑"""
        mapping = {}
        
        returns = performance['returns']['data']
        
        # Buy Signal 점수별 성과
        if 'buy_signals' in predictions:
            buy_scores = predictions['buy_signals']
            if len(buy_scores) == len(returns):
                score_quintiles = np.percentile(buy_scores, [0, 20, 40, 60, 80, 100])
                buy_performance = {}
                
                for i in range(len(score_quintiles)-1):
                    mask = (buy_scores >= score_quintiles[i]) & (buy_scores < score_quintiles[i+1])
                    if mask.sum() > 0:
                        quintile_returns = returns[mask]
                        buy_performance[f'quintile_{i+1}'] = {
                            'score_range': [score_quintiles[i], score_quintiles[i+1]],
                            'avg_return': quintile_returns.mean(),
                            'success_rate': (quintile_returns > 0).mean(),
                            'count': len(quintile_returns)
                        }
                
                mapping['buy_signal_performance'] = buy_performance
        
        # Sell Signal과 Quality도 동일하게 처리
        for signal_type in ['sell_signals', 'quality_scores']:
            if signal_type in predictions:
                scores = predictions[signal_type]
                if len(scores) == len(returns):
                    mapping[f'{signal_type}_performance'] = self._calculate_quintile_performance(
                        scores, returns
                    )
        
        return mapping
    
    def _calculate_quintile_performance(self, scores, returns):
        """점수별 5분위 성과 계산"""
        quintiles = np.percentile(scores, [0, 20, 40, 60, 80, 100])
        performance = {}
        
        for i in range(len(quintiles)-1):
            mask = (scores >= quintiles[i]) & (scores < quintiles[i+1])
            if mask.sum() > 0:
                quintile_returns = returns[mask]
                performance[f'quintile_{i+1}'] = {
                    'score_range': [quintiles[i], quintiles[i+1]],
                    'avg_return': quintile_returns.mean(),
                    'success_rate': (quintile_returns > 0).mean(),
                    'count': len(quintile_returns)
                }
        
        return performance
    
    def _calculate_percentile_thresholds(self, performance):
        """백분위수 기반 임계값 계산"""
        returns = performance['returns']['data']
        
        return {
            'returns': {
                'excellent': np.percentile(returns, 90),    # 상위 10%
                'good': np.percentile(returns, 75),         # 상위 25%
                'average': np.percentile(returns, 50),      # 중앙값
                'poor': np.percentile(returns, 25),         # 하위 25%
                'bad': np.percentile(returns, 10)           # 하위 10%
            }
        }
    
    def analyze_trade(self, trade_data, verbose=True):
        """데이터 기반 거래 분석"""
        if not self.models_loaded:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        # 데이터 형태 통일
        if isinstance(trade_data, dict):
            trade_df = pd.DataFrame([trade_data])
        else:
            trade_df = pd.DataFrame([trade_data])
        
        try:
            # 1. 모델 예측
            current_predictions = self._get_current_predictions(trade_df)
            
            # 2. 데이터 기반 평가
            evaluation = self._evaluate_with_benchmarks(trade_data, current_predictions)
            
            # 3. 적응적 피드백 생성
            feedback = self._generate_adaptive_feedback(trade_data, current_predictions, evaluation)
            
            if verbose:
                self._print_adaptive_feedback(feedback)
            
            return feedback
            
        except Exception as e:
            return {'error': f"분석 중 오류: {str(e)}", 'success': False}
    
    def _get_current_predictions(self, trade_df):
        """현재 거래에 대한 모델 예측"""
        predictions = {}
        
        if self.buy_predictor.is_trained:
            predictions['buy_signal'] = self.buy_predictor.predict_entry_signal(trade_df, verbose=False)[0]
        
        if self.sell_predictor.is_trained:
            predictions['sell_signal'] = self.sell_predictor.predict_exit_signal(trade_df, verbose=False)[0]
        
        if self.quality_evaluator.is_trained:
            predictions['quality_score'] = self.quality_evaluator.predict_quality(trade_df, verbose=False)[0]
        
        return predictions
    
    def _evaluate_with_benchmarks(self, trade_data, predictions):
        """동적 기준으로 평가"""
        evaluation = {}
        
        if not self.evaluation_benchmarks:
            return {'message': '동적 평가 기준이 설정되지 않았습니다.'}
        
        benchmarks = self.evaluation_benchmarks
        
        # 1. 예측 점수의 상대적 위치
        if 'model_distributions' in benchmarks:
            score_rankings = {}
            for model_type, score in predictions.items():
                if model_type.replace('_signal', '_signals') in benchmarks['model_distributions']:
                    historical_scores = benchmarks['model_distributions'][model_type.replace('_signal', '_signals')]
                    percentile = (historical_scores < score).mean() * 100
                    score_rankings[model_type] = {
                        'score': score,
                        'percentile': percentile,
                        'rank_description': self._describe_percentile_rank(percentile)
                    }
            evaluation['score_rankings'] = score_rankings
        
        # 2. 실제 성과와 비교
        actual_return = trade_data.get('return_pct', 0)
        if 'performance_distributions' in benchmarks:
            perf_dist = benchmarks['performance_distributions']['returns']
            return_percentile = (perf_dist['data'] < actual_return).mean() * 100
            evaluation['performance_ranking'] = {
                'return': actual_return,
                'percentile': return_percentile,
                'vs_average': actual_return - perf_dist['mean'],
                'rank_description': self._describe_percentile_rank(return_percentile)
            }
        
        # 3. 예측-성과 일치도 분석
        if 'score_performance_mapping' in benchmarks:
            consistency_analysis = self._analyze_prediction_consistency(
                predictions, actual_return, benchmarks['score_performance_mapping']
            )
            evaluation['prediction_consistency'] = consistency_analysis
        
        return evaluation
    
    def _describe_percentile_rank(self, percentile):
        """백분위수를 설명 텍스트로 변환"""
        if percentile >= 90:
            return f"상위 {100-percentile:.0f}% (Excellent)"
        elif percentile >= 75:
            return f"상위 {100-percentile:.0f}% (Good)"
        elif percentile >= 50:
            return f"상위 {100-percentile:.0f}% (Above Average)"
        elif percentile >= 25:
            return f"상위 {100-percentile:.0f}% (Below Average)"
        else:
            return f"하위 {percentile:.0f}% (Poor)"
    
    def _analyze_prediction_consistency(self, predictions, actual_return, mapping):
        """예측과 실제 성과의 일치도 분석"""
        consistency = {}
        
        # Buy Signal 일치도
        if 'buy_signal' in predictions and 'buy_signal_performance' in mapping:
            buy_score = predictions['buy_signal']
            buy_mapping = mapping['buy_signal_performance']
            
            # 해당 점수 구간의 예상 성과 찾기
            expected_performance = None
            for quintile, data in buy_mapping.items():
                score_min, score_max = data['score_range']
                if score_min <= buy_score < score_max:
                    expected_performance = data
                    break
            
            if expected_performance:
                expected_return = expected_performance['avg_return']
                consistency['buy_signal'] = {
                    'predicted_score': buy_score,
                    'expected_return': expected_return,
                    'actual_return': actual_return,
                    'prediction_accuracy': 'Good' if abs(actual_return - expected_return) < 5 else 'Poor',
                    'expectation_vs_actual': actual_return - expected_return
                }
        
        return consistency
    
    def _generate_adaptive_feedback(self, trade_data, predictions, evaluation):
        """적응적 피드백 생성"""
        feedback = {
            'trade_info': self._extract_trade_info(trade_data),
            'model_predictions': predictions,
            'data_driven_evaluation': evaluation,
            'adaptive_insights': self._generate_adaptive_insights(evaluation),
            'learning_opportunities': self._identify_learning_opportunities(evaluation),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return feedback
    
    def _extract_trade_info(self, trade_data):
        """거래 정보 추출"""
        return {
            'symbol': trade_data.get('symbol', 'N/A'),
            'return_pct': trade_data.get('return_pct', 0),
            'holding_days': trade_data.get('holding_period_days', 0),
            'entry_date': trade_data.get('entry_date', 'N/A'),
            'exit_date': trade_data.get('exit_date', 'N/A')
        }
    
    def _generate_adaptive_insights(self, evaluation):
        """적응적 인사이트 생성"""
        insights = []
        
        # 점수 순위 기반 인사이트
        if 'score_rankings' in evaluation:
            for model_type, ranking in evaluation['score_rankings'].items():
                percentile = ranking['percentile']
                if percentile >= 75:
                    insights.append({
                        'type': 'strength',
                        'message': f"{model_type.replace('_', ' ').title()}이 {ranking['rank_description']}로 우수했습니다.",
                        'data_basis': f"과거 동일 모델 예측 대비 상위 {100-percentile:.0f}%"
                    })
                elif percentile <= 25:
                    insights.append({
                        'type': 'weakness',
                        'message': f"{model_type.replace('_', ' ').title()}이 {ranking['rank_description']}로 아쉬웠습니다.",
                        'data_basis': f"과거 동일 모델 예측 대비 하위 {percentile:.0f}%"
                    })
        
        # 성과 순위 기반 인사이트
        if 'performance_ranking' in evaluation:
            perf = evaluation['performance_ranking']
            insights.append({
                'type': 'performance',
                'message': f"실제 수익률이 {perf['rank_description']}을 기록했습니다.",
                'data_basis': f"과거 전체 거래 대비 {perf['vs_average']:+.2f}%p 차이"
            })
        
        return insights
    
    def _identify_learning_opportunities(self, evaluation):
        """학습 기회 식별"""
        opportunities = []
        
        # 예측-실제 불일치 분석
        if 'prediction_consistency' in evaluation:
            consistency = evaluation['prediction_consistency']
            for model_type, analysis in consistency.items():
                if analysis['prediction_accuracy'] == 'Poor':
                    diff = analysis['expectation_vs_actual']
                    if diff > 0:
                        opportunities.append({
                            'area': f"{model_type.replace('_', ' ').title()} 예측",
                            'opportunity': f"모델이 예상({analysis['expected_return']:.1f}%)보다 {diff:.1f}%p 더 좋은 성과",
                            'learning': "이런 패턴의 거래를 더 찾아볼 가치가 있습니다.",
                            'confidence': 'data_based'
                        })
                    else:
                        opportunities.append({
                            'area': f"{model_type.replace('_', ' ').title()} 예측",
                            'opportunity': f"모델이 예상({analysis['expected_return']:.1f}%)보다 {abs(diff):.1f}%p 못한 성과",
                            'learning': "이런 패턴에서는 더 신중한 접근이 필요합니다.",
                            'confidence': 'data_based'
                        })
        
        return opportunities
    
    def _print_benchmark_summary(self):
        """벤치마크 요약 출력"""
        if 'performance_distributions' in self.evaluation_benchmarks:
            perf = self.evaluation_benchmarks['performance_distributions']['returns']
            print(f"   수익률 분포: {perf['mean']:.2f}% ± {perf['std']:.2f}%")
            print(f"   백분위: P10={perf['percentiles'][1]:.1f}% | P50={perf['percentiles'][3]:.1f}% | P90={perf['percentiles'][5]:.1f}%")
    
    def _print_adaptive_feedback(self, feedback):
        """적응적 피드백 출력"""
        print("=" * 80)
        print("AI 데이터 기반 거래 분석")
        print("=" * 80)
        
        # 거래 정보
        info = feedback['trade_info']
        print(f"거래: {info['symbol']} | {info['entry_date']} ~ {info['exit_date']}")
        print(f"수익률: {info['return_pct']:+.2f}% | 보유기간: {info['holding_days']}일")
        
        # 모델 예측 결과
        if 'model_predictions' in feedback:
            print(f"\nAI 모델 예측:")
            preds = feedback['model_predictions']
            for model_type, score in preds.items():
                print(f"  {model_type.replace('_', ' ').title()}: {score:.2f}")
        
        # 데이터 기반 평가
        if 'data_driven_evaluation' in feedback:
            eval_data = feedback['data_driven_evaluation']
            
            if 'score_rankings' in eval_data:
                print(f"\n모델 점수 순위 (과거 예측 대비):")
                for model_type, ranking in eval_data['score_rankings'].items():
                    print(f"  {model_type.replace('_', ' ').title()}: {ranking['rank_description']}")
            
            if 'performance_ranking' in eval_data:
                perf = eval_data['performance_ranking']
                print(f"\n실제 성과 순위 (과거 거래 대비):")
                print(f"  {perf['rank_description']} | 평균 대비 {perf['vs_average']:+.2f}%p")
        
        # 적응적 인사이트
        if 'adaptive_insights' in feedback:
            print(f"\n데이터 기반 인사이트:")
            for i, insight in enumerate(feedback['adaptive_insights'][:3], 1):
                print(f"  {i}. {insight['message']}")
                print(f"     근거: {insight['data_basis']}")
        
        # 학습 기회
        if 'learning_opportunities' in feedback:
            print(f"\n학습 기회:")
            for i, opp in enumerate(feedback['learning_opportunities'][:2], 1):
                print(f"  {i}. [{opp['area']}] {opp['learning']}")
        
        print("=" * 80)

def main():
    """테스트용 메인 함수"""
    print("Data-Driven Trading Feedback Service")
    print("=" * 60)
    
    # 데이터 경로
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '../results/final/trading_episodes_with_rebuilt_market_component.csv')
    
    # 서비스 초기화
    feedback_service = DataDrivenFeedbackService(historical_data_path=data_path)
    
    # 모델 파일 찾기
    import glob
    model_files = {
        'buy': sorted(glob.glob('buy_signal_predictor_*.pkl'), reverse=True),
        'sell': sorted(glob.glob('sell_signal_predictor_*.pkl'), reverse=True),
        'quality': sorted(glob.glob('trade_quality_evaluator_*.pkl'), reverse=True)
    }
    
    model_paths = {}
    print("모델 파일 검색:")
    for model_type, files in model_files.items():
        if files:
            model_paths[model_type] = files[0]
            print(f"  {model_type}: {files[0]}")
        else:
            print(f"  {model_type}: 파일 없음")
    
    if len(model_paths) > 0:
        # 모델 로드 및 동적 기준 설정
        success = feedback_service.load_models_and_calibrate(model_paths, verbose=True)
        
        if success:
            print(f"\n샘플 거래 분석:")
            
            # 실제 CSV에서 샘플 거래 데이터 가져오기
            import pandas as pd
            df_sample = pd.read_csv(data_path)
            has_fundamental = df_sample['entry_pe_ratio'].notna() | df_sample['entry_roe'].notna() | df_sample['entry_earnings_growth'].notna()
            sample_row = df_sample[has_fundamental].iloc[0]
            
            # 실제 샘플 거래 데이터
            sample_trade = sample_row.to_dict()
            
            # 분석 실행
            result = feedback_service.analyze_trade(sample_trade, verbose=True)
            
            if 'error' not in result:
                print(f"\n데이터 기반 분석 완료")
            else:
                print(f"\n분석 실패: {result['error']}")
    else:
        print(f"\n모델 파일이 없습니다. 먼저 모델을 학습시키세요.")

if __name__ == "__main__":
    main()