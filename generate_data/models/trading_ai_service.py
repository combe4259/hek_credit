"""
Trading AI Service - 실서비스용 통합 API

A, B, C 타입 모델을 통합하여 실시간 트레이딩 서비스를 지원하는 API
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 각 타입별 모델 import
from trade_quality_evaluator import TradeQualityEvaluator
from buy_signal_predictor import BuySignalPredictor  
from sell_signal_predictor import SellSignalPredictor

class TradingAIService:
    """
    실서비스용 트레이딩 AI 통합 서비스
    
    서비스 흐름:
    1. buy-type: 실시간 매수 신호 판단 → 매수 실행
    2. sell-type: 보유 중인 종목의 매도 신호 판단 → 매도 실행
    3. trade-type: 매수-매도 완료 후 거래 품질 평가 → 학습 데이터
    """
    
    def __init__(self):
        # 각 타입별 모델 초기화
        self.trade_evaluator = TradeQualityEvaluator()
        self.buy_predictor = BuySignalPredictor()
        self.sell_predictor = SellSignalPredictor()
        
        # 모델 로드 상태 추적
        self.models_loaded = {
            'A': False,
            'B': False, 
            'C': False
        }
        
        # 서비스 통계
        self.service_stats = {
            'total_buy_signals': 0,
            'total_sell_signals': 0,
            'total_quality_evaluations': 0,
            'service_start_time': datetime.now()
        }
    
    # ================================
    # 모델 관리 (로드/저장)
    # ================================
    
    def load_models(self, model_paths: Dict[str, str], verbose: bool = False) -> Dict[str, bool]:
        """
        각 타입별 모델 로드
        
        Args:
            model_paths: {'A': 'trade_quality_evaluator.pkl', 'B': 'buy_signal_predictor.pkl', 'C': 'sell_signal_predictor.pkl'}
            verbose: 로그 출력 여부
            
        Returns:
            로드 성공 여부: {'A': True, 'B': True, 'C': False}
        """
        results = {}
        
        try:
            if 'A' in model_paths and model_paths['A']:
                self.trade_evaluator.load_model(model_paths['A'])
                self.models_loaded['A'] = True
                results['A'] = True
                if verbose:
                    print("✅ A-type 거래품질평가 모델 로드 성공")
            
            if 'B' in model_paths and model_paths['B']:
                self.buy_predictor.load_model(model_paths['B'])
                self.models_loaded['B'] = True
                results['B'] = True
                if verbose:
                    print("✅ B-type 매수신호 모델 로드 성공")
            
            if 'C' in model_paths and model_paths['C']:
                self.sell_predictor.load_model(model_paths['C'])
                self.models_loaded['C'] = True
                results['C'] = True
                if verbose:
                    print("✅ C-type 매도신호 모델 로드 성공")
                    
        except Exception as e:
            if verbose:
                print(f"❌ 모델 로드 중 오류: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def get_model_status(self) -> Dict:
        """모델 상태 확인"""
        return {
            'models_loaded': self.models_loaded,
            'service_stats': self.service_stats,
            'ready_for_service': all(self.models_loaded.values())
        }
    
    # ================================
    # B-type: 매수 신호 API
    # ================================
    
    def get_buy_signals(self, candidate_data: pd.DataFrame, threshold: float = 60.0, 
                       verbose: bool = False) -> Dict:
        """
        매수 신호 평가 (실시간 서비스)
        
        Args:
            candidate_data: 매수 후보 종목들의 데이터
            threshold: 매수 신호 최소 임계값 (0-100)
            verbose: 로그 출력 여부
            
        Returns:
            {
                'recommendations': [{'symbol': 'AAPL', 'signal_strength': 85.2, 'recommendation': '강한 매수 신호'}, ...],
                'summary': {'total_candidates': 100, 'buy_recommendations': 12, 'avg_signal': 58.3},
                'timestamp': '2024-01-15 09:30:00'
            }
        """
        if not self.models_loaded['B']:
            raise ValueError("B-type 모델이 로드되지 않았습니다.")
        
        if verbose:
            print(f"🚀 매수 신호 평가: {len(candidate_data)}개 종목")
        
        try:
            # 매수 신호 점수 예측
            signal_scores = self.buy_predictor.predict_entry_signal(candidate_data, verbose=verbose)
            
            # 결과 정리
            recommendations = []
            for i, score in enumerate(signal_scores):
                if i < len(candidate_data):
                    symbol = candidate_data.iloc[i].get('symbol', f'STOCK_{i+1}')
                    
                    if score >= threshold:
                        recommendations.append({
                            'symbol': symbol,
                            'signal_strength': float(score),
                            'recommendation': self.buy_predictor.get_signal_interpretation(score),
                            'rank': len([s for s in signal_scores if s > score]) + 1
                        })
            
            # 신호 강도 순으로 정렬
            recommendations.sort(key=lambda x: x['signal_strength'], reverse=True)
            
            # 통계 업데이트
            self.service_stats['total_buy_signals'] += len(candidate_data)
            
            summary = {
                'total_candidates': len(candidate_data),
                'buy_recommendations': len(recommendations),
                'avg_signal': float(np.mean(signal_scores)),
                'max_signal': float(np.max(signal_scores)),
                'threshold_used': threshold
            }
            
            return {
                'recommendations': recommendations,
                'summary': summary,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'service_type': 'B_TYPE_BUY_SIGNAL'
            }
            
        except Exception as e:
            return {
                'error': f"매수 신호 평가 중 오류: {str(e)}",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    # ================================
    # C-type: 매도 신호 API  
    # ================================
    
    def get_sell_signals(self, portfolio_data: pd.DataFrame, threshold: float = 0.0,
                        verbose: bool = False) -> Dict:
        """
        매도 신호 평가 (실시간 서비스)
        
        Args:
            portfolio_data: 현재 보유 중인 포지션 데이터
            threshold: 매도 신호 최소 임계값 (표준화된 값)
            verbose: 로그 출력 여부
            
        Returns:
            {
                'recommendations': [{'symbol': 'TSLA', 'signal_strength': 2.15, 'recommendation': '즉시 매도 권장'}, ...],
                'summary': {'total_positions': 15, 'sell_recommendations': 3, 'avg_signal': 0.12},
                'timestamp': '2024-01-15 15:45:00'  
            }
        """
        if not self.models_loaded['C']:
            raise ValueError("C-type 모델이 로드되지 않았습니다.")
        
        if verbose:
            print(f"🛑 매도 신호 평가: {len(portfolio_data)}개 포지션")
        
        try:
            # 매도 신호 점수 예측
            signal_scores = self.sell_predictor.predict_exit_signal(portfolio_data, verbose=verbose)
            
            # 결과 정리
            recommendations = []
            for i, score in enumerate(signal_scores):
                if i < len(portfolio_data):
                    symbol = portfolio_data.iloc[i].get('symbol', f'POS_{i+1}')
                    current_return = portfolio_data.iloc[i].get('return_pct', 0)
                    
                    if score >= threshold:
                        recommendations.append({
                            'symbol': symbol,
                            'signal_strength': float(score),
                            'recommendation': self.sell_predictor.get_signal_interpretation(score),
                            'current_return_pct': float(current_return),
                            'holding_days': int(portfolio_data.iloc[i].get('holding_period_days', 0)),
                            'urgency': 'HIGH' if score > 1.5 else 'MEDIUM' if score > 0.5 else 'LOW'
                        })
            
            # 신호 강도 순으로 정렬 (높은 순)
            recommendations.sort(key=lambda x: x['signal_strength'], reverse=True)
            
            # 통계 업데이트
            self.service_stats['total_sell_signals'] += len(portfolio_data)
            
            summary = {
                'total_positions': len(portfolio_data),
                'sell_recommendations': len(recommendations),
                'avg_signal': float(np.mean(signal_scores)),
                'max_signal': float(np.max(signal_scores)),
                'threshold_used': threshold,
                'high_urgency_count': len([r for r in recommendations if r['urgency'] == 'HIGH'])
            }
            
            return {
                'recommendations': recommendations,
                'summary': summary,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'service_type': 'C_TYPE_SELL_SIGNAL'
            }
            
        except Exception as e:
            return {
                'error': f"매도 신호 평가 중 오류: {str(e)}",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    # ================================
    # A-type: 거래 품질 평가 API
    # ================================
    
    def evaluate_trade_quality(self, completed_trades: pd.DataFrame, 
                             verbose: bool = False) -> Dict:
        """
        완료된 거래 품질 평가 (배치 서비스)
        
        Args:
            completed_trades: 완료된 거래 데이터
            verbose: 로그 출력 여부
            
        Returns:
            {
                'evaluations': [{'trade_id': 'T001', 'quality_score': 1.25, 'grade': 'Good'}, ...],
                'summary': {'total_trades': 50, 'avg_quality': 0.85, 'excellent_trades': 8},
                'timestamp': '2024-01-15 18:00:00'
            }
        """
        if not self.models_loaded['A']:
            raise ValueError("A-type 모델이 로드되지 않았습니다.")
        
        if verbose:
            print(f"🎯 거래 품질 평가: {len(completed_trades)}건의 거래")
        
        try:
            # 거래 품질 점수 예측
            quality_scores = self.trade_evaluator.predict_quality(completed_trades, verbose=verbose)
            
            # 결과 정리
            evaluations = []
            for i, score in enumerate(quality_scores):
                if i < len(completed_trades):
                    trade_id = completed_trades.iloc[i].get('trade_id', f'T{i+1:03d}')
                    return_pct = completed_trades.iloc[i].get('return_pct', 0)
                    
                    # 품질 등급 부여
                    if score > 2.0:
                        grade = 'Excellent'
                    elif score > 1.0:
                        grade = 'Good'  
                    elif score > 0:
                        grade = 'Average'
                    elif score > -1.0:
                        grade = 'Below Average'
                    else:
                        grade = 'Poor'
                    
                    evaluations.append({
                        'trade_id': trade_id,
                        'quality_score': float(score),
                        'grade': grade,
                        'return_pct': float(return_pct),
                        'holding_days': int(completed_trades.iloc[i].get('holding_period_days', 0))
                    })
            
            # 품질 점수 순으로 정렬
            evaluations.sort(key=lambda x: x['quality_score'], reverse=True)
            
            # 통계 업데이트
            self.service_stats['total_quality_evaluations'] += len(completed_trades)
            
            # 등급별 통계
            grade_counts = {}
            for eval_item in evaluations:
                grade = eval_item['grade']
                grade_counts[grade] = grade_counts.get(grade, 0) + 1
            
            summary = {
                'total_trades': len(completed_trades),
                'avg_quality': float(np.mean(quality_scores)),
                'max_quality': float(np.max(quality_scores)),
                'min_quality': float(np.min(quality_scores)),
                'grade_distribution': grade_counts,
                'profitable_trades': len([e for e in evaluations if e['return_pct'] > 0])
            }
            
            return {
                'evaluations': evaluations,
                'summary': summary,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'service_type': 'A_TYPE_QUALITY_EVALUATION'
            }
            
        except Exception as e:
            return {
                'error': f"거래 품질 평가 중 오류: {str(e)}",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    # ================================
    # 통합 대시보드 API
    # ================================
    
    def get_service_dashboard(self) -> Dict:
        """서비스 전체 현황 대시보드"""
        uptime = datetime.now() - self.service_stats['service_start_time']
        
        return {
            'service_status': {
                'models_loaded': self.models_loaded,
                'ready_for_service': all(self.models_loaded.values()),
                'uptime_hours': round(uptime.total_seconds() / 3600, 2)
            },
            'usage_stats': self.service_stats,
            'model_info': {
                'A_TYPE': 'Trade Quality Evaluator - 거래 품질 평가',
                'B_TYPE': 'Entry Signal Predictor - 매수 신호 예측', 
                'C_TYPE': 'Exit Signal Predictor - 매도 신호 예측'
            },
            'api_endpoints': {
                'buy_signals': '/api/buy-signals',
                'sell_signals': '/api/sell-signals',
                'quality_evaluation': '/api/quality-evaluation',
                'dashboard': '/api/dashboard'
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def create_sample_service() -> TradingAIService:
    """테스트용 샘플 서비스 생성"""
    service = TradingAIService()
    
    # 샘플 데이터 생성 함수들
    def create_sample_buy_candidates():
        return pd.DataFrame({
            'symbol': ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL'],
            'entry_momentum_20d': [-8.2, 12.5, -15.3, 2.1, -3.5],
            'entry_ma_dev_20d': [-12.1, 5.8, -8.9, 15.2, -2.3],
            'entry_ratio_52w_high': [45.2, 85.1, 25.8, 78.9, 62.3],
            'entry_volatility_20d': [22.5, 45.8, 18.9, 35.2, 28.1],
            'entry_pe_ratio': [12.5, 35.8, 8.9, 25.2, 18.7],
            'entry_roe': [15.2, 8.9, 22.1, 5.8, 18.3],
            'entry_earnings_growth': [8.5, -2.1, 15.8, 25.9, 3.2],
            'entry_vix': [18.5, 32.1, 15.2, 28.7, 22.3],
            'entry_tnx_yield': [2.8, 4.2, 1.9, 3.5, 2.5],
            'position_size_pct': [2.5, 1.8, 3.2, 1.5, 2.1]
        })
    
    def create_sample_portfolio():
        return pd.DataFrame({
            'symbol': ['AMZN', 'META', 'AMD'],
            'return_pct': [8.5, -4.2, 12.3],
            'holding_period_days': [25, 12, 45],
            'exit_volatility_20d': [22.5, 35.8, 18.9],
            'exit_momentum_20d': [-8.2, -15.5, 5.8],
            'change_volatility_5d': [5.2, 18.9, -8.5],
            'change_vix': [8.5, -3.2, 12.1],
            'position_size_pct': [2.5, 1.8, 3.2]
        })
    
    def create_sample_completed_trades():
        return pd.DataFrame({
            'trade_id': ['T001', 'T002', 'T003'],
            'return_pct': [5.2, -3.1, 8.7],
            'entry_volatility_20d': [18.5, 25.3, 15.2],
            'entry_ratio_52w_high': [65.2, 85.1, 45.3],
            'holding_period_days': [15, 8, 25],
            'position_size_pct': [2.5, 1.8, 3.2]
        })
    
    # 테스트 실행
    print("🎯 Trading AI Service 테스트")
    print("="*50)
    
    # 대시보드
    dashboard = service.get_service_dashboard()
    print(f"서비스 상태: {'준비됨' if dashboard['service_status']['ready_for_service'] else '모델 로드 필요'}")
    
    return service

def main():
    """테스트용 메인 함수"""
    import os
    import glob
    
    # 서비스 초기화
    service = TradingAIService()
    
    # 현재 디렉토리에서 가장 최근 모델 파일 찾기
    model_files = {
        'A': sorted(glob.glob('trade_quality_evaluator_*.pkl'), reverse=True),
        'B': sorted(glob.glob('buy_signal_predictor_*.pkl'), reverse=True),
        'C': sorted(glob.glob('sell_signal_predictor_*.pkl'), reverse=True)
    }
    
    # 최신 모델 파일 경로
    model_paths = {}
    print("\n🔍 모델 파일 검색 중...")
    for model_type, files in model_files.items():
        if files:
            model_paths[model_type] = files[0]
            print(f"  - {model_type}-type: {files[0]}")
        else:
            print(f"  - {model_type}-type: 파일 없음")
    
    # 모델 로드 시도
    if len(model_paths) == 3:
        print("\n📂 모델 로드 중...")
        load_results = service.load_models(model_paths, verbose=True)
        
        # 서비스 상태 확인
        status = service.get_model_status()
        if status['ready_for_service']:
            print("\n✅ 모든 모델 로드 완료! 서비스 준비 완료!")
            print("\n📊 서비스 상태:")
            print(f"  - A-type (거래품질평가): {'✅' if status['models_loaded']['A'] else '❌'}")
            print(f"  - B-type (매수신호): {'✅' if status['models_loaded']['B'] else '❌'}")
            print(f"  - C-type (매도신호): {'✅' if status['models_loaded']['C'] else '❌'}")
            
            # 간단한 테스트 데이터로 API 테스트
            print("\n🧪 API 테스트...")
            test_api_calls(service)
        else:
            print("\n⚠️ 일부 모델 로드 실패")
    else:
        print("\n❌ 모델 파일이 모두 준비되지 않았습니다.")
        print("다음 명령어로 모델을 학습시키세요:")
        if 'A' not in model_paths:
            print("  python trade_quality_evaluator.py")
        if 'B' not in model_paths:
            print("  python buy_signal_predictor.py")
        if 'C' not in model_paths:
            print("  python sell_signal_predictor.py")

def test_api_calls(service):
    """간단한 API 호출 테스트"""
    import pandas as pd
    import numpy as np
    
    # 테스트용 더미 데이터 생성
    print("\n1️⃣ Buy Signal API 테스트")
    buy_test_data = pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL', 'MSFT'],
        'rsi': [45, 30, 65],
        'macd_signal': [0.5, 1.2, -0.3],
        'entry_pe_ratio': [25, 30, 28],
        'entry_roe': [35, 28, 40],
        'vix': [15, 15, 15],
        'volume_ratio': [1.2, 0.8, 1.5]
    })
    
    try:
        buy_result = service.get_buy_signals(buy_test_data, threshold=50.0, verbose=True)
        if 'recommendations' in buy_result:
            print(f"  - 추천 종목 수: {len(buy_result['recommendations'])}")
            if buy_result['recommendations']:
                print(f"  - 최고 점수: {buy_result['recommendations'][0]['signal_strength']:.2f}")
    except Exception as e:
        print(f"  - 오류: {e}")
    
    print("\n2️⃣ Sell Signal API 테스트")
    sell_test_data = pd.DataFrame({
        'symbol': ['TSLA', 'NVDA'],
        'holding_period_days': [30, 45],
        'return_pct': [15.5, -5.2],
        'exit_rsi': [75, 25],
        'profit_taking_signal': [0.8, -0.2],
        'exit_vix': [18, 18]
    })
    
    try:
        sell_result = service.get_sell_signals(sell_test_data, threshold=0.0, verbose=True)
        if 'recommendations' in sell_result:
            print(f"  - 매도 추천 수: {len(sell_result['recommendations'])}")
    except Exception as e:
        print(f"  - 오류: {e}")
    
    print("\n3️⃣ Trade Quality API 테스트")
    trade_test_data = pd.DataFrame({
        'symbol': ['AMD', 'INTC'],
        'entry_signal_strength': [75, 60],
        'exit_signal_strength': [1.2, 0.8],
        'return_pct': [25.0, -3.5],
        'holding_period_days': [20, 15]
    })
    
    try:
        quality_result = service.evaluate_trade_quality(trade_test_data, verbose=True)
        if 'evaluations' in quality_result:
            print(f"  - 평가 완료: {len(quality_result['evaluations'])} 거래")
    except Exception as e:
        print(f"  - 오류: {e}")

if __name__ == "__main__":
    main()