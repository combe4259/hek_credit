import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TradingSimulator:
    """
    하이브리드 트레이딩 AI 시뮬레이터
    
    기능:
    1. B유형: 실시간 진입 점수 계산
    2. A유형: 거래 완료 후 품질 피드백
    3. 시나리오 테스트 및 백테스팅
    """
    
    def __init__(self, model_path=None, data_path=None):
        self.data_path = data_path or '../results/final/trading_episodes_with_rebuilt_market_component.csv'
        self.model_path = model_path
        
        # 모델 로드
        self.a_model = None
        self.b_model = None
        self.scalers = {}
        
        if model_path:
            self.load_models(model_path)
        
        # 데이터 로드
        print("📊 데이터 로드 중...")
        self.df = pd.read_csv(self.data_path)
        print(f"  총 {len(self.df):,}개 거래 에피소드 로드")
        
        # 시뮬레이션 결과
        self.simulation_history = []
    
    def load_models(self, model_path):
        """저장된 모델 로드"""
        try:
            if model_path.endswith('.pkl'):
                model_data = joblib.load(model_path)
                print(f"✅ 모델 로드 완료: {model_path}")
                return model_data
            else:
                print("❌ .pkl 파일만 지원됩니다")
                return None
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return None
    
    def get_random_scenario(self, symbol=None, date_range=None):
        """랜덤 시나리오 생성"""
        df_filtered = self.df.copy()
        
        if symbol:
            df_filtered = df_filtered[df_filtered['symbol'] == symbol]
        
        if date_range:
            start_date, end_date = date_range
            df_filtered = df_filtered[
                (pd.to_datetime(df_filtered['entry_datetime']) >= start_date) &
                (pd.to_datetime(df_filtered['entry_datetime']) <= end_date)
            ]
        
        if len(df_filtered) == 0:
            print("❌ 조건에 맞는 데이터가 없습니다")
            return None
        
        return df_filtered.sample(1).iloc[0]
    
    def calculate_b_score(self, data):
        """B유형: 진입 조건 점수 계산 (현재 정보만)"""
        try:
            # NaN 처리
            entry_vix = data.get('entry_vix', 0) or 0
            entry_volatility = data.get('entry_volatility_20d', 0) or 0  
            entry_ratio_52w = data.get('entry_ratio_52w_high', 0) or 0
            entry_momentum = data.get('entry_momentum_20d', 0) or 0
            
            # 1. 기술적 조건 점수 (40%)
            # 과매도 조건 (52주 고점 대비 낮을수록 좋음)
            rsi_proxy = max(0, min(1, (100 - entry_ratio_52w) / 100))
            
            # 모멘텀 점수 (적당한 하락 후 반등이 좋음)  
            momentum_safe = max(-50, min(50, entry_momentum))
            if momentum_safe < -10:
                momentum_score = 0.8  # 하락 후
            elif momentum_safe > 10:
                momentum_score = 0.3  # 상승 중
            else:
                momentum_score = 0.6  # 중립
            
            technical_score = rsi_proxy * 0.6 + momentum_score * 0.4
            
            # 2. 시장 환경 점수 (35%)
            # VIX가 낮을수록 좋은 진입 환경
            vix_safe = max(10, min(50, entry_vix))
            vix_score = (50 - vix_safe) / 40
            
            market_env_score = vix_score
            
            # 3. 리스크 관리 점수 (25%)
            # 변동성이 적당할수록 좋음
            vol_safe = max(10, min(100, entry_volatility))
            if vol_safe < 25:
                vol_score = 1.0  # 낮은 변동성
            elif vol_safe > 50:
                vol_score = 0.3  # 높은 변동성  
            else:
                vol_score = 0.7  # 적당한 변동성
            
            risk_score = vol_score
            
            # 종합 점수 계산
            final_score = (technical_score * 0.4 + 
                          market_env_score * 0.35 + 
                          risk_score * 0.25) * 100
            
            return {
                'total_score': round(final_score, 1),
                'technical_score': round(technical_score * 100, 1),
                'market_env_score': round(market_env_score * 100, 1),
                'risk_score': round(risk_score * 100, 1),
                'components': {
                    'rsi_proxy': round(rsi_proxy, 3),
                    'momentum_score': round(momentum_score, 3),
                    'vix_score': round(vix_score, 3),
                    'vol_score': round(vol_score, 3)
                }
            }
            
        except Exception as e:
            print(f"❌ B점수 계산 오류: {e}")
            return {'total_score': 0, 'error': str(e)}
    
    def calculate_a_score(self, data):
        """A유형: 거래 품질 점수 계산 (완료된 거래 모든 정보)"""
        try:
            # NaN 처리
            return_pct = data.get('return_pct', 0) or 0
            entry_volatility = data.get('entry_volatility_20d', 0) or 0
            entry_ratio_52w = data.get('entry_ratio_52w_high', 0) or 0  
            holding_days = data.get('holding_period_days', 0) or 0
            market_return = data.get('market_return_during_holding', 0) or 0
            
            # 1. Risk Management Quality (40%)
            # 리스크 조정 수익률
            volatility_safe = max(0.01, entry_volatility)
            risk_adj_return = return_pct / volatility_safe
            
            # 가격 안전도
            ratio_safe = max(0, min(100, entry_ratio_52w))
            price_safety = (100 - ratio_safe) / 100
            
            risk_management_score = risk_adj_return * 0.6 + price_safety * 0.4
            
            # 2. Efficiency Quality (60%)  
            # 시간 효율성
            holding_safe = max(1, holding_days)
            time_efficiency = return_pct / holding_safe
            
            # 시장 대비 효율성
            market_efficiency = return_pct - market_return
            
            efficiency_score = time_efficiency * 0.7 + market_efficiency * 0.3
            
            # 종합 품질 점수 (정규화)
            # 간단한 정규화: 평균적으로 0-100 범위로 맞춤
            normalized_risk = max(0, min(100, (risk_management_score + 2) * 25))
            normalized_eff = max(0, min(100, (efficiency_score + 1) * 50))
            
            final_score = normalized_risk * 0.4 + normalized_eff * 0.6
            
            return {
                'total_score': round(final_score, 1),
                'risk_management_score': round(normalized_risk, 1),
                'efficiency_score': round(normalized_eff, 1),
                'components': {
                    'risk_adj_return': round(risk_adj_return, 3),
                    'price_safety': round(price_safety, 3),
                    'time_efficiency': round(time_efficiency, 3),
                    'market_efficiency': round(market_efficiency, 3)
                },
                'raw_metrics': {
                    'return_pct': round(return_pct, 2),
                    'holding_days': holding_days,
                    'market_return': round(market_return, 2)
                }
            }
            
        except Exception as e:
            print(f"❌ A점수 계산 오류: {e}")
            return {'total_score': 0, 'error': str(e)}
    
    def format_recommendation(self, b_score):
        """B점수 기반 추천 메시지"""
        score = b_score['total_score']
        
        if score >= 80:
            return {
                'action': '💚 강력 추천',
                'confidence': '높음',
                'message': '매우 좋은 진입 타이밍입니다!',
                'risk_level': '낮음'
            }
        elif score >= 65:
            return {
                'action': '🟡 추천', 
                'confidence': '보통',
                'message': '양호한 진입 조건입니다.',
                'risk_level': '보통'
            }
        elif score >= 45:
            return {
                'action': '🟠 신중',
                'confidence': '낮음', 
                'message': '신중한 접근이 필요합니다.',
                'risk_level': '높음'
            }
        else:
            return {
                'action': '🔴 비추천',
                'confidence': '매우낮음',
                'message': '진입을 피하는 것이 좋습니다.',
                'risk_level': '매우높음'
            }
    
    def format_feedback(self, a_score):
        """A점수 기반 피드백 메시지"""
        score = a_score['total_score']
        
        if score >= 80:
            grade = 'A'
            feedback = '🏆 우수한 거래! 모든 면에서 잘 관리된 트레이딩입니다.'
        elif score >= 65:
            grade = 'B' 
            feedback = '👍 양호한 거래. 몇 가지 개선점이 있지만 전반적으로 좋습니다.'
        elif score >= 45:
            grade = 'C'
            feedback = '📝 보통 수준. 리스크 관리나 타이밍 개선이 필요합니다.'
        else:
            grade = 'D'
            feedback = '📉 아쉬운 거래. 전략 재검토가 필요합니다.'
        
        # 구체적 개선점
        improvements = []
        if a_score.get('risk_management_score', 0) < 50:
            improvements.append('리스크 관리 강화')
        if a_score.get('efficiency_score', 0) < 50:
            improvements.append('보유 기간 최적화')
        if a_score.get('raw_metrics', {}).get('return_pct', 0) < 0:
            improvements.append('손절 전략 재검토')
        
        return {
            'grade': grade,
            'feedback': feedback,
            'improvements': improvements
        }
    
    def run_scenario(self, symbol=None, date_range=None, interactive=True):
        """시나리오 실행"""
        print("🎮 트레이딩 시뮬레이터 시작!")
        print("=" * 50)
        
        # 랜덤 시나리오 선택
        scenario = self.get_random_scenario(symbol, date_range)
        if scenario is None:
            return
        
        symbol = scenario['symbol']
        entry_date = scenario['entry_datetime']
        entry_price = scenario['entry_price'] 
        exit_price = scenario['exit_price']
        return_pct = scenario['return_pct']
        
        print(f"📊 시나리오: {symbol}")
        print(f"📅 날짜: {entry_date}")
        print(f"💰 진입가: ${entry_price:.2f}")
        print("-" * 30)
        
        # 1단계: B유형 진입 분석
        print("\n🔮 1단계: 진입 조건 분석 (현재 시점)")
        b_result = self.calculate_b_score(scenario)
        recommendation = self.format_recommendation(b_result)
        
        print(f"진입 점수: {b_result['total_score']}/100")
        print(f"  ├ 기술적 조건: {b_result['technical_score']}/100")  
        print(f"  ├ 시장 환경: {b_result['market_env_score']}/100")
        print(f"  └ 리스크 관리: {b_result['risk_score']}/100")
        print(f"\n{recommendation['action']}")
        print(f"💡 {recommendation['message']}")
        print(f"🎯 신뢰도: {recommendation['confidence']} | 🛡️ 리스크: {recommendation['risk_level']}")
        
        if interactive:
            decision = input(f"\n❓ 매수하시겠습니까? (y/n): ").lower().strip()
            if decision != 'y':
                print("📋 거래 취소됨")
                return
        
        print(f"\n💸 매수 실행: ${entry_price:.2f}")
        print("⏳ 거래 진행 중...")
        print(f"💰 매도 완료: ${exit_price:.2f}")
        print(f"📈 수익률: {return_pct:.2f}%")
        
        # 2단계: A유형 품질 분석
        print("\n🎯 2단계: 거래 품질 분석 (완료 후)")
        a_result = self.calculate_a_score(scenario)
        feedback = self.format_feedback(a_result)
        
        print(f"품질 점수: {a_result['total_score']}/100 [{feedback['grade']}등급]")
        print(f"  ├ 리스크 관리: {a_result['risk_management_score']}/100")
        print(f"  └ 효율성: {a_result['efficiency_score']}/100")
        print(f"\n{feedback['feedback']}")
        
        if feedback['improvements']:
            print(f"📝 개선점:")
            for improvement in feedback['improvements']:
                print(f"  • {improvement}")
        
        # 결과 저장
        self.simulation_history.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return_pct': return_pct,
            'b_score': b_result,
            'a_score': a_result,
            'recommendation': recommendation,
            'feedback': feedback
        })
        
        print("\n" + "=" * 50)
        print("✅ 시뮬레이션 완료!")
        
        return {
            'scenario': scenario,
            'b_result': b_result,
            'a_result': a_result,
            'recommendation': recommendation,
            'feedback': feedback
        }
    
    def batch_test(self, n_scenarios=10, symbol=None):
        """배치 테스트"""
        print(f"🔄 {n_scenarios}개 시나리오 배치 테스트")
        print("=" * 50)
        
        results = []
        for i in range(n_scenarios):
            print(f"\n📊 시나리오 {i+1}/{n_scenarios}")
            result = self.run_scenario(symbol=symbol, interactive=False)
            if result:
                results.append(result)
        
        # 통계 분석
        if results:
            b_scores = [r['b_result']['total_score'] for r in results]
            a_scores = [r['a_result']['total_score'] for r in results]
            returns = [r['scenario']['return_pct'] for r in results]
            
            print(f"\n📊 배치 테스트 결과:")
            print(f"  B점수 평균: {np.mean(b_scores):.1f} ± {np.std(b_scores):.1f}")
            print(f"  A점수 평균: {np.mean(a_scores):.1f} ± {np.std(a_scores):.1f}")
            print(f"  수익률 평균: {np.mean(returns):.2f}% ± {np.std(returns):.2f}%")
            
            # B점수별 수익률 분석
            high_b_returns = [r['scenario']['return_pct'] for r in results if r['b_result']['total_score'] >= 70]
            low_b_returns = [r['scenario']['return_pct'] for r in results if r['b_result']['total_score'] < 70]
            
            if high_b_returns and low_b_returns:
                print(f"\n🎯 B점수별 성능:")
                print(f"  B점수 ≥70: 평균 수익률 {np.mean(high_b_returns):.2f}%")
                print(f"  B점수 <70: 평균 수익률 {np.mean(low_b_returns):.2f}%")
        
        return results
    
    def save_history(self, filename=None):
        """시뮬레이션 기록 저장"""
        if not filename:
            filename = f"simulation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.simulation_history, f, indent=2, ensure_ascii=False)
        
        print(f"💾 시뮬레이션 기록 저장: {filename}")

def main():
    """메인 실행 함수"""
    print("🚀 하이브리드 트레이딩 AI 시뮬레이터")
    print("=" * 50)
    
    # 시뮬레이터 초기화
    simulator = TradingSimulator()
    
    while True:
        print("\n📋 메뉴 선택:")
        print("1. 랜덤 시나리오 테스트")
        print("2. 특정 종목 테스트")  
        print("3. 배치 테스트 (10개)")
        print("4. 시뮬레이션 기록 저장")
        print("5. 종료")
        
        choice = input("\n선택 (1-5): ").strip()
        
        if choice == '1':
            simulator.run_scenario()
        elif choice == '2':
            symbol = input("종목 코드 입력 (예: AAPL): ").strip().upper()
            simulator.run_scenario(symbol=symbol if symbol else None)
        elif choice == '3':
            simulator.batch_test(n_scenarios=10)
        elif choice == '4':
            simulator.save_history()
        elif choice == '5':
            print("👋 시뮬레이터 종료!")
            break
        else:
            print("❌ 잘못된 선택입니다.")

if __name__ == "__main__":
    main()