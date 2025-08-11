#!/usr/bin/env python3
"""
AI 트레이딩 시나리오 테스터
다양한 시장 상황에서 AI의 매매 의사결정을 테스트합니다.
"""

from advanced_trading_ai_v2 import AdvancedTradingAI

def test_trading_scenarios():
    """다양한 시나리오로 AI 매매 의사결정 테스트"""
    
    # AI 모델 로드
    ai = AdvancedTradingAI()
    try:
        ai.load_model('trained_trading_ai_v2.pkl')
    except FileNotFoundError:
        print("❌ 훈련된 모델이 없습니다. 먼저 advanced_trading_ai_v2.py를 실행해주세요.")
        return
    
    print("=" * 80)
    print("🎯 AI 트레이딩 시나리오 테스트")
    print("=" * 80)
    
    scenarios = [
        {
            'name': '수익 중 - 보유 권장',
            'ticker': 'NVDA',
            'stock_name': 'NVIDIA',
            'current_profit_rate': 0.068,
            'holding_days': 8,
            'current_time': '14:30',
            'market_data': {
                'sector': '전자',
                'market_cap': '대형주',
                'daily_volatility': 0.021,
                'market_condition': '상승장'
            }
        },
        {
            'name': '손실 중 - 손절 고려',
            'ticker': 'NVDA',
            'stock_name': 'NVIDIA',
            'current_profit_rate': -0.042,
            'holding_days': 15,
            'current_time': '10:30',
            'market_data': {
                'sector': '전자',
                'market_cap': '대형주',
                'daily_volatility': 0.035,
                'market_condition': '하락장'
            }
        },
        {
            'name': '대박 수익 - 익절 고민',
            'ticker': 'AAPL',
            'stock_name': 'Apple',
            'current_profit_rate': 0.25,
            'holding_days': 45,
            'current_time': '15:30',
            'market_data': {
                'sector': '테크',
                'market_cap': '대형주',
                'daily_volatility': 0.018,
                'market_condition': '상승장'
            }
        },
        {
            'name': '단기 급락 - 패닉 상황',
            'ticker': 'TSLA',
            'stock_name': 'Tesla',
            'current_profit_rate': -0.12,
            'holding_days': 3,
            'current_time': '09:30',
            'market_data': {
                'sector': '자동차',
                'market_cap': '대형주',
                'daily_volatility': 0.058,
                'market_condition': '하락장'
            }
        },
        {
            'name': '중소형주 횡보 - 애매한 상황',
            'ticker': 'SMALL',
            'stock_name': 'Small Cap Stock',
            'current_profit_rate': 0.015,
            'holding_days': 20,
            'current_time': '11:45',
            'market_data': {
                'sector': '바이오',
                'market_cap': '소형주',
                'daily_volatility': 0.045,
                'market_condition': '횡보장'
            }
        },
        {
            'name': '장 마감 전 급등 - 타이밍 중요',
            'ticker': 'MSFT',
            'stock_name': 'Microsoft',
            'current_profit_rate': 0.085,
            'holding_days': 2,
            'current_time': '15:50',
            'market_data': {
                'sector': '소프트웨어',
                'market_cap': '대형주',
                'daily_volatility': 0.025,
                'market_condition': '상승장'
            }
        },
        {
            'name': '장기 손실 버티기 - 물타기 고민',
            'ticker': 'GOOGL',
            'stock_name': 'Google',
            'current_profit_rate': -0.08,
            'holding_days': 60,
            'current_time': '13:15',
            'market_data': {
                'sector': '인터넷',
                'market_cap': '대형주',
                'daily_volatility': 0.022,
                'market_condition': '횡보장'
            }
        },
        {
            'name': '고변동성 종목 - 리스크 관리',
            'ticker': 'CRYPTO',
            'stock_name': 'Crypto Stock',
            'current_profit_rate': 0.15,
            'holding_days': 7,
            'current_time': '10:00',
            'market_data': {
                'sector': '핀테크',
                'market_cap': '중형주',
                'daily_volatility': 0.075,
                'market_condition': '상승장'
            }
        },
        {
            'name': '아침 장 시작 - 갭상승',
            'ticker': 'AMD',
            'stock_name': 'AMD',
            'current_profit_rate': 0.12,
            'holding_days': 5,
            'current_time': '09:35',
            'market_data': {
                'sector': '반도체',
                'market_cap': '대형주',
                'daily_volatility': 0.038,
                'market_condition': '상승장'
            }
        },
        {
            'name': '점심시간 횡보 - 관망',
            'ticker': 'SPY',
            'stock_name': 'S&P 500 ETF',
            'current_profit_rate': 0.022,
            'holding_days': 12,
            'current_time': '12:30',
            'market_data': {
                'sector': 'ETF',
                'market_cap': '대형주',
                'daily_volatility': 0.015,
                'market_condition': '횡보장'
            }
        }
    ]
    
    # 각 시나리오 테스트
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📊 시나리오 {i} - {scenario['name']}")
        
        result = ai.predict_realtime(
            ticker=scenario['ticker'],
            stock_name=scenario['stock_name'],
            current_profit_rate=scenario['current_profit_rate'],
            holding_days=scenario['holding_days'],
            current_time=scenario['current_time'],
            market_data=scenario['market_data']
        )
        
        print(f"종목: {result['stock_name']} ({result['ticker']})")
        print(f"현재 상태: {result['current_status']['profit_rate']} ({result['current_status']['holding_days']})")
        print(f"시간: {result['current_status']['time']}")
        print(f"\n분석:")
        print(f"  - 매도 확률: {result['analysis']['sell_probability']}")
        print(f"  - 임계값: {result['analysis']['optimal_threshold']}")
        print(f"  - 결정: {result['analysis']['decision']}")
        print(f"\n추천: {result['recommendation']['summary']}")
        for reason in result['recommendation']['reasons']:
            print(f"  - {reason}")
            
        # 3-Class 예측도 있다면 표시
        if result['analysis']['action_prediction'] is not None:
            action_pred = result['analysis']['action_prediction']
            print(f"\n🎯 3-Class 액션 예측:")
            print(f"  - BUY 확률: {action_pred['BUY_prob']}")
            print(f"  - HOLD 확률: {action_pred['HOLD_prob']}")
            print(f"  - SELL 확률: {action_pred['SELL_prob']}")
            print(f"  - 추천 액션: {action_pred['predicted_action']} (확신도: {action_pred['confidence']})")
        
        print("-" * 60)
    
    print(f"\n✅ 총 {len(scenarios)}개 시나리오 테스트 완료!")
    print("\n📈 시나리오 분석 요약:")
    print("  - AI는 다양한 시장 상황에서 일관된 논리로 의사결정")
    print("  - 수익률, 보유기간, 시간대, 변동성을 종합 고려")
    print("  - 리스크 관리와 수익 실현의 균형점 찾기")

def test_custom_scenario():
    """사용자 정의 시나리오 테스트"""
    print("\n" + "=" * 60)
    print("🎮 커스텀 시나리오 테스트")
    print("=" * 60)
    
    # AI 모델 로드
    ai = AdvancedTradingAI()
    try:
        ai.load_model('trained_trading_ai_v2.pkl')
    except FileNotFoundError:
        print("❌ 훈련된 모델이 없습니다.")
        return
    
    try:
        ticker = input("종목 코드 (예: AAPL): ").upper()
        stock_name = input("종목명 (예: Apple): ")
        profit_rate = float(input("현재 수익률 (예: 0.05 = 5%): "))
        holding_days = int(input("보유 기간 (일): "))
        current_time = input("현재 시간 (예: 14:30): ")
        sector = input("섹터 (예: 테크): ")
        market_cap = input("시총 (대형주/중형주/소형주): ")
        volatility = float(input("일일 변동성 (예: 0.02 = 2%): "))
        market_condition = input("시장 상황 (상승장/하락장/횡보장): ")
        
        result = ai.predict_realtime(
            ticker=ticker,
            stock_name=stock_name,
            current_profit_rate=profit_rate,
            holding_days=holding_days,
            current_time=current_time,
            market_data={
                'sector': sector,
                'market_cap': market_cap,
                'daily_volatility': volatility,
                'market_condition': market_condition
            }
        )
        
        print(f"\n📊 {stock_name} ({ticker}) 분석 결과:")
        print(f"현재 상태: {result['current_status']['profit_rate']} ({result['current_status']['holding_days']})")
        print(f"\n🤖 AI 분석:")
        print(f"  - 매도 확률: {result['analysis']['sell_probability']}")
        print(f"  - 결정: {result['analysis']['decision']}")
        print(f"\n✅ 최종 추천: {result['recommendation']['summary']}")
        for reason in result['recommendation']['reasons']:
            print(f"  - {reason}")
            
        # 3-Class 예측 표시
        if result['analysis']['action_prediction'] is not None:
            action_pred = result['analysis']['action_prediction']
            print(f"\n🎯 3-Class 액션 예측:")
            print(f"  - BUY 확률: {action_pred['BUY_prob']}")
            print(f"  - HOLD 확률: {action_pred['HOLD_prob']}")
            print(f"  - SELL 확률: {action_pred['SELL_prob']}")
            print(f"  - 추천 액션: {action_pred['predicted_action']} (확신도: {action_pred['confidence']})")
            
    except (ValueError, KeyboardInterrupt):
        print("\n❌ 입력이 취소되었거나 잘못되었습니다.")

if __name__ == "__main__":
    # 기본 시나리오 테스트
    test_trading_scenarios()
    
    # 사용자 정의 시나리오 테스트 (선택사항)
    print("\n" + "=" * 80)
    custom_test = input("🎮 커스텀 시나리오를 테스트하시겠습니까? (y/n): ")
    if custom_test.lower() == 'y':
        test_custom_scenario()