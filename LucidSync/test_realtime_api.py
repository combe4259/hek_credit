#!/usr/bin/env python3
"""
실시간 AI Trading API 테스트
"""
import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000/api/ai/realtime"

def test_initialize():
    """모델 초기화"""
    print("="*50)
    print("AI 모델 초기화")
    print("="*50)
    
    response = requests.post(f"{BASE_URL}/initialize")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()

def test_buy_analysis(ticker="AAPL"):
    """실시간 매수 신호 분석"""
    print("\n" + "="*50)
    print(f"실시간 매수 신호 분석: {ticker}")
    print("="*50)
    
    data = {
        "ticker": ticker,
        "position_size_pct": 5.0  # 포트폴리오의 5%
    }
    
    response = requests.post(f"{BASE_URL}/buy-analysis", json=data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n📊 {ticker} 분석 결과:")
        print(f"  현재가: ${result['current_price']:.2f}")
        print(f"  신호 점수: {result['analysis']['signal_score']:.1f}/100")
        print(f"  추천: {result['analysis']['recommendation']}")
        print(f"  신뢰도: {result['analysis']['confidence']*100:.1f}%")
        
        print(f"\n📈 기술적 지표:")
        for key, value in result['technical_indicators'].items():
            print(f"  {key}: {value}")
        
        print(f"\n💰 펀더멘털:")
        print(f"  P/E: {result['market_data']['pe_ratio']:.1f}")
        print(f"  P/B: {result['market_data']['pb_ratio']:.1f}")
        print(f"  ROE: {result['market_data']['roe']:.2%}")
        print(f"  VIX: {result['market_data']['vix']:.1f}")
        
        if 'buy_recommendation' in result:
            print(f"\n✅ 매수 추천:")
            print(f"  포지션 크기: {result['buy_recommendation']['suggested_position_size']}")
            print(f"  신호 강도: {result['buy_recommendation']['signal_strength']}")
            print(f"  리스크: {result['buy_recommendation']['risk_level']}")
    else:
        print(f"Error: {response.text}")
    
    return response.json() if response.status_code == 200 else None

def test_sell_analysis(ticker="AAPL", entry_price=150.0):
    """실시간 매도 신호 분석"""
    print("\n" + "="*50)
    print(f"실시간 매도 신호 분석: {ticker}")
    print("="*50)
    
    # 10일 전 날짜 (예시)
    entry_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
    
    data = {
        "ticker": ticker,
        "entry_price": entry_price,
        "entry_date": entry_date,
        "position_size": 100  # 100주
    }
    
    response = requests.post(f"{BASE_URL}/sell-analysis", json=data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n📊 {ticker} 포지션 분석:")
        print(f"  매수가: ${result['entry_price']:.2f}")
        print(f"  현재가: ${result['current_price']:.2f}")
        print(f"  보유일: {result['holding_days']}일")
        print(f"  수익률: {result['current_return']}")
        
        print(f"\n📉 매도 신호:")
        print(f"  신호 점수: {result['analysis']['signal_score']:.1f}/100")
        print(f"  추천: {result['analysis']['recommendation']}")
        print(f"  신뢰도: {result['analysis']['confidence']*100:.1f}%")
        
        print(f"\n💼 성과:")
        print(f"  총 수익률: {result['performance']['total_return']}")
        print(f"  시장 수익률: {result['performance']['market_return']}")
        print(f"  초과 수익률: {result['performance']['excess_return']}")
        
        if 'sell_recommendation' in result:
            print(f"\n⚠️ 매도 추천:")
            print(f"  긴급도: {result['sell_recommendation']['urgency']}")
            print(f"  이유:")
            for reason in result['sell_recommendation']['reasons']:
                print(f"    - {reason}")
    else:
        print(f"Error: {response.text}")
    
    return response.json() if response.status_code == 200 else None

def test_quick_check(ticker="AAPL"):
    """빠른 가격 체크"""
    print("\n" + "="*50)
    print(f"빠른 가격 체크: {ticker}")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/quick-check/{ticker}")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        if result['status'] == 'success':
            print(f"  {ticker}: ${result['current_price']:.2f}")
        else:
            print(f"  Error: {result.get('message', 'Unknown error')}")
    
    return response.json() if response.status_code == 200 else None

if __name__ == "__main__":
    print("🚀 실시간 AI Trading API 테스트 시작\n")
    
    # 1. 모델 초기화
    init_result = test_initialize()
    
    if init_result and init_result.get('status') == 'success':
        # 2. 매수 신호 테스트 (여러 종목)
        for ticker in ["AAPL", "MSFT", "GOOGL"]:
            test_buy_analysis(ticker)
        
        # 3. 매도 신호 테스트
        test_sell_analysis("AAPL", entry_price=170.0)
        
        # 4. 빠른 가격 체크
        test_quick_check("TSLA")
    else:
        print("❌ 모델 초기화 실패")