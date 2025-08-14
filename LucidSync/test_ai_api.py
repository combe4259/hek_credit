#!/usr/bin/env python3
"""
AI Trading API 테스트 스크립트
"""
import requests
import json

# API 베이스 URL
BASE_URL = "http://localhost:8000/api/ai-trading"

def test_buy_signal():
    """매수 신호 테스트 - 모든 필수 feature 포함"""
    print("=" * 50)
    print("매수 신호 테스트")
    print("=" * 50)
    
    # 모델이 필요로 하는 모든 feature 준비
    data = {
        "ticker": "AAPL",
        "current_price": 180.5,
        "volume": 50000000,
        "ma_20": 175.3,
        "ma_50": 172.8,
        "rsi": 55.2,
        "market_data": {
            # 모멘텀 지표
            "entry_momentum_5d": 0.03,
            "entry_momentum_60d": 0.15,
            
            # 이동평균 편차
            "entry_ma_dev_5d": 0.02,
            "entry_ma_dev_60d": 0.05,
            
            # 변동성
            "entry_volatility_5d": 0.015,
            "entry_volatility_60d": 0.025,
            
            # 거래량 변화
            "entry_vol_change_5d": 0.1,
            "entry_vol_change_20d": 0.05,
            "entry_vol_change_60d": -0.02,
            
            # 펀더멘털 지표
            "entry_pb_ratio": 35.5,
            "entry_operating_margin": 0.297,
            "entry_debt_equity_ratio": 1.95,
            
            # 시장 지표
            "market_entry_ma_return_5d": 0.01,
            "market_entry_ma_return_20d": 0.03,
            "market_entry_cum_return_5d": 0.02,
            "market_entry_volatility_20d": 0.012,
            
            # 포지션 크기
            "position_size_pct": 5.0
        }
    }
    
    response = requests.post(f"{BASE_URL}/buy-signal", json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()

def test_sell_signal():
    """매도 신호 테스트"""
    print("\n" + "=" * 50)
    print("매도 신호 테스트")
    print("=" * 50)
    
    data = {
        "ticker": "AAPL",
        "entry_price": 170.0,
        "current_price": 180.5,
        "holding_days": 10,
        "position_size": 100,
        "market_data": {
            # 추가 시장 데이터 (필요시)
            "current_momentum": 0.05,
            "market_volatility": 0.015
        }
    }
    
    response = requests.post(f"{BASE_URL}/sell-signal", json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()

def test_evaluate_trade():
    """거래 품질 평가 테스트"""
    print("\n" + "=" * 50)
    print("거래 품질 평가 테스트")
    print("=" * 50)
    
    data = {
        "ticker": "AAPL",
        "entry_price": 170.0,
        "exit_price": 180.5,
        "entry_date": "2024-01-01",
        "exit_date": "2024-01-15",
        "position_size": 5.0,
        "trade_type": "long"
    }
    
    response = requests.post(f"{BASE_URL}/evaluate-trade", json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()

def test_model_status():
    """모델 상태 확인"""
    print("\n" + "=" * 50)
    print("모델 상태 확인")
    print("=" * 50)
    
    response = requests.get(f"{BASE_URL}/model-status")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()

if __name__ == "__main__":
    # 1. 먼저 모델 초기화
    print("모델 초기화 중...")
    response = requests.post(f"{BASE_URL}/initialize")
    print(f"초기화 결과: {response.json()['status']}")
    
    # 2. 모델 상태 확인
    test_model_status()
    
    # 3. 각 API 테스트
    try:
        test_buy_signal()
    except Exception as e:
        print(f"매수 신호 테스트 실패: {e}")
    
    try:
        test_sell_signal()
    except Exception as e:
        print(f"매도 신호 테스트 실패: {e}")
    
    try:
        test_evaluate_trade()
    except Exception as e:
        print(f"거래 평가 테스트 실패: {e}")