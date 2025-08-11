# test_api.py
import requests
import json
import time

# API 기본 설정
BASE_URL = "http://localhost:8000"
headers = {"Content-Type": "application/json"}

def test_api_endpoint(method, endpoint, data=None, description=""):
    """API 엔드포인트 테스트 함수"""
    print(f"\n{'='*60}")
    print(f"🧪 테스트: {description}")
    print(f"📍 {method} {endpoint}")
    print(f"{'='*60}")

    try:
        if method == "GET":
            response = requests.get(f"{BASE_URL}{endpoint}")
        elif method == "POST":
            response = requests.post(f"{BASE_URL}{endpoint}", json=data, headers=headers)

        print(f"📊 상태코드: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ 성공!")
            print(f"📄 응답 데이터:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"❌ 실패: {response.text}")

    except Exception as e:
        print(f"💥 에러 발생: {e}")

    print(f"\n{'⏱️  대기 중...'}")
    time.sleep(1)  # API 부하 방지

def main():
    """전체 API 테스트 실행"""
    print("🚀 매매패턴 AI API 테스트 시작")
    print(f"🌐 서버 주소: {BASE_URL}")

    # 1. 기본 엔드포인트 테스트
    test_api_endpoint("GET", "/", description="API 기본 정보 확인")

    # 2. 헬스 체크
    test_api_endpoint("GET", "/health", description="서버 상태 확인")

    # 3. 데모 실행 (AI 모델이 훈련된 후)
    test_api_endpoint("POST", "/demo", description="데모 예측 실행")

    # 4. 간단한 거래 분석 테스트
    simple_trade_data = {
        "user_id": "test_user_001",
        "trades": [
            {"profit_rate": 0.08, "holding_days": 5},   # 8% 수익, 5일
            {"profit_rate": -0.04, "holding_days": 15}, # -4% 손실, 15일
            {"profit_rate": 0.12, "holding_days": 3},   # 12% 수익, 3일
            {"profit_rate": -0.02, "holding_days": 8},  # -2% 손실, 8일
            {"profit_rate": 0.06, "holding_days": 12},  # 6% 수익, 12일
            {"profit_rate": -0.08, "holding_days": 25}, # -8% 손실, 25일
            {"profit_rate": 0.15, "holding_days": 1},   # 15% 수익, 1일 (단타)
            {"profit_rate": 0.03, "holding_days": 20}   # 3% 수익, 20일
        ]
    }

    test_api_endpoint("POST", "/quick-analysis", simple_trade_data,
                      "간단한 거래 데이터로 투자 스타일 분석")

    # 5. 상세한 거래 스타일 예측 테스트
    detailed_trade_data = {
        "user_id": "test_user_002",
        "trades": [
            {
                "user_id": "test_user_002",
                "profit_rate": 0.07,
                "holding_days": 10,
                "market_volatility": 0.03,
                "market_trend": 1,
                "is_profit_taking": 1,
                "is_loss_cutting": 0,
                "is_panic_sell": 0,
                "is_diamond_hands": 0,
                "risk_tolerance": 0.6
            },
            {
                "user_id": "test_user_002",
                "profit_rate": -0.05,
                "holding_days": 20,
                "market_volatility": 0.04,
                "market_trend": -1,
                "is_profit_taking": 0,
                "is_loss_cutting": 1,
                "is_panic_sell": 0,
                "is_diamond_hands": 0,
                "risk_tolerance": 0.6
            },
            {
                "user_id": "test_user_002",
                "profit_rate": 0.12,
                "holding_days": 2,
                "market_volatility": 0.02,
                "market_trend": 1,
                "is_profit_taking": 1,
                "is_loss_cutting": 0,
                "is_panic_sell": 0,
                "is_diamond_hands": 0,
                "risk_tolerance": 0.8
            }
        ]
    }

    test_api_endpoint("POST", "/predict/trading-style", detailed_trade_data,
                      "상세 데이터로 투자 스타일 예측")

    # 6. 매도 확률 예측 테스트
    sell_prediction_data = {
        "user_id": "test_user_003",
        "current_profit_rate": 0.08,  # 현재 8% 수익
        "holding_days": 15,           # 15일 보유
        "market_volatility": 0.03,    # 3% 변동성
        "user_style_probs": {
            "보수적": 0.2,
            "공격적": 0.7,
            "단타형": 0.1
        }
    }

    test_api_endpoint("POST", "/predict/sell-probability", sell_prediction_data,
                      "현재 상황에서 매도 확률 예측")

    # 7. 모델 통계 정보
    test_api_endpoint("GET", "/model/stats", description="AI 모델 통계 정보")

    print(f"\n{'='*60}")
    print("🎉 모든 API 테스트 완료!")
    print("📖 더 자세한 API 문서: http://localhost:8000/docs")
    print("🔍 실시간 API 문서: http://localhost:8000/redoc")
    print(f"{'='*60}")

def test_different_trading_styles():
    """다양한 투자 스타일 테스트"""
    print("\n🎭 다양한 투자 스타일 테스트")

    # 보수적 투자자
    conservative_data = {
        "user_id": "conservative_investor",
        "trades": [
            {"profit_rate": 0.03, "holding_days": 30},  # 작은 수익으로 만족
            {"profit_rate": 0.04, "holding_days": 25},
            {"profit_rate": -0.02, "holding_days": 45}, # 손실을 오래 참음
            {"profit_rate": 0.02, "holding_days": 35},
            {"profit_rate": 0.05, "holding_days": 20}
        ]
    }

    test_api_endpoint("POST", "/quick-analysis", conservative_data,
                      "보수적 투자자 패턴 분석")

    # 공격적 투자자
    aggressive_data = {
        "user_id": "aggressive_investor",
        "trades": [
            {"profit_rate": 0.15, "holding_days": 7},   # 큰 수익 추구
            {"profit_rate": -0.12, "holding_days": 3},  # 큰 손실도 감수
            {"profit_rate": 0.22, "holding_days": 5},
            {"profit_rate": -0.08, "holding_days": 2},
            {"profit_rate": 0.18, "holding_days": 4}
        ]
    }

    test_api_endpoint("POST", "/quick-analysis", aggressive_data,
                      "공격적 투자자 패턴 분석")

    # 단타형 투자자
    day_trader_data = {
        "user_id": "day_trader",
        "trades": [
            {"profit_rate": 0.02, "holding_days": 1},   # 매우 짧은 보유
            {"profit_rate": 0.01, "holding_days": 1},
            {"profit_rate": -0.01, "holding_days": 1},
            {"profit_rate": 0.03, "holding_days": 2},
            {"profit_rate": 0.02, "holding_days": 1},
            {"profit_rate": -0.02, "holding_days": 1},
            {"profit_rate": 0.04, "holding_days": 2}
        ]
    }

    test_api_endpoint("POST", "/quick-analysis", day_trader_data,
                      "단타형 투자자 패턴 분석")

if __name__ == "__main__":
    print("⏳ 서버가 시작될 때까지 잠시 대기... (AI 모델 훈련 중)")
    time.sleep(5)  # 서버 시작 대기

    try:
        # 기본 테스트
        main()

        # 추가 스타일 테스트
        test_different_trading_styles()

    except KeyboardInterrupt:
        print("\n👋 테스트 중단됨")
    except Exception as e:
        print(f"\n💥 테스트 실행 중 에러: {e}")