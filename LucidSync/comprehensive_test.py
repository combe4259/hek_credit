#!/usr/bin/env python3
"""
종합 테스트 스크립트
"""
import requests
import json
import time
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000/api/ai/realtime"

def test_buy_signals():
    """여러 종목의 매수 신호 테스트"""
    print("\n" + "="*60)
    print("매수 신호 테스트")
    print("="*60)
    
    # 다양한 종목 테스트
    test_stocks = [
        "AAPL",   # Apple
        "MSFT",   # Microsoft  
        "NVDA",   # NVIDIA
        "TSLA",   # Tesla
        "GOOGL",  # Google
        "META",   # Meta
        "AMZN",   # Amazon
        "INVALID" # 잘못된 심볼 테스트
    ]
    
    results = []
    
    for ticker in test_stocks:
        print(f"\n테스트: {ticker}")
        print("-" * 40)
        
        try:
            response = requests.post(
                f"{BASE_URL}/buy-analysis",
                json={"ticker": ticker, "position_size_pct": 5.0},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 성공: {ticker}")
                print(f"  - 현재가: ${data.get('current_price', 'N/A')}")
                print(f"  - 추천: {data['analysis']['recommendation']}")
                print(f"  - 신호 점수: {data['analysis']['signal_score']:.1f}/100")
                print(f"  - 신뢰도: {data['analysis']['confidence']*100:.1f}%")
                
                results.append({
                    'ticker': ticker,
                    'status': 'success',
                    'recommendation': data['analysis']['recommendation'],
                    'score': data['analysis']['signal_score']
                })
            else:
                print(f"❌ 실패: {ticker}")
                print(f"  - 에러: {response.status_code}")
                print(f"  - 메시지: {response.text[:200]}")
                
                results.append({
                    'ticker': ticker,
                    'status': 'failed',
                    'error': response.status_code
                })
                
        except Exception as e:
            print(f"❌ 예외 발생: {ticker}")
            print(f"  - 에러: {str(e)}")
            
            results.append({
                'ticker': ticker,
                'status': 'exception',
                'error': str(e)
            })
        
        time.sleep(1)  # API 부하 방지
    
    return results

def test_sell_signals():
    """매도 신호 테스트"""
    print("\n" + "="*60)
    print("매도 신호 테스트")
    print("="*60)
    
    # 테스트 시나리오
    test_cases = [
        {
            'ticker': 'AAPL',
            'entry_price': 150.0,
            'days_ago': 30  # 30일 전 매수
        },
        {
            'ticker': 'MSFT',
            'entry_price': 300.0,
            'days_ago': 7   # 7일 전 매수
        },
        {
            'ticker': 'NVDA',
            'entry_price': 400.0,
            'days_ago': 60  # 60일 전 매수
        }
    ]
    
    results = []
    
    for test in test_cases:
        ticker = test['ticker']
        entry_date = (datetime.now() - timedelta(days=test['days_ago'])).strftime('%Y-%m-%d')
        
        print(f"\n테스트: {ticker}")
        print(f"  매수가: ${test['entry_price']}")
        print(f"  매수일: {entry_date}")
        print("-" * 40)
        
        try:
            response = requests.post(
                f"{BASE_URL}/sell-analysis",
                json={
                    "ticker": ticker,
                    "entry_price": test['entry_price'],
                    "entry_date": entry_date,
                    "position_size": 100
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 성공: {ticker}")
                print(f"  - 현재가: ${data.get('current_price', 'N/A')}")
                print(f"  - 수익률: {data.get('current_return', 'N/A')}")
                print(f"  - 추천: {data['analysis']['recommendation']}")
                print(f"  - 신호 점수: {data['analysis']['signal_score']:.1f}/100")
                
                results.append({
                    'ticker': ticker,
                    'status': 'success',
                    'return': data.get('current_return'),
                    'recommendation': data['analysis']['recommendation']
                })
            else:
                print(f"❌ 실패: {ticker}")
                print(f"  - 에러: {response.status_code}")
                print(f"  - 메시지: {response.text[:200]}")
                
                results.append({
                    'ticker': ticker,
                    'status': 'failed',
                    'error': response.status_code
                })
                
        except Exception as e:
            print(f"❌ 예외 발생: {ticker}")
            print(f"  - 에러: {str(e)}")
            
            results.append({
                'ticker': ticker,
                'status': 'exception',
                'error': str(e)
            })
        
        time.sleep(1)
    
    return results

def test_edge_cases():
    """엣지 케이스 테스트"""
    print("\n" + "="*60)
    print("엣지 케이스 테스트")
    print("="*60)
    
    # 1. 빈 요청
    print("\n1. 빈 요청 테스트")
    try:
        response = requests.post(f"{BASE_URL}/buy-analysis", json={})
        print(f"  상태 코드: {response.status_code}")
        if response.status_code != 200:
            print("  ✅ 올바른 에러 처리")
    except Exception as e:
        print(f"  ✅ 예외 처리: {str(e)[:50]}")
    
    # 2. 잘못된 날짜 형식
    print("\n2. 잘못된 날짜 형식 테스트")
    try:
        response = requests.post(
            f"{BASE_URL}/sell-analysis",
            json={
                "ticker": "AAPL",
                "entry_price": 150,
                "entry_date": "2024/01/01",  # 잘못된 형식
                "position_size": 100
            }
        )
        print(f"  상태 코드: {response.status_code}")
        if response.status_code != 200:
            print("  ✅ 날짜 형식 검증")
    except Exception as e:
        print(f"  ✅ 예외 처리: {str(e)[:50]}")
    
    # 3. 음수 가격
    print("\n3. 음수 가격 테스트")
    try:
        response = requests.post(
            f"{BASE_URL}/sell-analysis",
            json={
                "ticker": "AAPL",
                "entry_price": -150,  # 음수 가격
                "entry_date": "2024-01-01",
                "position_size": 100
            }
        )
        print(f"  상태 코드: {response.status_code}")
        result = response.json()
        if 'current_return' in result:
            print(f"  수익률: {result['current_return']}")
    except Exception as e:
        print(f"  에러: {str(e)[:50]}")
    
    # 4. 매우 큰 포지션 크기
    print("\n4. 큰 포지션 크기 테스트")
    try:
        response = requests.post(
            f"{BASE_URL}/buy-analysis",
            json={
                "ticker": "AAPL",
                "position_size_pct": 100.0  # 100% 포지션
            }
        )
        print(f"  상태 코드: {response.status_code}")
        if response.status_code == 200:
            print("  ✅ 큰 포지션 처리 가능")
    except Exception as e:
        print(f"  에러: {str(e)[:50]}")

def test_performance():
    """성능 테스트"""
    print("\n" + "="*60)
    print("성능 테스트")
    print("="*60)
    
    # 동시 요청 테스트
    import concurrent.futures
    
    def make_request(ticker):
        start_time = time.time()
        try:
            response = requests.post(
                f"{BASE_URL}/buy-analysis",
                json={"ticker": ticker, "position_size_pct": 5.0},
                timeout=30
            )
            elapsed = time.time() - start_time
            return ticker, response.status_code, elapsed
        except Exception as e:
            elapsed = time.time() - start_time
            return ticker, 'error', elapsed
    
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    print("\n동시 요청 테스트 (5개 종목)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request, ticker) for ticker in tickers]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    for ticker, status, elapsed in results:
        print(f"  {ticker}: {status} ({elapsed:.2f}초)")
    
    avg_time = sum(r[2] for r in results) / len(results)
    print(f"\n평균 응답 시간: {avg_time:.2f}초")

def main():
    print("🚀 종합 테스트 시작")
    print("="*60)
    
    # 1. 모델 상태 확인
    print("\n📊 모델 상태 확인")
    try:
        response = requests.get(f"{BASE_URL}/model-status")
        if response.status_code == 200:
            status = response.json()
            print(f"  초기화: {status.get('initialized', False)}")
            print(f"  준비 상태: {status.get('ready', False)}")
            if not status.get('initialized'):
                print("\n모델 초기화 중...")
                init_response = requests.post(f"{BASE_URL}/initialize")
                print(f"  초기화 결과: {init_response.json().get('status')}")
        else:
            print("  ❌ 상태 확인 실패")
    except Exception as e:
        print(f"  ❌ 연결 실패: {str(e)}")
        return
    
    # 2. 매수 신호 테스트
    buy_results = test_buy_signals()
    
    # 3. 매도 신호 테스트
    sell_results = test_sell_signals()
    
    # 4. 엣지 케이스 테스트
    test_edge_cases()
    
    # 5. 성능 테스트
    test_performance()
    
    # 결과 요약
    print("\n" + "="*60)
    print("📈 테스트 결과 요약")
    print("="*60)
    
    # 매수 신호 결과
    success_buy = sum(1 for r in buy_results if r['status'] == 'success')
    print(f"\n매수 신호 테스트: {success_buy}/{len(buy_results)} 성공")
    
    buy_recommendations = [r for r in buy_results if r['status'] == 'success']
    if buy_recommendations:
        buy_signals = [r for r in buy_recommendations if r.get('recommendation') == 'BUY']
        print(f"  - 매수 추천: {len(buy_signals)}개")
        print(f"  - 평균 점수: {sum(r['score'] for r in buy_recommendations)/len(buy_recommendations):.1f}")
    
    # 매도 신호 결과
    success_sell = sum(1 for r in sell_results if r['status'] == 'success')
    print(f"\n매도 신호 테스트: {success_sell}/{len(sell_results)} 성공")
    
    sell_recommendations = [r for r in sell_results if r['status'] == 'success']
    if sell_recommendations:
        sell_signals = [r for r in sell_recommendations if r.get('recommendation') == 'SELL']
        print(f"  - 매도 추천: {len(sell_signals)}개")
    
    print("\n✅ 테스트 완료!")

if __name__ == "__main__":
    main()