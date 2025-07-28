import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import sys

# 모듈 임포트
from data.data_loader import DataLoader
from patterns.pattern_analyzer import PatternAnalyzer
from simulation.trading_simulator import TradingSimulator
from utils.visualization import PatternVisualizer
from config import *

def setup_environment():
    """환경 설정 및 출력 디렉토리 생성"""
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 로깅 설정
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    plt.style.use('seaborn-v0_8')
    
    print(f"""
    ==========================================
       Trading Pattern Analysis for {SYMBOL}
       Period: {PERIOD}
       Output Directory: {os.path.abspath(OUTPUT_DIR)}
    ==========================================
    """)

def main():
    # 환경 설정
    setup_environment()
    
    try:
        # 1. 데이터 로드
        print("\n[1/4] 데이터 로딩 중...")
        data_loader = DataLoader(SYMBOL, PERIOD)
        df = data_loader.load_data()
        
        if df is None or df.empty:
            raise ValueError("데이터 로딩 실패!")
        
        # 2. 기술적 지표 계산
        print("\n[2/4] 기술적 지표 계산 중...")
        df = data_loader.calculate_technical_indicators()
        
        # 3. 트레이딩 시뮬레이션 실행 (패턴 분석기는 시뮬레이터 내부에서 생성됨)
        print("\n[3/4] 트레이딩 시뮬레이션 실행 중...")
        simulator = TradingSimulator(df)
        
        # 모든 프로필에 대해 시뮬레이션 실행
        results = simulator.simulate_all_profiles()
        
        if not results:
            raise ValueError("시뮬레이션 결과가 없습니다!")
        
        # 4. 결과 분석 및 시각화
        print("\n[4/4] 결과 분석 및 저장 중...")
        visualizer = PatternVisualizer(results)
        
        # 데이터셋 요약 정보 출력
        dataset = visualizer.create_dataset_summary()
        
        # 시각화 생성
        visualizer.create_pattern_analysis_chart(OUTPUT_CHART)
        visualizer.create_correlation_heatmap()
        
        # 데이터셋 저장 (ML 분석용)
        visualizer.save_dataset(OUTPUT_CSV, ml_format=True)
        
        print("\n✅ 시뮬레이션이 성공적으로 완료되었습니다!")
        print(f"- 결과 파일: {os.path.abspath(OUTPUT_CSV)}")
        print(f"- 차트 파일: {os.path.abspath(OUTPUT_CHART)}")
        
        return 0
        
    except Exception as e:
        print(f"\nX 오류 발생: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())