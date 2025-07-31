#!/usr/bin/env python3
"""
데이터 생성 메인 실행 파일
NVDA 주식 데이터를 기반으로 매매 패턴 데이터를 생성합니다.
"""

import os
import sys
import pandas as pd
from datetime import datetime

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 모듈 임포트
from data.data_loader import DataLoader
from hybrid_simulator import HybridTradingSimulator
from patterns.pattern_analyzer import PatternAnalyzer
from utils.visualization import PatternVisualizer
from advanced_data_generator import AdvancedDataGenerator
from config import *

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("🚀 매매 패턴 데이터 생성 시작")
    print("=" * 60)
    
    try:
        # 1. 데이터 로드
        print("\n📊 1단계: 주식 데이터 로드")
        data_loader = DataLoader(symbol=SYMBOL, period=PERIOD)
        stock_data = data_loader.load_data()
        
        if stock_data is None or stock_data.empty:
            print("❌ 데이터 로드 실패")
            return
            
        # 2. 기술적 지표 계산
        print("\n📈 2단계: 기술적 지표 계산")
        processed_data = data_loader.calculate_technical_indicators()
        
        print(f"✅ 데이터 처리 완료: {len(processed_data)}일치 데이터")
        print(f"   - 시작일: {processed_data['date'].min()}")
        print(f"   - 종료일: {processed_data['date'].max()}")
        print(f"   - 컬럼 수: {len(processed_data.columns)}")
        
        # 3. 고급 매매 데이터 생성
        print("\n🎯 3단계: 고급 매매 데이터 생성")
        advanced_generator = AdvancedDataGenerator()
        advanced_generator.save_advanced_dataset("output/advanced_trading_data.csv")
        
        # 생성된 데이터 로드
        df = pd.read_csv("output/advanced_trading_data.csv")
        print(f"✅ 고급 데이터 생성 완료: {len(df):,}개 레코드")
        
        # 메타데이터 저장
        metadata_file = f'{OUTPUT_DIR}/advanced_trading_data_metadata.txt'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(f"고급 매매 패턴 데이터 생성 정보\n")
            f.write(f"=" * 40 + "\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"데이터 타입: 고급 매매 패턴 (Advanced Trading Data)\n")
            f.write(f"총 레코드 수: {len(df):,}개\n")
            f.write(f"유저 수: 300개\n")
            f.write(f"종목 수: 30개\n")
            f.write(f"분석 기간: 365일\n")
            f.write(f"\n컬럼 정보:\n")
            for col in df.columns:
                f.write(f"- {col}\n")
        
        print(f"✅ 메타데이터 저장: {metadata_file}")

        
        # 6. 결과 요약
        print("\n" + "=" * 60)
        print("🎉 고급 매매 데이터 생성 완료!")
        print("=" * 60)
        print(f"📁 출력 디렉토리: {OUTPUT_DIR}")
        print(f"📊 생성된 데이터: {len(df):,}개 레코드")
        print(f"👥 유저 수: 300개")
        print(f"📈 종목 수: 30개")
        print(f"📅 분석 기간: 365일")
        
        # 데이터 샘플 출력 (실제 컬럼 확인 후 출력)
        print(f"\n📋 데이터 컬럼: {list(df.columns)}")
        print(f"\n📋 데이터 샘플 (첫 3개 레코드):")
        print(df.head(3).to_string(index=False))
        
        # 액션별 분포
        if 'action' in df.columns:
            action_counts = df['action'].value_counts()
            print(f"\n📊 액션 분포:")
            for action, count in action_counts.items():
                percentage = count / len(df) * 100
                print(f"   {action}: {count:,}개 ({percentage:.1f}%)")
        
        print(f"\n✅ 모든 파일이 '{OUTPUT_DIR}' 디렉토리에 저장되었습니다.")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
