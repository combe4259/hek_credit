# utils/visualization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from datetime import datetime
import os
from config import *

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Arial Unicode MS', 'AppleGothic', 'Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PatternVisualizer:
    """패턴 분석 결과 및 데이터셋 생성"""
    
    def __init__(self, pattern_data: List[Dict]):
        self.df = pd.DataFrame(pattern_data)
        self._preprocess_data()
    
    def _preprocess_data(self):
        """데이터 전처리"""
        if self.df.empty:
            return
            
        # 날짜 타입 변환
        if 'timestamp' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['timestamp']).dt.date
            self.df['day_of_week'] = pd.to_datetime(self.df['timestamp']).dt.dayofweek
            self.df['is_month_start'] = pd.to_datetime(self.df['timestamp']).dt.is_month_start.astype(int)
            self.df['is_month_end'] = pd.to_datetime(self.df['timestamp']).dt.is_month_end.astype(int)
        
        # 액션 인코딩
        self.df['action_encoded'] = self.df['action'].map({'BUY': 1, 'SELL': -1, 'HOLD': 0})
        
        # 수익률 범주화
        for period in ['1d', '7d', '30d']:
            col = f'return_{period}'
            if col in self.df.columns:
                # 데이터가 충분한지 확인
                unique_values = self.df[col].nunique()
                
                if unique_values < 5 or len(self.df) < 5:
                    # 데이터가 부족하면 간단한 범주화 사용
                    self.df[f'{col}_category'] = pd.cut(
                        self.df[col], 
                        bins=[-float('inf'), -0.02, -0.01, 0.01, 0.02, float('inf')],
                        labels=['매우 하락', '하락', '보합', '상승', '매우 상승']
                    )
                else:
                    # 충분한 데이터가 있으면 분위수 사용
                    try:
                        self.df[f'{col}_category'] = pd.qcut(
                            self.df[col], 
                            q=5, 
                            labels=['매우 하락', '하락', '보합', '상승', '매우 상승'],
                            duplicates='drop'  # 중복 경계값 처리
                        )
                    except ValueError:
                        # qcut이 실패하면 cut 사용
                        self.df[f'{col}_category'] = pd.cut(
                            self.df[col], 
                            bins=[-float('inf'), -0.02, -0.01, 0.01, 0.02, float('inf')],
                            labels=['매우 하락', '하락', '보합', '상승', '매우 상승']
                        )
    
    def create_dataset_summary(self) -> pd.DataFrame:
        """데이터셋 요약 정보 생성 및 ML용 피처 엔지니어링"""
        print("\n=== 데이터셋 요약 ===")
        
        if self.df.empty:
            print("X 데이터가 없습니다!")
            return pd.DataFrame()
        
        # 기본 정보 출력
        print(f"\n1. 데이터셋 개요:")
        print(f"   - 기간: {self.df['timestamp'].min()} ~ {self.df['timestamp'].max()}")
        print(f"   - 총 레코드 수: {len(self.df):,}건")
        
        # 투자자 프로필별 통계
        print("\n2. 투자자 프로필별 거래 현황:")
        profile_stats = self.df.groupby('investor_profile')['action'].value_counts().unstack().fillna(0)
        print(profile_stats)
        
        # 8가지 핵심 패턴 점수 요약
        print("\n3. 8가지 핵심 매매 패턴 점수 요약:")
        pattern_cols = [
            'technical_indicator_reliance', 'chart_pattern_recognition',
            'volume_reaction', 'candle_analysis', 'profit_taking_tendency',
            'stop_loss_tendency', 'volatility_reaction', 'time_based_trading'
        ]
        
        pattern_stats = self.df[pattern_cols].describe().T
        print("\n" + str(pattern_stats))
        
        # 수익률 통계
        if 'return_1d' in self.df.columns:
            print("\n4. 수익률 통계 (1일, 7일, 30일):")
            for period in ['1d', '7d', '30d']:
                col = f'return_{period}'
                print(f"\n{period} 수익률:")
                print(self.df[col].describe().to_string())
        
        return self.df
    
    def save_dataset(self, filename: str = OUTPUT_CSV, ml_format: bool = True):
        """데이터셋을 CSV로 저장 (ML 분석용)
        
        Args:
            filename: 저장할 파일 경로
            ml_format: True면 ML 분석에 적합한 형식으로 저장
        """
        if self.df.empty:
            print("X 저장할 데이터가 없습니다!")
            return
            
        # 디렉토리 생성
        directory = os.path.dirname(filename)
        if directory:  # 디렉토리 경로가 있는 경우에만 생성
            os.makedirs(directory, exist_ok=True)
        
        if ml_format:
            # ML 분석용 컬럼만 선택
            ml_columns = [
                # 기본 정보
                'timestamp', 'investor_profile', 'action', 'price',
                
                # 8가지 핵심 패턴
                'technical_indicator_reliance',  # 기술적 지표 의존도
                'chart_pattern_recognition',     # 차트 패턴 인식
                'volume_reaction',               # 거래량 반응
                'candle_analysis',               # 캔들 분석
                'profit_taking_tendency',        # 수익 실현 성향
                'stop_loss_tendency',            # 손절 성향
                'volatility_reaction',           # 변동성 반응
                'time_based_trading',            # 시간대별 거래
                
                # 시장 상황
                'rsi', 'macd_signal', 'bb_position', 'volume_ratio',
                'daily_return', 'gap',
                
                # 결과
                'return_1d', 'return_7d', 'return_30d',
                'return_1d_category', 'return_7d_category', 'return_30d_category'
            ]
            
            # 존재하는 컬럼만 선택
            ml_columns = [col for col in ml_columns if col in self.df.columns]
            ml_df = self.df[ml_columns]
            
            # CSV 저장
            ml_df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"✅ ML 분석용 데이터셋 저장 완료: {filename}")
            print(f"   - 컬럼 수: {len(ml_columns)}개")
            print(f"   - 레코드 수: {len(ml_df):,}건")
            
            # 추가 메타데이터 저장
            meta_filename = os.path.splitext(filename)[0] + '_metadata.txt'
            with open(meta_filename, 'w', encoding='utf-8') as f:
                f.write(f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"기간: {self.df['timestamp'].min()} ~ {self.df['timestamp'].max()}\n")
                f.write(f"총 레코드 수: {len(ml_df):,}건\n\n")
                
                f.write("[컬럼 설명]\n")
                column_descriptions = {
                    'technical_indicator_reliance': '기술적 지표 의존도 (0-1: 낮음-높음)',
                    'chart_pattern_recognition': '차트 패턴 인식 민감도 (0-1: 무시-민감)',
                    'volume_reaction': '거래량 반응 (0-1: 무시-추종)',
                    'candle_analysis': '캔들 분석 (0-1: 무시-민감)',
                    'profit_taking_tendency': '수익 실현 성향 (0-1: 단타-장기보유)',
                    'stop_loss_tendency': '손절 성향 (0-1: 빠른손절-버티기)',
                    'volatility_reaction': '변동성 반응 (0-1: 패닉-침착)',
                    'time_based_trading': '시간대별 거래 (0-1: 충동적-계획적)'
                }
                
                for col in ml_columns:
                    desc = column_descriptions.get(col, '')
                    f.write(f"{col}: {desc}\n")
            
            print(f"메타데이터 저장 완료: {meta_filename}")
        else:
            # 원본 형식 그대로 저장
            self.df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"데이터셋 저장 완료: {filename}")
        
        # 저장된 파일 정보
        file_size = os.path.getsize(filename) / (1024 * 1024)  # MB 단위
        print(f"   - 파일 크기: {file_size:.2f} MB")
    
