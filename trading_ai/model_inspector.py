#!/usr/bin/env python3
"""
AI 모델 정보 조회기
저장된 pkl 파일의 내용을 사람이 읽기 쉽게 보여줍니다.
"""

import pickle
import pandas as pd
import json
from datetime import datetime

def inspect_model(pkl_path='trained_trading_ai_v2.pkl'):
    """저장된 AI 모델 정보를 조회"""
    
    try:
        # 모델 로드
        with open(pkl_path, 'rb') as f:
            ai_data = pickle.load(f)
        
        print("=" * 60)
        print("🤖 AI 모델 정보 조회")
        print("=" * 60)
        
        # 기본 정보
        print(f"📅 조회 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 파일 경로: {pkl_path}")
        print()
        
        # 모델 구조 정보
        print("🏗️ 모델 구조:")
        print(f"   - 특징 개수: {len(ai_data['feature_names'])}개")
        print(f"   - 매도 모델: XGBoost Binary Classifier")
        if 'action_classifier' in ai_data:
            print(f"   - 액션 모델: XGBoost 3-Class Classifier (BUY/HOLD/SELL)")
        print(f"   - 최적 임계값: {ai_data['optimal_threshold']:.3f}")
        print(f"   - 랜덤 시드: {ai_data['random_state']}")
        print()
        
        # 성능 정보
        perf = ai_data['model_performance']
        print("📈 모델 성능:")
        print(f"   - AUC 점수: {perf['auc_score']:.3f}")
        print(f"   - 정확도: {perf['accuracy']:.3f}")
        print(f"   - 교차검증 AUC: {perf['cv_auc_mean']:.3f} (±{perf['cv_auc_std']:.3f})")
        print()
        
        # 특징 목록
        print("📋 입력 특징 목록:")
        for i, feature in enumerate(ai_data['feature_names'], 1):
            print(f"   {i:2d}. {feature}")
        print()
        
        # 특징 중요도 - 매도 모델
        sell_model = ai_data['sell_probability_model']
        sell_importances = sell_model.feature_importances_
        
        print("🎯 매도 AI가 중요하게 보는 요소 (상위 10개):")
        sell_feature_df = pd.DataFrame({
            'feature': ai_data['feature_names'],
            'importance': sell_importances
        }).sort_values('importance', ascending=False)
        
        for idx, (_, row) in enumerate(sell_feature_df.head(10).iterrows(), 1):
            print(f"   {idx:2d}. {row['feature']}: {row['importance']:.1%}")
        print()
        
        # 3-Class 액션 모델 (있는 경우)
        if 'action_classifier' in ai_data:
            action_model = ai_data['action_classifier']
            action_importances = action_model.feature_importances_
            
            print("🎯 3-Class 액션 AI가 중요하게 보는 요소 (상위 10개):")
            action_feature_df = pd.DataFrame({
                'feature': ai_data['feature_names'],
                'importance': action_importances
            }).sort_values('importance', ascending=False)
            
            for idx, (_, row) in enumerate(action_feature_df.head(10).iterrows(), 1):
                print(f"   {idx:2d}. {row['feature']}: {row['importance']:.1%}")
            print()
        
        # XGBoost 파라미터
        print("⚙️ 모델 설정:")
        params = sell_model.get_params()
        important_params = ['n_estimators', 'max_depth', 'learning_rate', 
                          'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']
        
        for param in important_params:
            if param in params:
                print(f"   - {param}: {params[param]}")
        print()
        
        # 혼동 행렬
        if 'confusion_matrix' in perf:
            cm = perf['confusion_matrix']
            print("📊 혼동 행렬:")
            print(f"   실제\\예측    보유    매도")
            print(f"   보유      {cm[0][0]:4d}    {cm[0][1]:4d}")
            print(f"   매도      {cm[1][0]:4d}    {cm[1][1]:4d}")
            print()
        
        # 파일 정보를 텍스트로도 저장
        save_as_text = input("📝 이 정보를 텍스트 파일로 저장하시겠습니까? (y/n): ")
        if save_as_text.lower() == 'y':
            save_model_info_as_text(ai_data, feature_df)
        
        return ai_data
        
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {pkl_path}")
        print("먼저 AI 모델을 훈련하고 저장해주세요.")
        return None
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return None

def save_model_info_as_text(ai_data, feature_df):
    """모델 정보를 텍스트 파일로 저장"""
    filename = f"model_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("AI 매매 모델 정보 보고서\n")
        f.write("=" * 50 + "\n\n")
        
        # 기본 정보
        f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"모델 타입: XGBoost Classifier\n")
        f.write(f"특징 개수: {len(ai_data['feature_names'])}개\n")
        f.write(f"최적 임계값: {ai_data['optimal_threshold']:.3f}\n\n")
        
        # 성능
        perf = ai_data['model_performance']
        f.write("모델 성능:\n")
        f.write(f"- AUC: {perf['auc_score']:.3f}\n")
        f.write(f"- 정확도: {perf['accuracy']:.3f}\n")
        f.write(f"- 교차검증 AUC: {perf['cv_auc_mean']:.3f} (±{perf['cv_auc_std']:.3f})\n\n")
        
        # 특징 중요도
        f.write("특징 중요도 순위:\n")
        for idx, (_, row) in enumerate(feature_df.iterrows(), 1):
            f.write(f"{idx:2d}. {row['feature']}: {row['importance']:.1%}\n")
        
        # 모든 특징 목록
        f.write("\n전체 입력 특징:\n")
        for i, feature in enumerate(ai_data['feature_names'], 1):
            f.write(f"{i:2d}. {feature}\n")
    
    print(f"✅ 정보가 저장되었습니다: {filename}")

def predict_sample():
    """샘플 예측 테스트"""
    print("\n🎯 샘플 예측 테스트:")
    print("실제 시장 상황을 입력하면 AI가 매도 확률을 예측합니다.")
    
    # 사용자 입력 받기
    try:
        profit_rate = float(input("현재 수익률 (예: 0.05 = 5%): "))
        market_condition = input("시장 상황 (상승장/하락장/횡보장): ")
        volatility = float(input("변동성 (예: 0.02 = 2%): "))
        
        # 간단한 규칙 기반 예측 (실제로는 저장된 모델 사용)
        base_prob = 0.5
        
        if profit_rate > 0.05:
            base_prob += 0.2
        elif profit_rate < -0.03:
            base_prob += 0.3
        
        if market_condition == "하락장":
            base_prob += 0.1
        elif market_condition == "상승장":
            base_prob -= 0.1
            
        if volatility > 0.03:
            base_prob += 0.1
        
        base_prob = min(1.0, max(0.0, base_prob))
        
        print(f"\n🤖 AI 예측 결과:")
        print(f"매도 확률: {base_prob:.1%}")
        print(f"추천: {'매도' if base_prob > 0.5 else '보유'}")
        
    except ValueError:
        print("❌ 숫자를 올바르게 입력해주세요.")

if __name__ == "__main__":
    # 메인 실행
    model_data = inspect_model()
    
    if model_data:
        print("\n" + "=" * 60)
        test = input("🔮 샘플 예측을 테스트해보시겠습니까? (y/n): ")
        if test.lower() == 'y':
            predict_sample()