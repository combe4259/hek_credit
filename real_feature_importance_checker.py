#!/usr/bin/env python3
"""
실제 학습된 PKL 모델에서 피처 중요도 추출
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 스타일 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'

# KB 컬러 팔레트
KB_COLORS = {
    'primary': '#FFB800',
    'secondary': '#1E3A8A', 
    'accent': '#059669',
    'danger': '#DC2626',
    'text': '#1F2937'
}

def find_pkl_models():
    """PKL 모델 파일들 찾기"""
    base_path = Path('/Users/inter4259/Desktop/Programming/hek_credit')
    pkl_files = []
    
    # generate_data/models에서 찾기
    models_path = base_path / 'generate_data' / 'models'
    if models_path.exists():
        pkl_files.extend(list(models_path.glob('*.pkl')))
    
    # trading_ai에서 찾기
    trading_ai_path = base_path / 'trading_ai'
    if trading_ai_path.exists():
        pkl_files.extend(list(trading_ai_path.glob('*.pkl')))
    
    return pkl_files

def load_and_analyze_model(pkl_path):
    """PKL 모델 로드하고 피처 중요도 분석"""
    try:
        print(f"📂 로딩 중: {pkl_path.name}")
        
        # 모델 로드
        model_data = joblib.load(pkl_path)
        
        # 모델 구조 확인
        if isinstance(model_data, dict):
            # 딕셔너리 형태의 경우
            if 'model' in model_data:
                model = model_data['model']
                feature_names = model_data.get('feature_names', model_data.get('features', None))
            else:
                print(f"   ❌ 딕셔너리에서 'model' 키를 찾을 수 없음")
                return None
        else:
            # 모델 객체 직접 저장된 경우
            model = model_data
            feature_names = None
        
        # 피처 중요도 확인
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            # 상위 5개 추출
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            top5 = importance_df.head(5)
            
            print(f"   ✅ 피처 중요도 추출 성공 ({len(importances)}개 피처)")
            print(f"   🏆 Top 5:")
            for i, (_, row) in enumerate(top5.iterrows()):
                pct = row['importance'] / importances.sum() * 100
                print(f"      #{i+1} {row['feature']}: {row['importance']:.4f} ({pct:.1f}%)")
            
            return {
                'model_name': pkl_path.stem,
                'top5': top5,
                'all_importance': importance_df,
                'total_features': len(importances)
            }
        else:
            print(f"   ❌ 모델에 feature_importances_ 속성이 없음")
            return None
            
    except Exception as e:
        print(f"   ❌ 에러: {str(e)}")
        return None

def visualize_real_feature_importance(results):
    """실제 모델들의 피처 중요도 시각화"""
    if not results:
        print("❌ 시각화할 결과가 없습니다.")
        return
    
    n_models = len(results)
    if n_models == 0:
        return
    
    # 그림 크기 조정
    if n_models <= 3:
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 8))
        if n_models == 1:
            axes = [axes]
    else:
        rows = (n_models + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(24, 8*rows))
        axes = axes.flatten()
    
    fig.patch.set_facecolor('white')
    
    colors = [KB_COLORS['secondary'], KB_COLORS['danger'], KB_COLORS['accent'], 
              KB_COLORS['primary'], '#8B5CF6', '#F59E0B']
    
    for idx, result in enumerate(results):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        top5 = result['top5']
        
        # 데이터 준비
        features = top5['feature'].tolist()
        importances = top5['importance'].tolist()
        
        # 피처명 길이 제한
        short_features = []
        for f in features:
            if len(f) > 25:
                short_features.append(f[:22] + '...')
            else:
                short_features.append(f)
        
        # 수평 막대 그래프
        y_pos = np.arange(len(short_features))
        color = colors[idx % len(colors)]
        bars = ax.barh(y_pos, importances, color=color, alpha=0.8, edgecolor='white', linewidth=2)
        
        # 값 표시
        for i, (bar, imp) in enumerate(zip(bars, importances)):
            width = bar.get_width()
            total_imp = sum(importances)
            pct = imp / result['all_importance']['importance'].sum() * 100
            ax.text(width + max(importances)*0.01, bar.get_y() + bar.get_height()/2, 
                   f'{imp:.3f}\n({pct:.1f}%)', 
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        # 스타일링
        ax.set_title(f'{result["model_name"]}\n({result["total_features"]} features)', 
                    fontsize=14, fontweight='bold', color=color, pad=20)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_features, fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_facecolor('white')
        
        # 순위 표시
        for i, y in enumerate(y_pos):
            ax.text(-max(importances)*0.05, y, f'#{i+1}', 
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle="circle,pad=0.3", facecolor=color, alpha=0.3))
    
    # 빈 서브플롯 숨기기
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('🔍 Real Model Feature Importance Analysis', 
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 저장
    save_path = '/Users/inter4259/Desktop/Programming/hek_credit/real_feature_importance.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\n✅ 시각화 저장: {save_path}")
    plt.show()

def main():
    """메인 함수"""
    print("🔍 실제 PKL 모델에서 피처 중요도 분석")
    print("="*60)
    
    # PKL 파일들 찾기
    pkl_files = find_pkl_models()
    
    if not pkl_files:
        print("❌ PKL 파일을 찾을 수 없습니다.")
        return
    
    print(f"📂 발견된 PKL 파일: {len(pkl_files)}개")
    for pkl_file in pkl_files:
        print(f"   • {pkl_file}")
    
    print("\n" + "="*60)
    
    # 각 모델 분석
    results = []
    for pkl_file in pkl_files:
        result = load_and_analyze_model(pkl_file)
        if result:
            results.append(result)
    
    if not results:
        print("\n❌ 분석 가능한 모델이 없습니다.")
        return
    
    print(f"\n✅ 총 {len(results)}개 모델에서 피처 중요도 추출 완료")
    
    # 시각화
    print("\n📊 시각화 생성 중...")
    visualize_real_feature_importance(results)
    
    # 요약 출력
    print("\n" + "="*60)
    print("📋 요약:")
    for result in results:
        print(f"\n🏆 {result['model_name']}:")
        top1 = result['top5'].iloc[0]
        pct = top1['importance'] / result['all_importance']['importance'].sum() * 100
        print(f"   최고 중요 피처: {top1['feature']} ({pct:.1f}%)")
    print("="*60)

if __name__ == "__main__":
    main()