import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# 깔끔한 스타일 설정
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 12

# 명확한 컬러 팔레트
COLORS = {
    'buy': '#2E86AB',      # 진한 파란색
    'sell': '#A23B72',     # 진한 보라색  
    'trade': '#F18F01',    # 진한 주황색
    'optimal': '#28A745',  # 초록색
    'text': '#2C3E50'      # 어두운 회색
}

def generate_weight_combinations():
    """가중치 조합 생성 (진짜 3D)"""
    combinations = []
    step = 0.05
    weights_range = np.arange(0.1, 0.85, step)
    
    for w1 in weights_range:
        for w2 in weights_range:
            w3 = round(1.0 - w1 - w2, 2)
            if 0.1 <= w3 <= 0.8:
                combinations.append([w1, w2, w3])
    
    return np.array(combinations)

def simulate_buy_signal_performance(weights):
    """매수 신호 성능 (펀더멘털 우세)"""
    tech, fund, market = weights
    fundamental_dominance = fund ** 1.6 * 0.65
    technical_support = tech * 0.25
    market_timing = market * 0.15
    imbalance_penalty = np.std([tech, fund, market]) * 0.08
    interaction_bonus = (tech * market) ** 0.5 * 0.1
    
    base_score = 0.35 + fundamental_dominance + technical_support + market_timing + interaction_bonus - imbalance_penalty
    noise = np.random.normal(0, 0.015)
    
    return np.clip(base_score + noise, 0.2, 0.95)

def simulate_sell_signal_performance(weights):
    """매도 신호 성능 (타이밍 우세)"""
    timing, profit, market = weights
    timing_dominance = timing ** 1.4 * 0.6
    profit_realization = profit ** 1.1 * 0.3
    market_condition = market * 0.2
    balance_bonus = 1.0 - np.var([timing, profit, market]) * 0.1
    synergy = (timing * profit) ** 0.7 * 0.15
    
    base_score = 0.32 + timing_dominance + profit_realization + market_condition + synergy
    final_score = base_score * balance_bonus
    noise = np.random.normal(0, 0.02)
    
    return np.clip(final_score + noise, 0.25, 0.92)

def simulate_trade_quality_performance(weights):
    """거래품질 성능 (결과 중심)"""
    entry, exit_timing, result = weights
    result_importance = result ** 1.3 * 0.5
    entry_quality = entry * 0.25
    exit_efficiency = exit_timing * 0.25
    balance_multiplier = 1.0 - np.var([entry, exit_timing, result]) * 0.15
    process_consistency = min(entry, exit_timing) * 0.2
    
    base_score = 0.38 + result_importance + entry_quality + exit_efficiency + process_consistency
    final_score = base_score * balance_multiplier
    noise = np.random.normal(0, 0.025)
    
    return np.clip(final_score + noise, 0.28, 0.94)

def create_true_3d_visualization():
    """진짜 3D 가중치 시각화 (x,y,z 모두 가중치, 색상으로 R² 표현)"""
    np.random.seed(2024)
    
    # 데이터 생성
    weights = generate_weight_combinations()
    buy_scores = np.array([simulate_buy_signal_performance(w) for w in weights])
    sell_scores = np.array([simulate_sell_signal_performance(w) for w in weights])
    trade_scores = np.array([simulate_trade_quality_performance(w) for w in weights])
    
    # 최적점 찾기
    best_buy_idx = np.argmax(buy_scores)
    best_sell_idx = np.argmax(sell_scores)
    best_trade_idx = np.argmax(trade_scores)
    
    optimal_results = {
        'buy': {'weights': weights[best_buy_idx], 'score': buy_scores[best_buy_idx]},
        'sell': {'weights': weights[best_sell_idx], 'score': sell_scores[best_sell_idx]},
        'trade': {'weights': weights[best_trade_idx], 'score': trade_scores[best_trade_idx]}
    }
    
    # 큰 그림 생성
    fig = plt.figure(figsize=(28, 10))
    fig.patch.set_facecolor('white')
    
    # 메인 제목
    fig.suptitle('True 3D Weight Optimization: All Three Components as Axes', 
                fontsize=24, fontweight='bold', y=0.95, color=COLORS['text'])
    
    # 커스텀 컬러맵 생성 (진짜 파랑, 빨강, 오렌지)
    from matplotlib.colors import LinearSegmentedColormap
    
    # 파랑 컬러맵: 연한 파랑 → 선명한 파랑
    blue_colors = ['#E6F2FF', '#0066FF', '#003399']  # 연한파랑 → 선명한파랑 → 진한파랑
    blue_cmap = LinearSegmentedColormap.from_list('custom_blue', blue_colors, N=256)
    
    # 빨강 컬러맵: 연한 빨강 → 선명한 빨강  
    red_colors = ['#FFE6E6', '#FF3333', '#CC0000']  # 연한빨강 → 선명한빨강 → 진한빨강
    red_cmap = LinearSegmentedColormap.from_list('custom_red', red_colors, N=256)
    
    # 오렌지 컬러맵: 연한 오렌지 → 선명한 오렌지
    orange_colors = ['#FFF0E6', '#FF6600', '#CC3300']  # 연한오렌지 → 선명한오렌지 → 진한오렌지  
    orange_cmap = LinearSegmentedColormap.from_list('custom_orange', orange_colors, N=256)
    
    # 모델 설정 (커스텀 컬러맵 적용)
    models = [
        {
            'data': (weights, buy_scores),
            'optimal': optimal_results['buy'],
            'title': 'Buy Signal Model',
            'components': ['Technical', 'Fundamental', 'Market'],
            'color': COLORS['buy'],
            'cmap': blue_cmap
        },
        {
            'data': (weights, sell_scores), 
            'optimal': optimal_results['sell'],
            'title': 'Sell Signal Model',
            'components': ['Timing', 'Profit', 'Market'],
            'color': COLORS['sell'],
            'cmap': red_cmap
        },
        {
            'data': (weights, trade_scores),
            'optimal': optimal_results['trade'], 
            'title': 'Trade Quality Model',
            'components': ['Entry', 'Exit Timing', 'Result'],
            'color': COLORS['trade'],
            'cmap': orange_cmap
        }
    ]
    
    # 각 서브플롯 생성
    for i, model in enumerate(models):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        w, scores = model['data']
        optimal = model['optimal']
        
        # 진짜 3D 산점도: 3D 효과 강화
        # z축 위치에 따라 점 크기와 투명도 조절 (원근감)
        z_normalized = (w[:, 2] - w[:, 2].min()) / (w[:, 2].max() - w[:, 2].min())
        point_sizes = 60 + 40 * z_normalized  # 앞쪽 점들이 더 크게
        point_alphas = 0.6 + 0.3 * z_normalized  # 앞쪽 점들이 더 불투명
        
        scatter = ax.scatter(w[:, 0], w[:, 1], w[:, 2],  # 이제 z도 가중치!
                           c=scores, cmap=model['cmap'], vmin=scores.min() * 0.95, vmax=scores.max() * 1.02,
                           s=point_sizes, alpha=point_alphas, edgecolors='black', linewidth=0.1,
                           depthshade=True)  # 3D 그림자 효과
        
        # 최적점 강조 (노란 빈 원)
        opt_w = optimal['weights']
        ax.scatter(opt_w[0], opt_w[1], opt_w[2],  # z축도 가중치
                  s=600, facecolors='none', marker='o', 
                  edgecolors='gold', linewidth=4, alpha=0.95)
        
        # 진짜 3D 축 라벨
        ax.set_xlabel(f'{model["components"][0]} Weight', fontsize=12, color=COLORS['text'], labelpad=15)
        ax.set_ylabel(f'{model["components"][1]} Weight', fontsize=12, color=COLORS['text'], labelpad=15)
        ax.set_zlabel(f'{model["components"][2]} Weight', fontsize=12, color=COLORS['text'], labelpad=15)  # 이제 z도 가중치!
        
        # 제목
        ax.set_title(model['title'], fontsize=16, fontweight='bold', color=model['color'], pad=25)
        
        # 최적 결과
        opt_text = f"Best: {model['components'][0]}={opt_w[0]:.2f}, {model['components'][1]}={opt_w[1]:.2f}, {model['components'][2]}={opt_w[2]:.2f} | R² = {optimal['score']:.4f}"
        ax.text2D(0.5, -0.15, opt_text, transform=ax.transAxes,
                  fontsize=10, ha='center', va='top', fontweight='bold',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                           edgecolor=model['color'], linewidth=2, alpha=0.95))
        
        # 격자와 배경 설정
        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
        # 축 범위 (모든 축이 가중치이므로 동일한 범위)
        ax.set_xlim(0.1, 0.8)
        ax.set_ylim(0.1, 0.8)
        ax.set_zlim(0.1, 0.8)  # z축도 가중치 범위
        
        # 컬러바 (이제 R² Score를 나타냄)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=15, pad=0.15)
        cbar.set_label('R² Performance Score', rotation=270, labelpad=20, fontsize=11)
        
        # 더 극적인 3D 뷰 각도
        ax.view_init(elev=20, azim=60)  # 더 낮은 각도로 입체감 강화
    
    # 범례
    legend_elements = [
        mpatches.Patch(color=COLORS['buy'], label='Buy Signal'),
        mpatches.Patch(color=COLORS['sell'], label='Sell Signal'),  
        mpatches.Patch(color=COLORS['trade'], label='Trade Quality'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markersize=15, label='Optimal Point', markeredgecolor='gold', markeredgewidth=3)
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
              fontsize=13, bbox_to_anchor=(0.5, 0.02), frameon=True,
              edgecolor='gray', facecolor='white')
    
    # 하단 정보
    info_text = f"True 3D Grid Search: {len(weights)} combinations | All weights: 0.10-0.80 | Constraint: W1 + W2 + W3 = 1.0"
    fig.text(0.5, -0.02, info_text, ha='center', fontsize=11, 
            color=COLORS['text'], style='italic')
    
    # 레이아웃 조정
    plt.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.15, wspace=0.4)
    
    # 저장
    plt.savefig('/Users/inter4259/Desktop/Programming/hek_credit/true_3d_weight_optimization.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.savefig('/Users/inter4259/Desktop/Programming/hek_credit/true_3d_weight_optimization.svg', 
                format='svg', bbox_inches='tight', facecolor='white')
    
    # 결과 출력
    print("🎯 True 3D Weight Optimization Results")
    print("=" * 50)
    print("🔥 NOW ALL THREE AXES REPRESENT WEIGHTS!")
    print("🌈 R² Performance shown as COLOR")
    print(f"📊 Total combinations: {len(weights)}")
    print(f"🔍 Weight constraint: W1 + W2 + W3 = 1.0")
    print()
    
    for model_name, result in optimal_results.items():
        w = result['weights']
        score = result['score']
        print(f"🏆 {model_name.upper()} MODEL:")
        print(f"   Weights: [{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}] (sum={sum(w):.2f})")
        print(f"   R² Score: {score:.4f}")
        print(f"   Performance: +{((score-0.5)*200):.1f}%")
        print()
    
    print("📁 Saved:")
    print("   • true_3d_weight_optimization.png")
    print("   • true_3d_weight_optimization.svg")
    
    plt.show()
    return optimal_results

if __name__ == "__main__":
    print("🚀 Creating TRUE 3D Weight Visualization")
    print("=" * 50)
    print("📍 X-axis: Component 1 Weight")  
    print("📍 Y-axis: Component 2 Weight")
    print("📍 Z-axis: Component 3 Weight")
    print("🎨 Color: R² Performance Score")
    print("=" * 50)
    
    results = create_true_3d_visualization()
    
    print("\\n✅ True 3D visualization complete!")