import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Professional style settings
plt.style.use('default')  # Safe default style
plt.rcParams['font.family'] = 'DejaVu Sans'  # Cross-platform safe font
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'  # Clean white background
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelpad'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'

# Refined color palette
COLORS = {
    'buy': '#1B73B3',       # Muted blue for buy
    'sell': '#C0392B',      # Muted red for sell
    'trade': '#E67E22',     # Muted orange for trade
    'optimal': '#27AE60',   # Vibrant green for optimal point
    'grid': '#D5DBDB',      # Subtle grid
    'text': '#2C3E50',      # Dark navy for text
    'background': 'white'  # Clean white background
}

def generate_weight_combinations(step=0.05, min_weight=0.1, max_weight=0.85):
    """Generate weight combinations for grid search."""
    combinations = []
    weights_range = np.arange(min_weight, max_weight, step)

    for w1 in weights_range:
        for w2 in weights_range:
            w3 = round(1.0 - w1 - w2, 2)
            if min_weight <= w3 <= max_weight:
                combinations.append([w1, w2, w3])

    return np.array(combinations)

def simulate_buy_signal_performance(weights):
    """Simulate buy signal performance with fundamental dominance."""
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
    """Simulate sell signal performance with timing dominance."""
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
    """Simulate trade quality performance with balanced result focus."""
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

def create_professional_3d_visualization(save_path_png='weight_optimization_3d.png',
                                         save_path_svg='weight_optimization_3d.svg'):
    """Create a professional 3D visualization for weight optimization."""
    np.random.seed(2024)

    # Generate weights and compute performance
    weights = generate_weight_combinations()
    buy_scores = np.array([simulate_buy_signal_performance(w) for w in weights])
    sell_scores = np.array([simulate_sell_signal_performance(w) for w in weights])
    trade_scores = np.array([simulate_trade_quality_performance(w) for w in weights])

    # Find optimal combinations
    best_buy_idx = np.argmax(buy_scores)
    best_sell_idx = np.argmax(sell_scores)
    best_trade_idx = np.argmax(trade_scores)

    optimal_results = {
        'buy': {'weights': weights[best_buy_idx], 'score': buy_scores[best_buy_idx]},
        'sell': {'weights': weights[best_sell_idx], 'score': sell_scores[best_sell_idx]},
        'trade': {'weights': weights[best_trade_idx], 'score': trade_scores[best_trade_idx]}
    }

    # Create figure with more spacing
    fig = plt.figure(figsize=(24, 8), dpi=120)
    fig.patch.set_facecolor('white')

    # Main title
    fig.suptitle('Weight Optimization Analysis for Trading Models',
                 fontsize=18, fontweight='bold', y=0.97, color=COLORS['text'])

    # Model configurations
    models = [
        {
            'data': (weights, buy_scores),
            'optimal': optimal_results['buy'],
            'title': 'Buy Signal Optimization',
            'subtitle': 'Technical × Fundamental × Market',
            'labels': ['Technical Analysis', 'Fundamental Analysis', 'Market Environment'],
            'color': COLORS['buy'],
            'cmap': LinearSegmentedColormap.from_list('custom_blue', ['#E6F0FA', COLORS['buy']])
        },
        {
            'data': (weights, sell_scores),
            'optimal': optimal_results['sell'],
            'title': 'Sell Signal Optimization',
            'subtitle': 'Timing × Profit × Market',
            'labels': ['Exit Timing', 'Profit Taking', 'Market Condition'],
            'color': COLORS['sell'],
            'cmap': LinearSegmentedColormap.from_list('custom_red', ['#F9E6E6', COLORS['sell']])
        },
        {
            'data': (weights, trade_scores),
            'optimal': optimal_results['trade'],
            'title': 'Trade Quality Optimization',
            'subtitle': 'Entry × Exit × Result',
            'labels': ['Entry Quality', 'Exit Timing', 'Result Quality'],
            'color': COLORS['trade'],
            'cmap': LinearSegmentedColormap.from_list('custom_orange', ['#FFF0E6', COLORS['trade']])
        }
    ]

    # Create 3D plots
    for i, model in enumerate(models):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        w, scores = model['data']
        optimal = model['optimal']

        # Scatter plot with refined styling
        scatter = ax.scatter(w[:, 0], w[:, 1], scores,
                             c=scores, cmap=model['cmap'],
                             s=30, alpha=0.7, edgecolors='none', depthshade=True)

        # Highlight optimal point
        ax.scatter(optimal['weights'][0], optimal['weights'][1], optimal['score'],
                   s=250, c=COLORS['optimal'], marker='*',
                   edgecolors=COLORS['text'], linewidth=1.5, alpha=0.95)

        # Axis labels with refined styling
        ax.set_xlabel(model['labels'][0], fontsize=10, color=COLORS['text'])
        ax.set_ylabel(model['labels'][1], fontsize=10, color=COLORS['text'])
        ax.set_zlabel('R² Performance', fontsize=10, color=COLORS['text'])

        # Title with subtitle
        ax.set_title(f"{model['title']}\n{model['subtitle']}",
                     fontsize=12, fontweight='bold', color=model['color'], pad=20)

        # Optimal weights annotation
        opt_w = optimal['weights']
        opt_text = f"Optimal Weights: {opt_w[0]:.2f}, {opt_w[1]:.2f}, {opt_w[2]:.2f}\nR² = {optimal['score']:.4f}"
        ax.text2D(0.5, -0.05, opt_text, transform=ax.transAxes,
                  fontsize=9, ha='center', va='top',
                  bbox=dict(boxstyle="round,pad=0.4", facecolor='white',
                            edgecolor=model['color'], alpha=0.9))

        # Grid and pane styling
        ax.grid(True, linestyle='--', alpha=0.4, color=COLORS['grid'])
        ax.xaxis.pane.set_edgecolor(COLORS['grid'])
        ax.yaxis.pane.set_edgecolor(COLORS['grid'])
        ax.zaxis.pane.set_edgecolor(COLORS['grid'])
        ax.xaxis.pane.set_alpha(0.2)
        ax.yaxis.pane.set_alpha(0.2)
        ax.zaxis.pane.set_alpha(0.2)

        # Axis limits
        ax.set_xlim(0.1, 0.8)
        ax.set_ylim(0.1, 0.8)
        ax.set_zlim(scores.min() * 0.95, scores.max() * 1.02)

        # Colorbar with refined styling
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=12, pad=0.12)
        cbar.set_label('Performance Score', rotation=270, labelpad=15, fontsize=9, color=COLORS['text'])
        cbar.outline.set_edgecolor(COLORS['grid'])

        # Optimize view angle for clarity
        ax.view_init(elev=25, azim=50)

    # Legend with refined styling
    legend_elements = [
        mpatches.Patch(color=COLORS['buy'], label='Buy Signal', alpha=0.8),
        mpatches.Patch(color=COLORS['sell'], label='Sell Signal', alpha=0.8),
        mpatches.Patch(color=COLORS['trade'], label='Trade Quality', alpha=0.8),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=COLORS['optimal'],
                   markersize=12, label='Optimal Point', markeredgecolor=COLORS['text'])
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, 0.01), frameon=True,
               edgecolor=COLORS['grid'], facecolor='white')

    # Footer information
    info_text = f"Grid Search: {len(weights):,} combinations | Weight Range: 0.10–0.80 | Step: 0.05"
    fig.text(0.5, 0.005, info_text, ha='center', fontsize=9,
             color=COLORS['text'], style='italic')

    # Adjust layout with more spacing between subplots
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.12, wspace=0.3)

    # Save high-quality outputs
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path_svg, format='svg', bbox_inches='tight', facecolor='white')

    # Console output with professional formatting
    print("\n\033[1m🎯 Weight Optimization Analysis Results\033[0m")
    print("═" * 60)
    print(f"\033[94m📊 Total Combinations Tested:\033[0m {len(weights):,}")
    print(f"\033[94m🔍 Search Space:\033[0m Weights 0.10–0.80 (Step: 0.05)")
    print()

    for model_name, result in optimal_results.items():
        w = result['weights']
        score = result['score']
        print(f"\033[1m🏆 {model_name.upper()} MODEL:\033[0m")
        print(f"   Optimal Weights: [{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}]")
        print(f"   R² Score: {score:.4f}")
        print(f"   Performance vs Baseline: {((score-0.5)*200):+.1f}%")
        print()

    print("\033[94m📁 Output Files:\033[0m")
    print(f"   • {save_path_png} (High-Resolution PNG)")
    print(f"   • {save_path_svg} (Vector SVG)")
    print("\n\033[92m✅ Visualization Completed Successfully!\033[0m")

    plt.show()

    return optimal_results

if __name__ == "__main__":
    print("\033[1m🚀 Launching 3D Weight Optimization Visualization\033[0m")
    print("═" * 60)

    results = create_professional_3d_visualization(
        save_path_png='/Users/inter4259/Desktop/Programming/hek_credit/weight_optimization_3d.png',
        save_path_svg='/Users/inter4259/Desktop/Programming/hek_credit/weight_optimization_3d.svg'
    )