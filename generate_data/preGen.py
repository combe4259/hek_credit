import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path

class TradingDataPreprocessor:
    """거래 데이터를 AI 학습용 데이터로 전처리"""

    def __init__(self, output_dir='preprocessed_data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def preprocess_all(self, transactions_path: str, customer_info_path: str = None):
        """전체 전처리 파이프라인 실행"""
        print("📊 전처리 시작...")

        # 1. 원본 데이터 로드
        transactions_df = pd.read_csv(transactions_path)
        print(f"✅ 거래 데이터 로드: {len(transactions_df)}건")

        # 2. 에피소드 생성
        episodes_df = self.create_episodes(transactions_df)
        print(f"✅ 에피소드 생성: {len(episodes_df)}개")

        # 3. 특징 추출
        features_df = self.extract_features(episodes_df, transactions_df)
        print(f"✅ 특징 추출 완료: {features_df.shape[1]}개 특징")

        # 4. 상황 벡터 생성
        situations_df = self.create_situation_vectors(episodes_df, features_df)
        print(f"✅ 상황 벡터 생성: {situations_df.shape}")

        # 5. 파일 저장
        self.save_all_data(episodes_df, features_df, situations_df)
        print(f"✅ 모든 파일 저장 완료: {self.output_dir}")

        return episodes_df, features_df, situations_df

    def create_episodes(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Buy-Sell 페어를 에피소드로 변환"""
        episodes = []

        for customer_id in transactions_df['customerID'].unique():
            customer_data = transactions_df[
                transactions_df['customerID'] == customer_id
                ].sort_values('timestamp')

            # Buy 거래 찾기
            buy_mask = customer_data['transactionType'] == 'Buy'
            buy_trades = customer_data[buy_mask]

            for buy_idx, buy_row in buy_trades.iterrows():
                # 매칭되는 Sell 찾기
                sell_trades = customer_data[
                    (customer_data['transactionType'] == 'Sell') &
                    (customer_data['ISIN'] == buy_row['ISIN']) &
                    (pd.to_datetime(customer_data['timestamp']) > pd.to_datetime(buy_row['timestamp']))
                    ]

                if not sell_trades.empty:
                    sell_row = sell_trades.iloc[0]

                    # 수익률 계산
                    buy_price = buy_row['totalValue'] / buy_row['units']
                    sell_price = sell_row['totalValue'] / sell_row['units']
                    return_rate = (sell_price - buy_price) / buy_price * 100

                    # 보유 기간
                    holding_days = (pd.to_datetime(sell_row['timestamp']) -
                                    pd.to_datetime(buy_row['timestamp'])).days

                    episode = {
                        'episode_id': f"{customer_id}_{buy_row['transactionID']}_{sell_row['transactionID']}",
                        'customer_id': customer_id,
                        'isin': buy_row['ISIN'],
                        'buy_transaction_id': buy_row['transactionID'],
                        'sell_transaction_id': sell_row['transactionID'],
                        'buy_timestamp': buy_row['timestamp'],
                        'sell_timestamp': sell_row['timestamp'],
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'units': buy_row['units'],
                        'return_rate': return_rate,
                        'return_amount': sell_row['totalValue'] - buy_row['totalValue'],
                        'holding_days': holding_days,
                        'channel': buy_row['channel']
                    }

                    episodes.append(episode)

        return pd.DataFrame(episodes)

    def extract_features(self, episodes_df: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """에피소드별 특징 추출"""
        features_list = []

        for _, episode in episodes_df.iterrows():
            customer_id = episode['customer_id']
            buy_timestamp = pd.to_datetime(episode['buy_timestamp'])
            sell_timestamp = pd.to_datetime(episode['sell_timestamp'])

            # 해당 고객의 모든 거래
            customer_trades = transactions_df[
                transactions_df['customerID'] == customer_id
                ].sort_values('timestamp')

            # 이전 거래들
            prev_trades = customer_trades[
                pd.to_datetime(customer_trades['timestamp']) < buy_timestamp
                ]

            # 이전 에피소드들
            prev_episodes = episodes_df[
                (episodes_df['customer_id'] == customer_id) &
                (pd.to_datetime(episodes_df['buy_timestamp']) < buy_timestamp)
                ]

            features = {
                'episode_id': episode['episode_id'],

                # === 매수 시점 특징 ===
                # 최근 성과
                'buy_recent_avg_return': self._calc_recent_avg_return(prev_episodes, 5),
                'buy_recent_win_rate': self._calc_recent_win_rate(prev_episodes, 5),
                'buy_consecutive_wins': self._calc_consecutive_wins(prev_episodes),
                'buy_consecutive_losses': self._calc_consecutive_losses(prev_episodes),

                # 거래 패턴
                'buy_days_since_last_trade': self._calc_days_since_last_trade(
                    buy_timestamp, prev_trades
                ),
                'buy_trading_frequency_30d': self._calc_trading_frequency(
                    buy_timestamp, prev_trades, 30
                ),

                # 포트폴리오 상태
                'buy_active_positions': self._count_active_positions(
                    buy_timestamp, customer_trades
                ),

                # === 매도 시점 특징 ===
                # 현재 거래 상태
                'sell_current_return': episode['return_rate'],
                'sell_holding_days': episode['holding_days'],
                'sell_return_per_day': episode['return_rate'] / max(episode['holding_days'], 1),

                # 상대적 특징
                'sell_holding_vs_avg': self._calc_holding_vs_avg(
                    episode['holding_days'], prev_episodes
                ),
                'sell_return_vs_avg': self._calc_return_vs_avg(
                    episode['return_rate'], prev_episodes
                ),

                # 최고/최저점 대비 (실제로는 일별 가격 데이터 필요)
                'sell_drawdown_pct': 0,  # TODO: 일별 가격 데이터로 계산
                'sell_runup_pct': 0,     # TODO: 일별 가격 데이터로 계산

                # === 결과 (타겟) ===
                'outcome_return_rate': episode['return_rate'],
                'outcome_holding_days': episode['holding_days'],
                'outcome_profitable': 1 if episode['return_rate'] > 0 else 0,
            }

            features_list.append(features)

        return pd.DataFrame(features_list)

    def _calc_recent_avg_return(self, prev_episodes, n=5):
        """최근 n개 거래의 평균 수익률"""
        if len(prev_episodes) == 0:
            return 0
        recent = prev_episodes.tail(n)
        return recent['return_rate'].mean() if len(recent) > 0 else 0

    def _calc_recent_win_rate(self, prev_episodes, n=5):
        """최근 n개 거래의 승률"""
        if len(prev_episodes) == 0:
            return 0.5
        recent = prev_episodes.tail(n)
        if len(recent) == 0:
            return 0.5
        return (recent['return_rate'] > 0).sum() / len(recent)

    def _calc_consecutive_wins(self, prev_episodes):
        """연속 수익 횟수"""
        if len(prev_episodes) == 0:
            return 0

        count = 0
        for _, episode in prev_episodes.iloc[::-1].iterrows():
            if episode['return_rate'] > 0:
                count += 1
            else:
                break
        return count

    def _calc_consecutive_losses(self, prev_episodes):
        """연속 손실 횟수"""
        if len(prev_episodes) == 0:
            return 0

        count = 0
        for _, episode in prev_episodes.iloc[::-1].iterrows():
            if episode['return_rate'] <= 0:
                count += 1
            else:
                break
        return count

    def _calc_days_since_last_trade(self, current_timestamp, prev_trades):
        """마지막 거래 이후 일수"""
        if len(prev_trades) == 0:
            return 30  # 기본값

        last_timestamp = pd.to_datetime(prev_trades.iloc[-1]['timestamp'])
        return (current_timestamp - last_timestamp).days

    def _calc_trading_frequency(self, current_timestamp, prev_trades, days):
        """최근 n일간 거래 빈도"""
        cutoff_date = current_timestamp - pd.Timedelta(days=days)
        recent_trades = prev_trades[
            pd.to_datetime(prev_trades['timestamp']) >= cutoff_date
            ]
        return len(recent_trades)

    def _count_active_positions(self, timestamp, customer_trades):
        """해당 시점의 활성 포지션 수"""
        # 간단한 구현 (실제로는 더 정교하게)
        buys_before = customer_trades[
            (customer_trades['transactionType'] == 'Buy') &
            (pd.to_datetime(customer_trades['timestamp']) < timestamp)
            ]
        sells_before = customer_trades[
            (customer_trades['transactionType'] == 'Sell') &
            (pd.to_datetime(customer_trades['timestamp']) < timestamp)
            ]

        # ISIN별로 계산해야 하지만, 여기서는 단순화
        return max(0, len(buys_before) - len(sells_before))

    def _calc_holding_vs_avg(self, holding_days, prev_episodes):
        """평균 보유 기간 대비 비율"""
        if len(prev_episodes) == 0:
            return 1.0
        avg_holding = prev_episodes['holding_days'].mean()
        return holding_days / max(avg_holding, 1)

    def _calc_return_vs_avg(self, return_rate, prev_episodes):
        """평균 수익률 대비 차이"""
        if len(prev_episodes) == 0:
            return 0
        avg_return = prev_episodes['return_rate'].mean()
        return return_rate - avg_return

    def create_situation_vectors(self, episodes_df: pd.DataFrame,
                                 features_df: pd.DataFrame) -> pd.DataFrame:
        """매수/매도 시점별 상황 벡터 생성"""
        buy_situations = []
        sell_situations = []

        # 매수 시점 특징
        buy_features = [col for col in features_df.columns if col.startswith('buy_')]
        sell_features = [col for col in features_df.columns if col.startswith('sell_')]

        for _, row in features_df.iterrows():
            episode = episodes_df[episodes_df['episode_id'] == row['episode_id']].iloc[0]

            # 매수 시점 벡터
            buy_situation = {
                'situation_id': f"buy_{row['episode_id']}",
                'episode_id': row['episode_id'],
                'timestamp': episode['buy_timestamp'],
                'situation_type': 'buy',
                'customer_id': episode['customer_id'],
                'isin': episode['isin']
            }
            for feat in buy_features:
                buy_situation[f'feature_{feat}'] = row[feat]
            buy_situations.append(buy_situation)

            # 매도 시점 벡터
            sell_situation = {
                'situation_id': f"sell_{row['episode_id']}",
                'episode_id': row['episode_id'],
                'timestamp': episode['sell_timestamp'],
                'situation_type': 'sell',
                'customer_id': episode['customer_id'],
                'isin': episode['isin']
            }
            for feat in sell_features:
                sell_situation[f'feature_{feat}'] = row[feat]
            # 매도 시점에는 매수 시점 정보도 포함
            for feat in buy_features:
                sell_situation[f'feature_{feat}'] = row[feat]
            sell_situations.append(sell_situation)

        # 합치기
        all_situations = buy_situations + sell_situations
        return pd.DataFrame(all_situations)

    def save_all_data(self, episodes_df, features_df, situations_df):
        """모든 전처리된 데이터 저장"""
        # CSV 저장
        episodes_df.to_csv(self.output_dir / 'episodes.csv', index=False)
        features_df.to_csv(self.output_dir / 'features.csv', index=False)
        situations_df.to_csv(self.output_dir / 'situations.csv', index=False)

        # 메타데이터 저장
        metadata = {
            'processed_date': datetime.now().isoformat(),
            'num_episodes': len(episodes_df),
            'num_customers': episodes_df['customer_id'].nunique(),
            'num_features': len([col for col in features_df.columns
                                 if col not in ['episode_id', 'outcome_return_rate',
                                                'outcome_holding_days', 'outcome_profitable']]),
            'feature_names': {
                'buy_features': [col for col in features_df.columns if col.startswith('buy_')],
                'sell_features': [col for col in features_df.columns if col.startswith('sell_')],
                'outcome_features': [col for col in features_df.columns if col.startswith('outcome_')]
            }
        }

        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # 요약 통계 저장
        summary = {
            'episodes_summary': episodes_df.describe().to_dict(),
            'features_summary': features_df.describe().to_dict(),
            'return_distribution': {
                'mean': float(episodes_df['return_rate'].mean()),
                'std': float(episodes_df['return_rate'].std()),
                'min': float(episodes_df['return_rate'].min()),
                'max': float(episodes_df['return_rate'].max()),
                'percentiles': {
                    '25%': float(episodes_df['return_rate'].quantile(0.25)),
                    '50%': float(episodes_df['return_rate'].quantile(0.50)),
                    '75%': float(episodes_df['return_rate'].quantile(0.75))
                }
            }
        }

        with open(self.output_dir / 'summary_stats.json', 'w') as f:
            json.dump(summary, f, indent=2)


# 사용 예시
if __name__ == "__main__":
    # 전처리 실행
    preprocessor = TradingDataPreprocessor(output_dir='preprocessed_data')

    # 데이터 전처리
    episodes_df, features_df, situations_df = preprocessor.preprocess_all(
        '/Users/inter4259/Downloads/FAR-Trans/transactions.csv',
        '/Users/inter4259/Downloads/FAR-Trans/customer_information.csv'
    )

    print("\n📊 전처리 결과:")
    print(f"- 에피소드: {len(episodes_df)}개")
    print(f"- 특징: {features_df.shape[1]}개")
    print(f"- 상황 벡터: {len(situations_df)}개")
    print(f"\n💾 저장된 파일:")
    print(f"- preprocessed_data/episodes.csv")
    print(f"- preprocessed_data/features.csv")
    print(f"- preprocessed_data/situations.csv")
    print(f"- preprocessed_data/metadata.json")
    print(f"- preprocessed_data/summary_stats.json")