from dataclasses import dataclass

@dataclass
class InvestorProfile:
    """투자자 프로필 데이터 클래스"""
    name: str
    # 매매 패턴
    profit_take_5_percent: bool  # 5% 수익 시 매도
    profit_take_20_percent: bool # 20% 수익 시 매도
    loss_cut_3_percent: bool     # -3% 손실 시 손절
    loss_cut_10_percent: bool    # -10% 손실 시 버티기
    volatility_panic_trade: bool # 하루 5% 급등/급락 시 패닉 매매
    time_of_day_impulse_trade: bool # 장 초반/마감 전 충동적 거래
    technical_indicator_reliance: str # 이평선, RSI, MACD 등 어떤 지표를 신뢰하는지
    chart_pattern_recognition: str # 삼각수렴, 헤드앤숄더 등 패턴 매매 성향
    volume_spike_trade: bool     # 거래량 급증 시 따라서 매매
    candle_analysis_reaction: str # 장대양봉/음봉에 대한 반응 패턴

# 5가지 투자자 프로필 정의
INVESTOR_PROFILES = [
    InvestorProfile(
        name='Conservative',
        profit_take_5_percent=True,
        profit_take_20_percent=False,
        loss_cut_3_percent=True,
        loss_cut_10_percent=False,
        volatility_panic_trade=False,
        time_of_day_impulse_trade=False,
        technical_indicator_reliance='이평선',
        chart_pattern_recognition='삼각수렴',
        volume_spike_trade=False,
        candle_analysis_reaction='장대양봉 매수'
    ),
    InvestorProfile(
        name='Aggressive',
        profit_take_5_percent=False,
        profit_take_20_percent=True,
        loss_cut_3_percent=False,
        loss_cut_10_percent=True,
        volatility_panic_trade=True,
        time_of_day_impulse_trade=True,
        technical_indicator_reliance='MACD',
        chart_pattern_recognition='헤드앤숄더',
        volume_spike_trade=True,
        candle_analysis_reaction='장대음봉 매도'
    ),
    InvestorProfile(
        name='Technical_Trader',
        profit_take_5_percent=True,
        profit_take_20_percent=False,
        loss_cut_3_percent=True,
        loss_cut_10_percent=False,
        volatility_panic_trade=False,
        time_of_day_impulse_trade=False,
        technical_indicator_reliance='RSI',
        chart_pattern_recognition='이중바닥',
        volume_spike_trade=False,
        candle_analysis_reaction='도지 캔들 관망'
    ),
    InvestorProfile(
        name='Momentum_Trader',
        profit_take_5_percent=False,
        profit_take_20_percent=True,
        loss_cut_3_percent=False,
        loss_cut_10_percent=True,
        volatility_panic_trade=True,
        time_of_day_impulse_trade=True,
        technical_indicator_reliance='거래량',
        chart_pattern_recognition='돌파 매매',
        volume_spike_trade=True,
        candle_analysis_reaction='급등주 추격 매수'
    ),
    InvestorProfile(
        name='Swing_Trader',
        profit_take_5_percent=True,
        profit_take_20_percent=False,
        loss_cut_3_percent=False,
        loss_cut_10_percent=True,
        volatility_panic_trade=False,
        time_of_day_impulse_trade=False,
        technical_indicator_reliance='이평선',
        chart_pattern_recognition='지지저항',
        volume_spike_trade=False,
        candle_analysis_reaction='눌림목 매수'
    )
]

