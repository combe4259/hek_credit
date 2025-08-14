/**
 * AI Trading Service
 * AI 모델과 통신하는 서비스
 */

const API_BASE_URL = 'http://localhost:8000/api/ai/realtime';

class AITradingService {
  constructor() {
    this.initialized = false;
  }

  /**
   * AI 모델 초기화
   */
  async initialize() {
    try {
      const response = await fetch(`${API_BASE_URL}/initialize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to initialize AI models');
      }

      const data = await response.json();
      this.initialized = data.status === 'success';
      return data;
    } catch (error) {
      console.error('AI 모델 초기화 실패:', error);
      throw error;
    }
  }

  /**
   * 매수 신호 분석
   * @param {string} ticker - 종목 코드
   * @param {number} positionSizePct - 포트폴리오 대비 비중 (%)
   */
  async analyzeBuySignal(ticker, positionSizePct = 5.0) {
    try {
      // 자동 초기화
      if (!this.initialized) {
        await this.initialize();
      }

      const response = await fetch(`${API_BASE_URL}/buy-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ticker: ticker,
          position_size_pct: positionSizePct,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Buy analysis failed');
      }

      const data = await response.json();
      
      // 응답 데이터 가공
      return {
        ticker: data.ticker,
        currentPrice: data.current_price,
        recommendation: data.analysis.recommendation,
        signalScore: data.analysis.signal_score,
        confidence: data.analysis.confidence,
        shouldBuy: data.analysis.recommendation === 'BUY',
        
        // 기술적 지표 (백엔드에서 "11.15%" 형태로 오므로 파싱)
        technicalIndicators: {
          momentum5d: parseFloat(data.technical_indicators.momentum_5d.replace('%', '')),
          momentum20d: parseFloat(data.technical_indicators.momentum_20d.replace('%', '')),
          momentum60d: parseFloat(data.technical_indicators.momentum_60d.replace('%', '')),
          volatility: parseFloat(data.technical_indicators.volatility.replace('%', '')),
        },
        
        // 펀더멘털 지표
        fundamentals: {
          peRatio: data.market_data.pe_ratio,
          pbRatio: data.market_data.pb_ratio,
          roe: data.market_data.roe,
          ratio52wHigh: data.market_data['52w_high_ratio'],
          vix: data.market_data.vix,
        },
        
        // 매수 추천 정보
        buyRecommendation: data.buy_recommendation || null,
        
        timestamp: data.timestamp,
      };
    } catch (error) {
      console.error('매수 분석 실패:', error);
      throw error;
    }
  }

  /**
   * 매도 신호 분석
   * @param {string} ticker - 종목 코드
   * @param {number} entryPrice - 매수 가격
   * @param {string} entryDate - 매수 날짜 (YYYY-MM-DD)
   * @param {number} positionSize - 보유 주식 수
   */
  async analyzeSellSignal(ticker, entryPrice, entryDate, positionSize = 100) {
    try {
      // 자동 초기화
      if (!this.initialized) {
        await this.initialize();
      }

      const response = await fetch(`${API_BASE_URL}/sell-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ticker: ticker,
          entry_price: entryPrice,
          entry_date: entryDate,
          position_size: positionSize,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Sell analysis failed');
      }

      const data = await response.json();
      
      // 응답 데이터 가공
      return {
        ticker: data.ticker,
        entryPrice: data.entry_price,
        currentPrice: data.current_price,
        holdingDays: data.holding_days,
        currentReturn: data.current_return,
        
        recommendation: data.analysis.recommendation,
        signalScore: data.analysis.signal_score,
        confidence: data.analysis.confidence,
        shouldSell: data.analysis.recommendation === 'SELL',
        
        // 성과 지표
        performance: {
          totalReturn: data.performance.total_return,
          marketReturn: data.performance.market_return,
          excessReturn: data.performance.excess_return,
          profitStatus: data.performance.profit_status,
        },
        
        // 리스크 지표
        riskIndicators: {
          currentVolatility: data.risk_indicators.current_volatility,
          vixLevel: data.risk_indicators.vix_level,
          momentumChange: data.risk_indicators.momentum_change,
        },
        
        // 매도 추천 정보
        sellRecommendation: data.sell_recommendation || null,
        
        timestamp: data.timestamp,
      };
    } catch (error) {
      console.error('매도 분석 실패:', error);
      throw error;
    }
  }

  /**
   * 빠른 가격 체크
   * @param {string} ticker - 종목 코드
   */
  async quickPriceCheck(ticker) {
    try {
      const response = await fetch(`${API_BASE_URL}/quick-check/${ticker}`);
      
      if (!response.ok) {
        throw new Error('Price check failed');
      }

      return await response.json();
    } catch (error) {
      console.error('가격 체크 실패:', error);
      throw error;
    }
  }

  /**
   * 모델 상태 확인
   */
  async getModelStatus() {
    try {
      const response = await fetch(`${API_BASE_URL}/model-status`);
      
      if (!response.ok) {
        throw new Error('Status check failed');
      }

      return await response.json();
    } catch (error) {
      console.error('상태 확인 실패:', error);
      throw error;
    }
  }

  /**
   * 주식 티커 매핑 (한글명 -> 심볼)
   * LiveQuotesList에서 사용하는 심볼과 일치
   */
  getTickerSymbol(stockName) {
    // 이미 심볼 형태면 그대로 반환
    if (stockName && stockName.length <= 5 && stockName === stockName.toUpperCase()) {
      return stockName;
    }
    
    const tickerMap = {
      'Apple': 'AAPL',
      'Microsoft': 'MSFT',
      'NVIDIA': 'NVDA',
      'Tesla': 'TSLA',
      'Amazon': 'AMZN',
      'Google': 'GOOGL',
      'Alphabet': 'GOOGL',
      'Meta': 'META',
      'Meta Platforms': 'META',
      'Netflix': 'NFLX',
      'Disney': 'DIS',
      'Walt Disney': 'DIS',
      'Intel': 'INTC',
      'AMD': 'AMD',
      'Berkshire Hathaway': 'BRK.B',
      'JPMorgan Chase': 'JPM',
      'Visa': 'V',
      'Mastercard': 'MA',
      'PayPal': 'PYPL',
      'Bank of America': 'BAC',
      'Adobe': 'ADBE',
      'Salesforce': 'CRM',
      'Coca-Cola': 'KO',
      'Pfizer': 'PFE',
      'Walmart': 'WMT',
      'Qualcomm': 'QCOM',
      'Oracle': 'ORCL',
      'Cisco': 'CSCO',
      'Broadcom': 'AVGO',
      'Costco': 'COST'
    };
    
    return tickerMap[stockName] || stockName;
  }
}

// 싱글톤 인스턴스 생성
const aiTradingService = new AITradingService();

export default aiTradingService;