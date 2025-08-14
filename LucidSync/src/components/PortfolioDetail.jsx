import React, { useState, useEffect } from 'react';
import { portfolioService } from '../services/portfolioService';
import aiTradingService from '../services/aiTradingService';
import LiveQuotesList from './LiveQuotesList';
import './TabContent.css';

const PortfolioDetail = ({ portfolio, user, onBack }) => {
  const [holdings, setHoldings] = useState([]);
  const [transactions, setTransactions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('holdings'); // 'holdings', 'transactions', 'trading'
  const [selectedHolding, setSelectedHolding] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [sellAnalysis, setSellAnalysis] = useState(null);
  const [selectedStock, setSelectedStock] = useState('Apple');
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [currentStockData, setCurrentStockData] = useState(null);
  const [buyAnalysis, setBuyAnalysis] = useState(null);
  const [aiInitialized, setAiInitialized] = useState(false);
  const [realtimeAnalysis, setRealtimeAnalysis] = useState(null);
  const [holdingSellAnalysis, setHoldingSellAnalysis] = useState({}); // 각 보유 종목의 매도 분석 결과
  const [tradeEvaluations, setTradeEvaluations] = useState({}); // 거래 품질 평가 결과
  const [tradeFeedbacks, setTradeFeedbacks] = useState({}); // 거래 피드백 결과
  const [feedbackLoading, setFeedbackLoading] = useState({}); // 피드백 로딩 상태

  // AI 모델 초기화
  useEffect(() => {
    const initializeAI = async () => {
      try {
        console.log('AI 모델 초기화 시작...');
        await aiTradingService.initialize();
        setAiInitialized(true);
        console.log('AI 모델 초기화 완료');
      } catch (err) {
        console.error('AI 초기화 실패:', err);
      }
    };
    
    initializeAI();
  }, []);

  // 실시간 AI 분석 주기적 실행 (5초마다) - 주식 매수 탭에서만
  useEffect(() => {
    if (!aiInitialized || !selectedSymbol || activeTab !== 'trading') return;
    
    // 초기 분석 실행
    triggerRealtimeAIAnalysis(selectedSymbol);
    
    // 5초마다 자동 분석
    const intervalId = setInterval(() => {
      if (currentStockData && currentStockData.price) {
        console.log(`⏱️ 포트폴리오 실시간 AI 분석: ${selectedSymbol}`);
        triggerRealtimeAIAnalysis(selectedSymbol);
      }
    }, 5000); // 5초마다 실행
    
    return () => clearInterval(intervalId);
  }, [aiInitialized, selectedSymbol, activeTab, currentStockData]);

  // 보유 종목 실시간 매도 분석 (10초마다) - 보유 종목 탭에서만
  useEffect(() => {
    if (!aiInitialized || activeTab !== 'holdings' || holdings.length === 0) return;
    
    // 모든 보유 종목에 대해 초기 분석 실행
    holdings.forEach(holding => {
      triggerHoldingSellAnalysis(holding);
    });
    
    // 10초마다 모든 보유 종목 매도 분석
    const intervalId = setInterval(() => {
      console.log(`⏱️ 보유 종목 실시간 매도 분석 시작 (${holdings.length}개 종목)`);
      holdings.forEach(holding => {
        triggerHoldingSellAnalysis(holding);
      });
    }, 10000); // 10초마다 실행
    
    return () => clearInterval(intervalId);
  }, [aiInitialized, activeTab, holdings]);

  // 포트폴리오 데이터 로드
  useEffect(() => {
    loadPortfolioData();
  }, [portfolio?.id]);

  const loadPortfolioData = async () => {
    if (!portfolio?.id) return;
    
    setIsLoading(true);
    try {
      // 보유 종목 가져오기
      const holdingsData = await portfolioService.getHoldings(portfolio.id);
      setHoldings(holdingsData || []);
      
      // 거래 내역 가져오기
      const transactionsData = await portfolioService.getTransactions(portfolio.id);
      setTransactions(transactionsData || []);
      
      // 완료된 거래에 대해 AI 품질 평가 실행
      if (transactionsData && transactionsData.length > 0) {
        evaluateCompletedTrades(transactionsData);
      }
    } catch (error) {
      console.error('포트폴리오 데이터 로드 실패:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // 매도 신호 분석
  const analyzeSellSignal = async (holding) => {
    setIsAnalyzing(true);
    setSelectedHolding(holding);
    
    try {
      const result = await aiTradingService.analyzeSellSignal(
        holding.symbol,
        holding.avgPrice,
        holding.purchaseDate,
        holding.quantity
      );
      
      setSellAnalysis(result);
      
      // 매도 추천인 경우
      if (result.shouldSell) {
        const confirmSell = confirm(
          `AI 매도 분석 결과\n\n` +
          `종목: ${holding.symbol}\n` +
          `현재 수익률: ${result.currentReturn}\n` +
          `추천: ${result.recommendation}\n` +
          `신호 점수: ${result.signalScore.toFixed(1)}/100\n\n` +
          `매도하시겠습니까?`
        );
        
        if (confirmSell) {
          await handleSell(holding, result.currentPrice);
        }
      } else {
        alert(
          `AI 분석 결과: ${result.recommendation}\n` +
          `신호 점수: ${result.signalScore.toFixed(1)}/100\n` +
          `보유 유지를 권장합니다.`
        );
      }
    } catch (error) {
      console.error('매도 분석 실패:', error);
      alert('AI 분석 중 오류가 발생했습니다.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // 매도 실행
  const handleSell = async (holding, currentPrice) => {
    try {
      // 매도 거래 생성
      await portfolioService.createTransaction({
        portfolioId: portfolio.id,
        symbol: holding.symbol,
        type: 'SELL',
        quantity: holding.quantity,
        price: currentPrice,
        total: currentPrice * holding.quantity
      });
      
      // 보유 종목 업데이트
      await portfolioService.updateHolding(holding.id, {
        quantity: 0,
        status: 'SOLD'
      });
      
      // 거래 피드백 전송
      await sendTradeFeedback(holding, currentPrice);
      
      alert('매도가 완료되었습니다!');
      loadPortfolioData(); // 데이터 새로고침
    } catch (error) {
      console.error('매도 실행 실패:', error);
      alert('매도 중 오류가 발생했습니다.');
    }
  };

  // 거래 피드백 전송
  const sendTradeFeedback = async (holding, sellPrice) => {
    try {
      const feedback = {
        symbol: holding.symbol,
        entryPrice: holding.avgPrice,
        exitPrice: sellPrice,
        entryDate: holding.purchaseDate,
        exitDate: new Date().toISOString().split('T')[0],
        quantity: holding.quantity,
        returnPct: ((sellPrice - holding.avgPrice) / holding.avgPrice) * 100,
        holdingDays: Math.floor((new Date() - new Date(holding.purchaseDate)) / (1000 * 60 * 60 * 24))
      };
      
      // trade_feedback_service API 호출
      await fetch('http://localhost:8000/api/ai/trade-feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(feedback)
      });
      
      console.log('거래 피드백 전송 완료:', feedback);
    } catch (error) {
      console.error('거래 피드백 전송 실패:', error);
    }
  };

  // 주식 선택 핸들러
  const handleStockSelect = (stockName) => {
    setSelectedStock(stockName);
    const symbol = aiTradingService.getTickerSymbol(stockName);
    setSelectedSymbol(symbol);
    setBuyAnalysis(null);
    setRealtimeAnalysis(null); // 실시간 분석 결과 초기화
  };

  // 주가 데이터 업데이트
  const handleStockData = (stockData) => {
    setCurrentStockData(stockData);
    
    // 실시간 AI 분석 트리거 (자동) - 주식 매수 탭에서만
    if (aiInitialized && stockData && stockData.symbol && activeTab === 'trading') {
      triggerRealtimeAIAnalysis(stockData.symbol);
    }
  };

  // 실시간 AI 분석 함수 (백그라운드 자동 실행)
  const triggerRealtimeAIAnalysis = async (symbol) => {
    // 이미 분석 중이면 스킵
    if (isAnalyzing) return;
    
    try {
      console.log(` 포트폴리오 실시간 매수 분석 시작: ${symbol}`);
      
      // 매수 신호 분석
      const buyAnalysisResult = await aiTradingService.analyzeBuySignal(symbol, 5.0);
      
      // 실시간 분석 결과 업데이트
      setRealtimeAnalysis({
        type: 'realtime',
        ticker: symbol,
        buySignal: {
          recommendation: buyAnalysisResult.recommendation,
          signalScore: buyAnalysisResult.signalScore,
          confidence: buyAnalysisResult.confidence,
          shouldBuy: buyAnalysisResult.shouldBuy,
          technicalIndicators: buyAnalysisResult.technicalIndicators,
          fundamentals: buyAnalysisResult.fundamentals
        },
        timestamp: new Date().toISOString()
      });
      
      console.log(`✅ 포트폴리오 실시간 AI 분석 완료: ${symbol}`);
    } catch (error) {
      console.error('포트폴리오 실시간 AI 분석 실패:', error);
    }
  };

  // 보유 종목 실시간 매도 분석 함수
  const triggerHoldingSellAnalysis = async (holding) => {
    try {
      console.log(` 보유 종목 실시간 매도 분석: ${holding.symbol}`);
      
      const result = await aiTradingService.analyzeSellSignal(
        holding.symbol,
        holding.avgPrice,
        holding.purchaseDate,
        holding.quantity
      );
      
      // 보유 종목별 매도 분석 결과 업데이트
      setHoldingSellAnalysis(prev => ({
        ...prev,
        [holding.id]: {
          ...result,
          timestamp: new Date().toISOString(),
          holding: holding
        }
      }));
      
      console.log(`✅ 보유 종목 실시간 매도 분석 완료: ${holding.symbol}`);
    } catch (error) {
      console.error(`보유 종목 매도 분석 실패 (${holding.symbol}):`, error);
    }
  };

  // 매수 분석
  const analyzeBuySignal = async () => {
    setIsAnalyzing(true);
    try {
      const result = await aiTradingService.analyzeBuySignal(selectedSymbol, 5.0);
      setBuyAnalysis(result);
      
      if (result.shouldBuy) {
        const confirmBuy = confirm(
          `AI 매수 분석 결과\n\n` +
          `종목: ${selectedStock} (${selectedSymbol})\n` +
          `현재가: $${currentStockData?.price?.toFixed(2) || 'N/A'}\n` +
          `추천: ${result.recommendation}\n` +
          `신호 점수: ${result.signalScore.toFixed(1)}/100\n\n` +
          `매수하시겠습니까?`
        );
        
        if (confirmBuy) {
          await handleBuy(result.currentPrice || currentStockData?.price);
        }
      } else {
        alert(
          `AI 분석 결과: ${result.recommendation}\n` +
          `신호 점수: ${result.signalScore.toFixed(1)}/100\n` +
          `매수 대기를 권장합니다.`
        );
      }
    } catch (error) {
      console.error('매수 분석 실패:', error);
      alert('AI 분석 중 오류가 발생했습니다.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // 매수 실행
  const handleBuy = async (currentPrice) => {
    const quantity = prompt('매수할 수량을 입력하세요:');
    if (!quantity || isNaN(quantity) || quantity <= 0) return;
    
    const buyQuantity = parseInt(quantity);
    const totalCost = currentPrice * buyQuantity;
    
    try {
      // 매수 거래 생성
      await portfolioService.createTransaction({
        portfolioId: portfolio.id,
        symbol: selectedSymbol,
        type: 'BUY',
        quantity: buyQuantity,
        price: currentPrice,
        total: totalCost
      });
      
      alert('매수가 완료되었습니다!');
      loadPortfolioData(); // 데이터 새로고침
    } catch (error) {
      console.error('매수 실행 실패:', error);
      alert('매수 중 오류가 발생했습니다.');
    }
  };

  // 완료된 거래에 대한 AI 품질 평가
  const evaluateCompletedTrades = async (transactionsData) => {
    try {
      // 매수-매도 쌍을 찾아서 그룹화
      const tradepairs = findTradePairs(transactionsData);
      
      // 각 거래 쌍에 대해 AI 품질 평가 실행
      for (const tradePair of tradepairs) {
        if (!tradeEvaluations[tradePair.id]) {
          await evaluateTradeQuality(tradePair);
        }
      }
    } catch (error) {
      console.error('거래 품질 평가 실패:', error);
    }
  };

  // 거래 피드백 생성
  const generateTradeFeedback = async (tradePair) => {
    try {
      console.log(`📝 거래 피드백 생성 시작: ${tradePair.symbol}`);
      
      // 로딩 상태 설정
      setFeedbackLoading(prev => ({
        ...prev,
        [tradePair.id]: true
      }));

      // 1. 진입시점과 청산시점 데이터 수집 (품질 평가와 동일)
      const entryData = await fetchHistoricalMarketData(tradePair.symbol, tradePair.entryDate);
      const exitData = await fetchHistoricalMarketData(tradePair.symbol, tradePair.exitDate);
      const marketData = await calculateMarketReturns(tradePair.entryDate, tradePair.exitDate);

      // 2. 거래 피드백 요청 데이터 준비
      const feedbackData = {
        symbol: tradePair.symbol,
        entry_date: tradePair.entryDate,
        exit_date: tradePair.exitDate,
        return_pct: tradePair.returnPct,
        holding_period_days: tradePair.holdingDays,
        entry_price: tradePair.entryPrice,
        exit_price: tradePair.exitPrice,
        quantity: tradePair.quantity,
        position_size_pct: 5.0, // 기본값 5% (포트폴리오 대비 비중)
        
        // 진입시 데이터
        entry_momentum_5d: entryData.momentum_5d || 0,
        entry_momentum_20d: entryData.momentum_20d || 0,
        entry_momentum_60d: entryData.momentum_60d || 0,
        entry_ma_dev_5d: entryData.ma_dev_5d,
        entry_ma_dev_20d: entryData.ma_dev_20d,
        entry_ma_dev_60d: entryData.ma_dev_60d,
        entry_volatility_5d: entryData.volatility_5d,
        entry_volatility_20d: entryData.volatility_20d,
        entry_volatility_60d: entryData.volatility_60d,
        entry_vix: entryData.vix,
        entry_tnx_yield: entryData.tnx_yield,
        entry_pe_ratio: entryData.pe_ratio,
        entry_pb_ratio: entryData.pb_ratio,
        entry_roe: entryData.roe,
        entry_earnings_growth: entryData.earnings_growth,
        entry_operating_margin: entryData.operating_margin || 0.1, // 기본값 10%
        entry_debt_equity_ratio: entryData.debt_equity_ratio || 0.5, // 기본값 0.5
        
        // 변동성 변화 (기본값으로 설정)
        entry_vol_change_5d: entryData.vol_change_5d || 0,
        entry_vol_change_20d: entryData.vol_change_20d || 0,
        entry_vol_change_60d: entryData.vol_change_60d || 0,
        
        // 시장 관련 피처들 (기본값으로 설정)
        market_entry_ma_return_5d: entryData.market_ma_return_5d || 0,
        market_entry_ma_return_20d: entryData.market_ma_return_20d || 0,
        market_entry_cum_return_5d: entryData.market_cum_return_5d || 0,
        market_entry_volatility_20d: entryData.market_volatility_20d || 0.15,
        
        // 청산시 데이터
        exit_momentum_5d: exitData.momentum_5d,
        exit_momentum_20d: exitData.momentum_20d,
        exit_momentum_60d: exitData.momentum_60d,
        exit_ma_dev_5d: exitData.ma_dev_5d,
        exit_ma_dev_20d: exitData.ma_dev_20d,
        exit_ma_dev_60d: exitData.ma_dev_60d,
        exit_volatility_5d: exitData.volatility_5d,
        exit_volatility_20d: exitData.volatility_20d,
        exit_volatility_60d: exitData.volatility_60d,
        exit_vix: exitData.vix,
        exit_tnx_yield: exitData.tnx_yield,
        
        // 시장 성과
        market_return_during_holding: marketData.market_return_during_holding,
        excess_return: tradePair.returnPct / 100 - marketData.market_return_during_holding
      };

      // 3. trade_feedback_service API 호출
      console.log(' AI 거래 피드백 분석 시작:', feedbackData);
      const response = await fetch('http://localhost:8000/api/ai/realtime/trade-feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(feedbackData)
      });

      if (response.ok) {
        const feedbackResult = await response.json();
        
        // 피드백 결과 저장
        setTradeFeedbacks(prev => ({
          ...prev,
          [tradePair.id]: {
            ...feedbackResult,
            tradePair: tradePair,
            timestamp: new Date().toISOString()
          }
        }));

        console.log('✅ 거래 피드백 생성 완료:', feedbackResult);
      } else {
        const error = await response.json();
        console.error('거래 피드백 생성 실패:', error);
        throw new Error(error.detail || '거래 피드백 생성 실패');
      }
    } catch (error) {
      console.error('거래 피드백 생성 오류:', error);
      // 에러 상태도 저장
      setTradeFeedbacks(prev => ({
        ...prev,
        [tradePair.id]: {
          error: error.message,
          timestamp: new Date().toISOString()
        }
      }));
    } finally {
      // 로딩 상태 해제
      setFeedbackLoading(prev => ({
        ...prev,
        [tradePair.id]: false
      }));
    }
  };

  // 거래 쌍 찾기 (매수 후 매도)
  const findTradePairs = (transactionsData) => {
    console.log('🔍 거래 쌍 찾기 시작:', transactionsData);
    
    const pairs = [];
    // 원본 데이터를 수정하지 않도록 복사본을 만들어 작업
    const workingTransactions = transactionsData.map(t => ({ ...t, used: false }));
    const buyTransactions = workingTransactions.filter(t => t.type === 'BUY');
    const sellTransactions = workingTransactions.filter(t => t.type === 'SELL');
    
    console.log('매수 거래:', buyTransactions);
    console.log('매도 거래:', sellTransactions);
    
    buyTransactions.forEach(buyTx => {
      console.log(`${buyTx.symbol} 매수 검사:`, buyTx.date);
      
      const matchingSell = sellTransactions.find(sellTx => {
        console.log(`  - ${sellTx.symbol} 매도 검사:`, sellTx.date);
        console.log(`  - 심볼 일치:`, sellTx.symbol === buyTx.symbol);
        console.log(`  - 날짜 비교:`, new Date(sellTx.date), '>', new Date(buyTx.date), '=', new Date(sellTx.date) > new Date(buyTx.date));
        console.log(`  - 사용됨:`, sellTx.used);
        
        return sellTx.symbol === buyTx.symbol && 
               new Date(sellTx.date) > new Date(buyTx.date) &&
               !sellTx.used; // 이미 매칭되지 않은 매도
      });
      
      console.log(`${buyTx.symbol} 매수에 대한 매칭 매도:`, matchingSell);
      
      if (matchingSell) {
        matchingSell.used = true;
        const holdingDays = Math.floor((new Date(matchingSell.date) - new Date(buyTx.date)) / (1000 * 60 * 60 * 24));
        const returnPct = ((matchingSell.price - buyTx.price) / buyTx.price) * 100;
        
        const tradePair = {
          id: `${buyTx.id}-${matchingSell.id}`,
          symbol: buyTx.symbol,
          buyTransaction: buyTx,
          sellTransaction: matchingSell,
          entryPrice: buyTx.price,
          exitPrice: matchingSell.price,
          entryDate: buyTx.date,
          exitDate: matchingSell.date,
          quantity: buyTx.quantity,
          holdingDays: holdingDays,
          returnPct: returnPct
        };
        
        console.log('생성된 거래 쌍:', tradePair);
        pairs.push(tradePair);
      }
    });
    
    console.log('최종 거래 쌍들:', pairs);
    return pairs;
  };

  // 개별 거래 품질 평가
  const evaluateTradeQuality = async (tradePair) => {
    try {
      console.log(` 거래 품질 평가 시작: ${tradePair.symbol}`);
      
      // 1. 진입시점 데이터 수집 (WebSocket/Yahoo Finance)
      console.log(` 진입시점 데이터 수집: ${tradePair.entryDate}`);
      const entryData = await fetchHistoricalMarketData(tradePair.symbol, tradePair.entryDate);
      
      // 2. 청산시점 데이터 수집 (WebSocket/Yahoo Finance)  
      console.log(` 청산시점 데이터 수집: ${tradePair.exitDate}`);
      const exitData = await fetchHistoricalMarketData(tradePair.symbol, tradePair.exitDate);
      
      // 3. 시장 수익률 및 초과 수익률 계산
      const marketData = await calculateMarketReturns(tradePair.entryDate, tradePair.exitDate);
      
      // 4. AI 모델용 피처 데이터 준비
      const featureData = prepareTradeQualityFeatures(tradePair, entryData, exitData, marketData);
      
      // 5. trade_quality_evaluator API 호출 (포괄적 피처 데이터 사용)
      console.log('🚀 AI 모델 호출 데이터:', featureData);
      const response = await fetch('http://localhost:8000/api/ai/realtime/evaluate-trade-quality-comprehensive', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(featureData)
      });
      
      if (response.ok) {
        const evaluationResult = await response.json();
        
        // 평가 결과 저장
        setTradeEvaluations(prev => ({
          ...prev,
          [tradePair.id]: {
            ...evaluationResult,
            tradePair: tradePair,
            timestamp: new Date().toISOString()
          }
        }));
        
        console.log(`✅ 거래 품질 평가 완료: ${tradePair.symbol} - 점수: ${evaluationResult.qualityScore}/100`);
      } else {
        console.error('거래 품질 평가 API 실패:', response.status);
      }
    } catch (error) {
      console.error(`거래 품질 평가 실패 (${tradePair.symbol}):`, error);
    }
  };

  // 과거 시장 데이터 수집 (Yahoo Finance API 활용)
  const fetchHistoricalMarketData = async (symbol, date) => {
    try {
      // Yahoo Finance 과거 데이터 API 호출
      const response = await fetch(`http://localhost:8000/api/stock/historical-data?symbol=${symbol}&date=${date}`);
      
      if (response.ok) {
        const data = await response.json();
        return {
          // 모멘텀 지표 (5일, 20일, 60일)
          momentum_5d: data.momentum_5d || 0,
          momentum_20d: data.momentum_20d || 0, 
          momentum_60d: data.momentum_60d || 0,
          
          // 이동평균 이탈도
          ma_dev_5d: data.ma_dev_5d || 0,
          ma_dev_20d: data.ma_dev_20d || 0,
          ma_dev_60d: data.ma_dev_60d || 0,
          
          // 변동성 지표
          volatility_5d: data.volatility_5d || 0,
          volatility_20d: data.volatility_20d || 0,
          volatility_60d: data.volatility_60d || 0,
          
          // 변동성 변화
          vol_change_5d: data.vol_change_5d || 0,
          vol_change_20d: data.vol_change_20d || 0,
          vol_change_60d: data.vol_change_60d || 0,
          
          // 시장 지표
          vix: data.vix || 15,
          tnx_yield: data.tnx_yield || 4.5,
          ratio_52w_high: data.ratio_52w_high || 0.8,
          
          // 기본 데이터
          price: data.price || 0,
          volume: data.volume || 0
        };
      } else {
        // API 실패시 기본값 반환
        console.warn(`과거 데이터 수집 실패: ${symbol} ${date}`);
        return getDefaultMarketData();
      }
    } catch (error) {
      console.error(`과거 데이터 수집 오류: ${symbol} ${date}`, error);
      return getDefaultMarketData();
    }
  };

  // 기본 시장 데이터 (API 실패시 사용)
  const getDefaultMarketData = () => ({
    momentum_5d: Math.random() * 10 - 5,
    momentum_20d: Math.random() * 15 - 7.5,
    momentum_60d: Math.random() * 20 - 10,
    ma_dev_5d: Math.random() * 0.1 - 0.05,
    ma_dev_20d: Math.random() * 0.15 - 0.075,
    ma_dev_60d: Math.random() * 0.2 - 0.1,
    volatility_5d: Math.random() * 0.5 + 0.1,
    volatility_20d: Math.random() * 0.3 + 0.15,
    volatility_60d: Math.random() * 0.4 + 0.1,
    vol_change_5d: Math.random() * 0.2 - 0.1,
    vol_change_20d: Math.random() * 0.3 - 0.15,
    vol_change_60d: Math.random() * 0.4 - 0.2,
    vix: 15 + Math.random() * 10,
    tnx_yield: 4.5 + Math.random() * 2,
    ratio_52w_high: 0.7 + Math.random() * 0.3
  });

  // 시장 수익률 계산 (S&P 500 기준)
  const calculateMarketReturns = async (entryDate, exitDate) => {
    try {
      // S&P 500 수익률 계산
      const response = await fetch(`http://localhost:8000/api/stock/market-returns?start_date=${entryDate}&end_date=${exitDate}`);
      
      if (response.ok) {
        const data = await response.json();
        return {
          market_return_during_holding: data.market_return_pct / 100 || 0, // 백분율을 소수로 변환
          spy_return: data.market_return_pct / 100 || 0
        };
      } else {
        // 기본값 반환
        const days = Math.floor((new Date(exitDate) - new Date(entryDate)) / (1000 * 60 * 60 * 24));
        const annualizedReturn = 0.1; // 10% 연간 수익률 가정
        const marketReturn = (annualizedReturn / 365) * days;
        
        return {
          market_return_during_holding: marketReturn,
          spy_return: marketReturn
        };
      }
    } catch (error) {
      console.error('시장 수익률 계산 오류:', error);
      const days = Math.floor((new Date(exitDate) - new Date(entryDate)) / (1000 * 60 * 60 * 24));
      const marketReturn = (0.1 / 365) * days; // 기본 10% 연간 수익률
      
      return {
        market_return_during_holding: marketReturn,
        spy_return: marketReturn
      };
    }
  };

  // AI 모델용 피처 데이터 준비
  const prepareTradeQualityFeatures = (tradePair, entryData, exitData, marketData) => {
    // 변화량 계산
    const changes = {
      change_momentum_5d: exitData.momentum_5d - entryData.momentum_5d,
      change_momentum_20d: exitData.momentum_20d - entryData.momentum_20d,
      change_momentum_60d: exitData.momentum_60d - entryData.momentum_60d,
      change_ma_dev_5d: exitData.ma_dev_5d - entryData.ma_dev_5d,
      change_ma_dev_20d: exitData.ma_dev_20d - entryData.ma_dev_20d,
      change_ma_dev_60d: exitData.ma_dev_60d - entryData.ma_dev_60d,
      change_volatility_5d: exitData.volatility_5d - entryData.volatility_5d,
      change_volatility_20d: exitData.volatility_20d - entryData.volatility_20d,
      change_volatility_60d: exitData.volatility_60d - entryData.volatility_60d,
      change_vix: exitData.vix - entryData.vix,
      change_tnx_yield: exitData.tnx_yield - entryData.tnx_yield,
      change_ratio_52w_high: exitData.ratio_52w_high - entryData.ratio_52w_high
    };

    // 초과 수익률 계산
    const excessReturn = tradePair.returnPct / 100 - marketData.market_return_during_holding;

    return {
      // 기본 거래 정보
      symbol: tradePair.symbol,
      return_pct: tradePair.returnPct,
      holding_period_days: tradePair.holdingDays,
      position_size_pct: 5.0, // 기본값, 실제로는 포트폴리오 대비 계산 필요
      
      // 진입시 피처들
      entry_momentum_5d: entryData.momentum_5d,
      entry_momentum_20d: entryData.momentum_20d,
      entry_momentum_60d: entryData.momentum_60d,
      entry_ma_dev_5d: entryData.ma_dev_5d,
      entry_ma_dev_20d: entryData.ma_dev_20d,
      entry_ma_dev_60d: entryData.ma_dev_60d,
      entry_volatility_5d: entryData.volatility_5d,
      entry_volatility_20d: entryData.volatility_20d, // 추가된 20일 변동성
      entry_volatility_60d: entryData.volatility_60d,
      entry_vol_change_5d: entryData.vol_change_5d,
      entry_vol_change_20d: entryData.vol_change_20d,
      entry_vol_change_60d: entryData.vol_change_60d,
      entry_vix: entryData.vix,
      entry_tnx_yield: entryData.tnx_yield,
      
      // 청산시 피처들
      exit_momentum_5d: exitData.momentum_5d,
      exit_momentum_20d: exitData.momentum_20d,
      exit_momentum_60d: exitData.momentum_60d,
      exit_ma_dev_5d: exitData.ma_dev_5d,
      exit_ma_dev_20d: exitData.ma_dev_20d,
      exit_ma_dev_60d: exitData.ma_dev_60d,
      exit_volatility_5d: exitData.volatility_5d,
      exit_volatility_20d: exitData.volatility_20d,
      exit_volatility_60d: exitData.volatility_60d,
      exit_vix: exitData.vix,
      exit_tnx_yield: exitData.tnx_yield,
      exit_ratio_52w_high: exitData.ratio_52w_high,
      
      // 변화량 피처들
      ...changes,
      
      // 시장 대비 성과
      market_return_during_holding: marketData.market_return_during_holding,
      excess_return: excessReturn,
      
      // 거래 날짜 정보 (참조용)
      entryDate: tradePair.entryDate,
      exitDate: tradePair.exitDate,
      entryPrice: tradePair.entryPrice,
      exitPrice: tradePair.exitPrice,
      quantity: tradePair.quantity
    };
  };

  // 수익률 계산
  const calculateReturn = (holding) => {
    if (!holding.currentPrice || !holding.avgPrice) return 0;
    return ((holding.currentPrice - holding.avgPrice) / holding.avgPrice) * 100;
  };

  // 총 자산 계산
  const calculateTotalValue = () => {
    return holdings.reduce((total, holding) => {
      return total + (holding.currentPrice || holding.avgPrice) * holding.quantity;
    }, 0);
  };

  // 품질 등급 결정
  const getQualityGrade = (score) => {
    if (score >= 80) return 'excellent';
    if (score >= 70) return 'good';
    if (score >= 60) return 'average';
    if (score >= 50) return 'poor';
    return 'bad';
  };

  // 품질 설명
  const getQualityDescription = (score) => {
    if (score >= 80) return '우수한 거래';
    if (score >= 70) return '좋은 거래';
    if (score >= 60) return '평균적인 거래';
    if (score >= 50) return '아쉬운 거래';
    return '나쁜 거래';
  };

  return (
    <div className="portfolio-detail">
      {/* 헤더 */}
      <div className="detail-header">
        <button className="back-button" onClick={onBack}>
          ← 포트폴리오 목록
        </button>
        <div className="portfolio-title">
          <h2>{portfolio?.name}</h2>
          <p>생성일: {new Date(portfolio?.created_at).toLocaleDateString()}</p>
        </div>
      </div>

      {/* 포트폴리오 요약 */}
      <div className="portfolio-summary-card">
        <h3>포트폴리오 요약</h3>
        <div className="summary-grid">
          <div className="summary-item">
            <span className="label">총 자산</span>
            <span className="value">${calculateTotalValue().toLocaleString()}</span>
          </div>
          <div className="summary-item">
            <span className="label">보유 종목</span>
            <span className="value">{holdings.length}개</span>
          </div>
          <div className="summary-item">
            <span className="label">총 거래</span>
            <span className="value">{transactions.length}건</span>
          </div>
        </div>
      </div>

      {/* 탭 네비게이션 */}
      <div className="detail-tabs">
        <button 
          className={`tab-btn ${activeTab === 'holdings' ? 'active' : ''}`}
          onClick={() => setActiveTab('holdings')}
        >
          보유 종목
        </button>
        <button 
          className={`tab-btn ${activeTab === 'transactions' ? 'active' : ''}`}
          onClick={() => setActiveTab('transactions')}
        >
          거래 내역
        </button>
        <button 
          className={`tab-btn ${activeTab === 'trading' ? 'active' : ''}`}
          onClick={() => setActiveTab('trading')}
        >
          주식 매수
        </button>
      </div>

      {/* 탭 컨텐츠 */}
      <div className="tab-content">
        {isLoading ? (
          <div className="loading">데이터를 불러오는 중...</div>
        ) : (
          <>
            {/* 보유 종목 탭 */}
            {activeTab === 'holdings' && (
              <div className="holdings-section">
                {holdings.length === 0 ? (
                  <div className="empty-state">
                    <p>보유 중인 종목이 없습니다.</p>
                    <p>모의투자를 시작해보세요!</p>
                  </div>
                ) : (
                  <div className="holdings-list">
                    {holdings.map((holding) => {
                      const sellAnalysisResult = holdingSellAnalysis[holding.id];
                      return (
                        <div key={holding.id} className="holding-card">
                          <div className="holding-header">
                            <div className="stock-info">
                              <h4>{holding.symbol}</h4>
                              <span className="quantity">{holding.quantity}주</span>
                            </div>
                            <div className="stock-price">
                              <span className="current-price">
                                ${(holding.currentPrice || holding.avgPrice).toFixed(2)}
                              </span>
                              <span className={`return ${calculateReturn(holding) >= 0 ? 'positive' : 'negative'}`}>
                                {calculateReturn(holding) >= 0 ? '+' : ''}{calculateReturn(holding).toFixed(2)}%
                              </span>
                            </div>
                          </div>

                          {/* 실시간 AI 매도 분석 결과 */}
                          {sellAnalysisResult ? (
                            <div className="realtime-sell-analysis">
                              <div className="sell-analysis-header">
                                <h6> 실시간 매도 분석</h6>
                                <span className="live-indicator">● LIVE</span>
                              </div>
                              <div className="sell-analysis-summary">
                                <div className="sell-recommendation-row">
                                  <span className="label">추천:</span>
                                  <span className={`value ${sellAnalysisResult.shouldSell ? 'sell-positive' : 'hold-negative'}`}>
                                    {sellAnalysisResult.recommendation}
                                  </span>
                                </div>
                                <div className="sell-recommendation-row">
                                  <span className="label">신호 점수:</span>
                                  <span className="value">{sellAnalysisResult.signalScore?.toFixed(1)}/100</span>
                                </div>
                                <div className="sell-recommendation-row">
                                  <span className="label">신뢰도:</span>
                                  <span className="value">{(sellAnalysisResult.confidence * 100).toFixed(1)}%</span>
                                </div>
                                <div className="sell-recommendation-row">
                                  <span className="label">현재 수익률:</span>
                                  <span className={`value ${sellAnalysisResult.currentReturn?.includes('+') ? 'positive' : 'negative'}`}>
                                    {sellAnalysisResult.currentReturn}
                                  </span>
                                </div>
                              </div>
                              <div className="sell-last-update">
                                마지막 업데이트: {new Date(sellAnalysisResult.timestamp).toLocaleTimeString()}
                              </div>
                            </div>
                          ) : (
                            <div className="sell-analysis-waiting">
                              <p> 실시간 매도 분석 준비 중...</p>
                            </div>
                          )}
                        
                        <div className="holding-details">
                          <div className="detail-row">
                            <span>평균 매수가:</span>
                            <span>${holding.avgPrice.toFixed(2)}</span>
                          </div>
                          <div className="detail-row">
                            <span>매수일:</span>
                            <span>{new Date(holding.purchaseDate).toLocaleDateString()}</span>
                          </div>
                          <div className="detail-row">
                            <span>평가금액:</span>
                            <span>${((holding.currentPrice || holding.avgPrice) * holding.quantity).toLocaleString()}</span>
                          </div>
                          <div className="detail-row">
                            <span>손익:</span>
                            <span className={calculateReturn(holding) >= 0 ? 'positive' : 'negative'}>
                              ${(((holding.currentPrice || holding.avgPrice) - holding.avgPrice) * holding.quantity).toFixed(2)}
                            </span>
                          </div>
                        </div>
                        
                        <div className="holding-actions">
                          <button 
                            className="realtime-sell-btn"
                            style={{
                              background: sellAnalysisResult?.shouldSell 
                                ? 'linear-gradient(135deg, #FF5722 0%, #E64A19 100%)'
                                : sellAnalysisResult
                                ? 'linear-gradient(135deg, #4CAF50 0%, #388E3C 100%)'
                                : '#666'
                            }}
                            onClick={() => {
                              if (sellAnalysisResult?.shouldSell) {
                                const confirmSell = confirm(
                                  ` 실시간 AI 매도 분석 결과\n\n` +
                                  `종목: ${holding.symbol}\n` +
                                  `현재 수익률: ${sellAnalysisResult.currentReturn}\n` +
                                  `추천: ${sellAnalysisResult.recommendation}\n` +
                                  `신호 점수: ${sellAnalysisResult.signalScore.toFixed(1)}/100\n` +
                                  `신뢰도: ${(sellAnalysisResult.confidence * 100).toFixed(1)}%\n\n` +
                                  `매도하시겠습니까?`
                                );
                                
                                if (confirmSell) {
                                  handleSell(holding, sellAnalysisResult.currentPrice || holding.currentPrice);
                                }
                              } else {
                                alert(
                                  `AI 분석 결과: ${sellAnalysisResult?.recommendation || '분석 중'}\n` +
                                  `신호 점수: ${sellAnalysisResult?.signalScore?.toFixed(1) || 'N/A'}/100\n` +
                                  `보유 유지를 권장합니다.`
                                );
                              }
                            }}
                            disabled={!sellAnalysisResult}
                          >
                            {sellAnalysisResult ? 
                              (sellAnalysisResult.shouldSell ? '🔥 실시간 AI 매도' : '📊 보유 유지') : 
                              '분석 중...'
                            }
                          </button>
                        </div>
                      </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            {/* 거래 내역 탭 */}
            {activeTab === 'transactions' && (
              <div className="transactions-section">
                {transactions.length === 0 ? (
                  <div className="empty-state">
                    <p>거래 내역이 없습니다.</p>
                  </div>
                ) : (
                  <div className="transactions-list">
                    {(() => {
                      // 거래를 쌍으로 그룹화
                      const tradePairs = findTradePairs(transactions);
                      const individualTransactions = transactions.filter(t => {
                        return !tradePairs.some(pair => 
                          pair.buyTransaction.id === t.id || pair.sellTransaction.id === t.id
                        );
                      });
                      
                      return (
                        <>
                          {/* 디버깅 정보 */}
                          <div style={{ background: '#1E1E22', padding: '10px', marginBottom: '10px', borderRadius: '6px' }}>
                            <p style={{ color: '#F2F219', margin: 0 }}>
                              📊 거래 쌍: {tradePairs.length}개 | 개별 거래: {individualTransactions.length}개
                            </p>
                          </div>
                          
                          {/* 완료된 거래 쌍들 (AI 평가 포함) */}
                          {tradePairs.map((tradePair) => {
                            const evaluation = tradeEvaluations[tradePair.id];
                            return (
                              <div key={`pair-${tradePair.id}`} className="trade-pair-card">
                                <div className="trade-pair-header">
                                  <div className="pair-info">
                                    <h4>{tradePair.symbol} 완료된 거래</h4>
                                    <span className="pair-period">
                                      {new Date(tradePair.entryDate).toLocaleDateString()} ~ {new Date(tradePair.exitDate).toLocaleDateString()}
                                    </span>
                                  </div>
                                  <div className="pair-result">
                                    <span className={`return-pct ${tradePair.returnPct >= 0 ? 'positive' : 'negative'}`}>
                                      {tradePair.returnPct >= 0 ? '+' : ''}{tradePair.returnPct.toFixed(2)}%
                                    </span>
                                    <span className="holding-days">{tradePair.holdingDays}일 보유</span>
                                  </div>
                                </div>

                                {/* AI 품질 평가 결과 */}
                                {evaluation ? (
                                  <div className="ai-quality-evaluation">
                                    <div className="quality-header">
                                      <h5> AI 거래 품질 평가</h5>
                                      <span className="quality-score">
                                        {evaluation.qualityScore?.toFixed(1) || 'N/A'}/100점
                                      </span>
                                    </div>
                                    <div className="quality-details">
                                      <div className="quality-row">
                                        <span className="label">전체 평가:</span>
                                        <span className={`value quality-${getQualityGrade(evaluation.qualityScore)}`}>
                                          {getQualityDescription(evaluation.qualityScore)}
                                        </span>
                                      </div>
                                      <div className="quality-row">
                                        <span className="label">진입 품질:</span>
                                        <span className="value">{evaluation.entryQuality?.toFixed(1) || 'N/A'}/100</span>
                                      </div>
                                      <div className="quality-row">
                                        <span className="label">청산 타이밍:</span>
                                        <span className="value">{evaluation.exitTiming?.toFixed(1) || 'N/A'}/100</span>
                                      </div>
                                      <div className="quality-row">
                                        <span className="label">결과 품질:</span>
                                        <span className="value">{evaluation.resultQuality?.toFixed(1) || 'N/A'}/100</span>
                                      </div>
                                    </div>
                                    <div className="quality-feedback">
                                      <p>{evaluation.feedback || '이 거래에 대한 AI 분석이 완료되었습니다.'}</p>
                                    </div>
                                    <div className="evaluation-time">
                                      평가 시간: {new Date(evaluation.timestamp).toLocaleString()}
                                    </div>
                                  </div>
                                ) : (
                                  <div className="quality-loading">
                                    <p> AI 품질 평가 중...</p>
                                  </div>
                                )}

                                {/* AI 거래 피드백 섹션 */}
                                <div className="trade-feedback-section">
                                  {tradeFeedbacks[tradePair.id] ? (
                                    // 피드백 결과가 있는 경우
                                    <div className="trade-feedback-result">
                                      <div className="feedback-header">
                                        <h5>📝 AI 데이터 기반 거래 피드백</h5>
                                        <span className="feedback-status-badge success">분석 완료</span>
                                      </div>
                                      
                                      {tradeFeedbacks[tradePair.id].error ? (
                                        // 에러가 있는 경우
                                        <div className="feedback-error">
                                          <p>❌ 피드백 생성 실패: {tradeFeedbacks[tradePair.id].error}</p>
                                        </div>
                                      ) : (
                                        // 정상적인 피드백 결과
                                        <div className="feedback-content">
                                          {/* 전체 평가 */}
                                          {tradeFeedbacks[tradePair.id].feedback_summary && (
                                            <div className="feedback-summary">
                                              <div className="summary-item">
                                                <strong>전체 평가:</strong>
                                                <span>{tradeFeedbacks[tradePair.id].feedback_summary.overall_assessment}</span>
                                              </div>
                                            </div>
                                          )}

                                          {/* 핵심 인사이트 */}
                                          {tradeFeedbacks[tradePair.id].adaptive_insights && tradeFeedbacks[tradePair.id].adaptive_insights.length > 0 && (
                                            <div className="feedback-insights">
                                              <strong>📊 데이터 기반 인사이트:</strong>
                                              <ul>
                                                {tradeFeedbacks[tradePair.id].adaptive_insights.slice(0, 3).map((insight, index) => (
                                                  <li key={index} className={`insight-${insight.type}`}>
                                                    {insight.message}
                                                    <small className="data-basis">({insight.data_basis})</small>
                                                  </li>
                                                ))}
                                              </ul>
                                            </div>
                                          )}

                                          {/* 학습 기회 */}
                                          {tradeFeedbacks[tradePair.id].learning_opportunities && tradeFeedbacks[tradePair.id].learning_opportunities.length > 0 && (
                                            <div className="feedback-opportunities">
                                              <strong>💡 학습 기회:</strong>
                                              <ul>
                                                {tradeFeedbacks[tradePair.id].learning_opportunities.slice(0, 2).map((opp, index) => (
                                                  <li key={index}>
                                                    <span className="opp-area">[{opp.area}]</span>
                                                    <span className="opp-learning">{opp.learning}</span>
                                                  </li>
                                                ))}
                                              </ul>
                                            </div>
                                          )}

                                          {/* SHAP 분석 결과 (상위 기여 요인) */}
                                          {tradeFeedbacks[tradePair.id].shap_analysis && (
                                            <div className="feedback-shap">
                                              <strong>모델 의사결정 요인 분석:</strong>
                                              {Object.entries(tradeFeedbacks[tradePair.id].shap_analysis).map(([modelType, analysis]) => {
                                                if (analysis.error || !analysis.top_contributors) return null;
                                                return (
                                                  <div key={modelType} className="shap-model-detailed">
                                                    <div className="model-header">
                                                      <span className="model-name">{modelType.replace('_shap', '').toUpperCase()}</span>
                                                      <span className="model-prediction">예측: {analysis.prediction?.toFixed(1)}점</span>
                                                    </div>
                                                    <div className="top-contributors">
                                                      {analysis.top_contributors.slice(0, 3).map((factor, index) => (
                                                        <div key={index} className="contributor">
                                                          <span className="factor-name">{factor.feature.replace(/_/g, ' ')}</span>
                                                          <span className="factor-value">값: {factor.actual_value}</span>
                                                          <span className={`factor-contribution ${factor.contribution >= 0 ? 'positive' : 'negative'}`}>
                                                            {factor.contribution >= 0 ? '+' : ''}{factor.contribution.toFixed(1)}점
                                                          </span>
                                                        </div>
                                                      ))}
                                                    </div>
                                                  </div>
                                                );
                                              })}
                                            </div>
                                          )}

                                          <div className="feedback-time">
                                            피드백 생성 시간: {new Date(tradeFeedbacks[tradePair.id].timestamp).toLocaleString()}
                                          </div>
                                        </div>
                                      )}
                                    </div>
                                  ) : feedbackLoading[tradePair.id] ? (
                                    // 피드백 생성 중인 경우
                                    <div className="feedback-loading">
                                      <p>AI 거래 피드백 생성 중...</p>
                                    </div>
                                  ) : (
                                    // 피드백 생성 버튼
                                    <div className="feedback-generate">
                                      <button 
                                        className="feedback-btn"
                                        onClick={() => generateTradeFeedback(tradePair)}
                                        disabled={feedbackLoading[tradePair.id]}
                                      >
                                        📝 AI 거래 피드백 생성
                                      </button>
                                      <small className="feedback-help">
                                        과거 데이터와 비교하여 상세한 거래 분석을 제공합니다
                                      </small>
                                    </div>
                                  )}
                                </div>

                                {/* 거래 상세 정보 */}
                                <div className="trade-pair-details">
                                  <div className="trade-detail">
                                    <span className="trade-type buy">매수</span>
                                    <span>${tradePair.entryPrice.toFixed(2)} × {tradePair.quantity}주</span>
                                    <span>${(tradePair.entryPrice * tradePair.quantity).toLocaleString()}</span>
                                  </div>
                                  <div className="trade-detail">
                                    <span className="trade-type sell">매도</span>
                                    <span>${tradePair.exitPrice.toFixed(2)} × {tradePair.quantity}주</span>
                                    <span>${(tradePair.exitPrice * tradePair.quantity).toLocaleString()}</span>
                                  </div>
                                </div>

                                {/* 거래 피드백 상태 */}
                                {tradePair.sellTransaction.feedbackSent && (
                                  <div className="feedback-status">

                                  </div>
                                )}
                              </div>
                            );
                          })}

                          {/* 개별 거래들 (아직 완료되지 않은 매수/매도) */}
                          {individualTransactions.map((transaction) => (
                            <div key={transaction.id} className="transaction-card">
                              <div className="transaction-header">
                                <span className={`transaction-type ${transaction.type.toLowerCase()}`}>
                                  {transaction.type === 'BUY' ? '매수' : '매도'}
                                </span>
                                <span className="transaction-date">
                                  {new Date(transaction.date).toLocaleDateString()}
                                </span>
                              </div>
                              
                              <div className="transaction-details">
                                <div className="detail-row">
                                  <span>종목:</span>
                                  <span className="symbol">{transaction.symbol}</span>
                                </div>
                                <div className="detail-row">
                                  <span>수량:</span>
                                  <span>{transaction.quantity}주</span>
                                </div>
                                <div className="detail-row">
                                  <span>가격:</span>
                                  <span>${transaction.price.toFixed(2)}</span>
                                </div>
                                <div className="detail-row">
                                  <span>총액:</span>
                                  <span className="total">${transaction.total.toLocaleString()}</span>
                                </div>
                              </div>
                              
                              {/* 거래 피드백 상태 */}
                              {transaction.feedbackSent && (
                                <div className="feedback-status">
                                  ✅ AI 학습 데이터 전송 완료
                                </div>
                              )}
                            </div>
                          ))}
                        </>
                      );
                    })()}
                  </div>
                )}
              </div>
            )}

            {/* 주식 매수 탭 */}
            {activeTab === 'trading' && (
              <div className="trading-section">
                <div className="trading-layout">
                  {/* 선택된 주식 정보 및 실시간 AI 분석 */}
                  <div className="selected-stock-card">
                    <h4>선택된 주식: {selectedStock}</h4>
                    <div className="stock-price-info">
                      <span className="current-price">
                        ${currentStockData?.price?.toFixed(2) || 'N/A'}
                      </span>
                      {currentStockData && (
                        <span className={`price-change ${currentStockData.change >= 0 ? 'positive' : 'negative'}`}>
                          {currentStockData.change >= 0 ? '+' : ''}{currentStockData.change?.toFixed(2)}%
                        </span>
                      )}
                    </div>

                    {/* 실시간 AI 분석 결과 */}
                    {realtimeAnalysis && realtimeAnalysis.buySignal ? (
                      <div className="realtime-analysis">
                        <div className="analysis-header">
                          <h5> 실시간 AI 분석</h5>
                          <span className="live-indicator">● LIVE</span>
                        </div>
                        <div className="analysis-summary">
                          <div className="recommendation-row">
                            <span className="label">추천:</span>
                            <span className={`value ${realtimeAnalysis.buySignal.shouldBuy ? 'positive' : 'negative'}`}>
                              {realtimeAnalysis.buySignal.recommendation}
                            </span>
                          </div>
                          <div className="recommendation-row">
                            <span className="label">신호 점수:</span>
                            <span className="value">{realtimeAnalysis.buySignal.signalScore?.toFixed(1)}/100</span>
                          </div>
                          <div className="recommendation-row">
                            <span className="label">신뢰도:</span>
                            <span className="value">{(realtimeAnalysis.buySignal.confidence * 100).toFixed(1)}%</span>
                          </div>
                          <div className="recommendation-row">
                            <span className="label">모멘텀(20일):</span>
                            <span className="value">{realtimeAnalysis.buySignal.technicalIndicators?.momentum20d?.toFixed(2)}%</span>
                          </div>
                        </div>
                        <div className="last-update">
                          마지막 업데이트: {new Date(realtimeAnalysis.timestamp).toLocaleTimeString()}
                        </div>
                      </div>
                    ) : (
                      <div className="analysis-waiting">
                        <p> 실시간 AI 분석 준비 중...</p>
                      </div>
                    )}

                    <div className="trading-actions">
                      <button 
                        className="buy-btn"
                        onClick={() => {
                          if (realtimeAnalysis?.buySignal) {
                            const confirmBuy = confirm(
                              ` 실시간 AI 분석 결과\n\n` +
                              `종목: ${selectedStock} (${selectedSymbol})\n` +
                              `추천: ${realtimeAnalysis.buySignal.recommendation}\n` +
                              `신호 점수: ${realtimeAnalysis.buySignal.signalScore.toFixed(1)}/100\n` +
                              `신뢰도: ${(realtimeAnalysis.buySignal.confidence * 100).toFixed(1)}%\n` +
                              `현재가: $${currentStockData?.price?.toFixed(2) || 'N/A'}\n\n` +
                              `매수를 진행하시겠습니까?`
                            );
                            
                            if (confirmBuy) {
                              handleBuy(currentStockData?.price);
                            }
                          } else {
                            alert('실시간 AI 분석 중입니다. 잠시 후 다시 시도해주세요.');
                          }
                        }}
                        disabled={!realtimeAnalysis?.buySignal}
                      >
                        {realtimeAnalysis?.buySignal ? '실시간 AI 매수' : '분석 중...'}
                      </button>
                    </div>
                  </div>

                  {/* 실시간 주식 목록 */}
                  <div className="stock-list-card">
                    <h4>실시간 주식 목록</h4>
                    <div className="stock-list-container" style={{ maxHeight: '500px', overflowY: 'auto' }}>
                      <LiveQuotesList 
                        onStockSelect={handleStockSelect}
                        selectedStock={selectedStock}
                        onStockData={handleStockData}
                      />
                    </div>
                  </div>
                </div>

                {/* 매수 분석 결과 */}
                {buyAnalysis && (
                  <div className="buy-analysis-card">
                    <h4>AI 매수 분석 결과</h4>
                    <div className="analysis-details">
                      <div className="detail-row">
                        <span>종목:</span>
                        <span>{selectedSymbol}</span>
                      </div>
                      <div className="detail-row">
                        <span>추천:</span>
                        <span className={buyAnalysis.shouldBuy ? 'positive' : 'negative'}>
                          {buyAnalysis.recommendation}
                        </span>
                      </div>
                      <div className="detail-row">
                        <span>신호 점수:</span>
                        <span>{buyAnalysis.signalScore?.toFixed(1)}/100</span>
                      </div>
                      <div className="detail-row">
                        <span>신뢰도:</span>
                        <span>{(buyAnalysis.confidence * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>

      {/* 매도 분석 결과 */}
      {sellAnalysis && (
        <div className="sell-analysis-modal">
          <div className="modal-content">
            <h3>AI 매도 분석 결과</h3>
            <div className="analysis-details">
              <p>종목: {selectedHolding?.symbol}</p>
              <p>현재 수익률: {sellAnalysis.currentReturn}</p>
              <p>추천: {sellAnalysis.recommendation}</p>
              <p>신호 점수: {sellAnalysis.signalScore?.toFixed(1)}/100</p>
              <p>신뢰도: {(sellAnalysis.confidence * 100).toFixed(1)}%</p>
            </div>
            <button onClick={() => setSellAnalysis(null)}>닫기</button>
          </div>
        </div>
      )}
    </div>
  );
};

export default PortfolioDetail;