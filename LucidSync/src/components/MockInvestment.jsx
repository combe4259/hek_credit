import React, { useState, useEffect } from 'react';
import { portfolioService } from '../services/portfolioService';
import aiTradingService from '../services/aiTradingService';
import LiveQuotesList from './LiveQuotesList';
import PortfolioDetail from './PortfolioDetail';
import './TabContent.css';

const MockInvestment = ({ user }) => {
  const [selectedStock, setSelectedStock] = useState('Apple'); // 회사명
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL'); // 심볼
  const [showPopup, setShowPopup] = useState(false);
  const [isActiveInvestment, setIsActiveInvestment] = useState(false);
  const [portfolios, setPortfolios] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedPortfolio, setSelectedPortfolio] = useState(null);
  const [currentStockData, setCurrentStockData] = useState(null);
  const [showPortfolioDetail, setShowPortfolioDetail] = useState(false);
  const [investmentSettings, setInvestmentSettings] = useState({
    portfolioTitle: '',
    totalAssets: '',
    assetUnit: 'USD', // 'USD' or 'KRW'
    startDate: '',
    endDate: '',
    riskLevel: 'medium'
  });
  
  // AI 분석 관련 state 추가
  const [aiAnalysisResult, setAiAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [aiInitialized, setAiInitialized] = useState(false);

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
  
  // 실시간 AI 분석 주기적 실행 (5초마다)
  useEffect(() => {
    if (!aiInitialized || !selectedSymbol || !isActiveInvestment) return;
    
    // 초기 분석 실행
    triggerRealtimeAIAnalysis(selectedSymbol);
    
    // 5초마다 자동 분석
    const intervalId = setInterval(() => {
      if (currentStockData && currentStockData.price) {
        console.log(`⏱️ 주기적 AI 분석: ${selectedSymbol}`);
        triggerRealtimeAIAnalysis(selectedSymbol);
      }
    }, 5000); // 5초마다 실행
    
    return () => clearInterval(intervalId);
  }, [aiInitialized, selectedSymbol, isActiveInvestment]);

  // 포트폴리오 목록 가져오기
  useEffect(() => {
    const fetchPortfolios = async () => {
      if (!user?.id) return;
      
      setIsLoading(true);
      setError(null);
      
      try {
        const portfolioData = await portfolioService.getPortfolios(user.id);
        setPortfolios(portfolioData);
      } catch (err) {
        console.error('포트폴리오 조회 실패:', err);
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };

    fetchPortfolios();
  }, [user?.id]);

  // 포트폴리오 생성
  const handleCreatePortfolio = async () => {
    if (!investmentSettings.portfolioTitle.trim()) {
      alert('포트폴리오 제목을 입력해주세요.');
      return;
    }

    if (!user?.id) {
      alert('사용자 정보를 찾을 수 없습니다.');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const newPortfolio = await portfolioService.createPortfolio(
        user.id, 
        investmentSettings.portfolioTitle
      );
      
      // 새 포트폴리오를 목록에 추가
      setPortfolios(prev => [...prev, newPortfolio]);
      
      // 설정 초기화
      setInvestmentSettings(prev => ({ ...prev, portfolioTitle: '' }));
      setShowPopup(false);
      
      alert('포트폴리오가 성공적으로 생성되었습니다!');
    } catch (err) {
      console.error('포트폴리오 생성 실패:', err);
      setError(err.message);
      alert('포트폴리오 생성에 실패했습니다: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  // 포트폴리오 삭제
  const handleDeletePortfolio = async (portfolioId, e) => {
    e.stopPropagation(); // 이벤트 버블링 방지
    if (!confirm('정말로 이 포트폴리오를 삭제하시겠습니까?')) {
      return;
    }

    try {
      await portfolioService.deletePortfolio(portfolioId);
      setPortfolios(prev => prev.filter(p => p.id !== portfolioId));
      alert('포트폴리오가 삭제되었습니다.');
    } catch (err) {
      console.error('포트폴리오 삭제 실패:', err);
      alert('포트폴리오 삭제에 실패했습니다: ' + err.message);
    }
  };

  // 포트폴리오 클릭 시 상세 화면으로 이동
  const handlePortfolioClick = (portfolio) => {
    setSelectedPortfolio(portfolio);
    setShowPortfolioDetail(true);
  };

  // 실시간 AI 분석 결과를 기반으로 표시할 내용
  const getAIAnalysisDisplay = () => {
    // 실시간 AI 분석 결과가 있으면 사용
    if (aiAnalysisResult) {
      // 실시간 분석 결과
      if (aiAnalysisResult.type === 'realtime' && aiAnalysisResult.buySignal) {
        const signal = aiAnalysisResult.buySignal;
        return {
          recommendation: signal.recommendation || '분석 중',
          confidence: `${(signal.confidence * 100).toFixed(1)}%`,
          signalScore: signal.signalScore,
          analysis: `🔴 실시간 AI 분석: 신호 점수 ${signal.signalScore?.toFixed(1)}/100`,
          keyPoints: [
            `📊 모멘텀(20일): ${signal.technicalIndicators?.momentum20d?.toFixed(2) || 'N/A'}%`,
            `📈 변동성(20일): ${signal.technicalIndicators?.volatility?.toFixed(2) || 'N/A'}%`,
            signal.shouldBuy ? '✅ 매수 추천' : '⏸️ 대기 추천'
          ],
          riskFactors: signal.fundamentals ? [
            `VIX: ${signal.fundamentals.vix?.toFixed(1)}`,
            `P/E: ${signal.fundamentals.peRatio?.toFixed(1)}`,
            `52주 고점 대비: ${(signal.fundamentals.ratio52wHigh * 100).toFixed(1)}%`
          ] : [],
          isRealtime: true,
          lastUpdate: aiAnalysisResult.timestamp
        };
      }
      // 수동 분석 결과
      else {
        return {
          recommendation: aiAnalysisResult.recommendation || '분석 중',
          confidence: `${(aiAnalysisResult.confidence * 100).toFixed(1)}%`,
          signalScore: aiAnalysisResult.signalScore,
          analysis: aiAnalysisResult.type === 'buy' 
            ? `AI 매수 신호 분석: 신호 점수 ${aiAnalysisResult.signalScore?.toFixed(1)}/100`
            : `AI 매도 신호 분석: 신호 점수 ${aiAnalysisResult.signalScore?.toFixed(1)}/100`,
          keyPoints: aiAnalysisResult.buyRecommendation 
            ? [`신호 강도: ${aiAnalysisResult.buyRecommendation.signal_strength}`,
               `리스크 수준: ${aiAnalysisResult.buyRecommendation.risk_level}`,
               `추천 포지션: ${aiAnalysisResult.buyRecommendation.suggested_position_size}`]
            : [],
          riskFactors: aiAnalysisResult.fundamentals 
            ? [`VIX: ${aiAnalysisResult.fundamentals.vix?.toFixed(1)}`,
               `변동성: ${(aiAnalysisResult.technicalIndicators?.volatility * 100).toFixed(2)}%`]
            : [],
          isRealtime: false
        };
      }
    }
    
    // AI 분석 결과가 없으면 기본 메시지
    return {
      recommendation: '실시간 분석 대기',
      confidence: '0%',
      signalScore: 0,
      analysis: '🔄 실시간 AI 분석이 자동으로 시작됩니다...',
      keyPoints: ['실시간 데이터 기반 분석', 'Yahoo Finance 연동', '5초마다 자동 업데이트'],
      riskFactors: ['시장 변동성 확인 필요'],
      isRealtime: false
    };
  };

  const currentAnalysis = getAIAnalysisDisplay();

  const handleStartInvestment = () => {
    setIsActiveInvestment(true);
    setShowPopup(false);
  };

  const handleCancelInvestment = () => {
    setIsActiveInvestment(false);
    setShowPopup(false);
  };

  const handleReturnToHome = () => {
    setIsActiveInvestment(false);
    setSelectedStock('Apple');
    setSelectedPortfolio(null);
  };

  const handleStockSelect = (stockName) => {
    setSelectedStock(stockName);
    // aiTradingService를 통해 심볼 가져오기
    const symbol = aiTradingService.getTickerSymbol(stockName);
    setSelectedSymbol(symbol);
    setAiAnalysisResult(null); // 종목 변경 시 이전 분석 결과 초기화
  };

  // LiveQuotesList에서 주가 데이터를 받아오기 위한 콜백
  const handleStockData = (stockData) => {
    // stockData는 { symbol, price, change } 형태
    console.log('선택된 주식 데이터:', stockData);
    setCurrentStockData(stockData);
    
    // 실시간 AI 분석 트리거 (자동)
    if (aiInitialized && stockData && stockData.symbol) {
      triggerRealtimeAIAnalysis(stockData.symbol);
    }
  };
  
  // 실시간 AI 분석 함수 (백그라운드 자동 실행)
  const triggerRealtimeAIAnalysis = async (symbol) => {
    // 이미 분석 중이면 스킵
    if (isAnalyzing) return;
    
    try {
      console.log(`🤖 실시간 AI 분석 시작: ${symbol}`);
      
      // 병렬로 매수/매도 신호 분석
      const [buyAnalysis, sellAnalysis] = await Promise.allSettled([
        // 매수 신호 분석
        aiTradingService.analyzeBuySignal(symbol, 5.0),
        // 매도 신호 분석 (임시 데이터 - 실제로는 포트폴리오에서 가져와야 함)
        aiTradingService.quickPriceCheck ? 
          aiTradingService.quickPriceCheck(symbol) : 
          Promise.resolve(null)
      ]);
      
      // 분석 결과 실시간 업데이트
      if (buyAnalysis.status === 'fulfilled') {
        const buyResult = buyAnalysis.value;
        setAiAnalysisResult(prev => ({
          ...prev,
          type: 'realtime',
          ticker: symbol,
          buySignal: {
            recommendation: buyResult.recommendation,
            signalScore: buyResult.signalScore,
            confidence: buyResult.confidence,
            shouldBuy: buyResult.shouldBuy,
            technicalIndicators: buyResult.technicalIndicators,
            fundamentals: buyResult.fundamentals
          },
          timestamp: new Date().toISOString()
        }));
      }
      
      console.log(`✅ 실시간 AI 분석 완료: ${symbol}`);
    } catch (error) {
      console.error('실시간 AI 분석 실패:', error);
    }
  };
  
  // 매수 버튼 핸들러 (실시간 AI 분석 결과 기반)
  const handleBuyClick = async () => {
    if (!aiInitialized) {
      alert('AI 모델이 아직 초기화되지 않았습니다. 잠시 후 다시 시도해주세요.');
      return;
    }
    
    // 실시간 분석 결과가 있으면 그것을 사용
    if (aiAnalysisResult && aiAnalysisResult.type === 'realtime' && aiAnalysisResult.buySignal) {
      const signal = aiAnalysisResult.buySignal;
      
      const confirmBuy = confirm(
        `🤖 실시간 AI 분석 결과\n\n` +
        `종목: ${selectedStock} (${selectedSymbol})\n` +
        `추천: ${signal.recommendation}\n` +
        `신호 점수: ${signal.signalScore.toFixed(1)}/100\n` +
        `신뢰도: ${(signal.confidence * 100).toFixed(1)}%\n` +
        `현재가: $${currentStockData?.price?.toFixed(2) || 'N/A'}\n\n` +
        `매수를 진행하시겠습니까?`
      );
      
      if (confirmBuy) {
        // TODO: 실제 매수 로직 구현
        alert('매수 주문이 접수되었습니다.');
      }
    } else {
      // 실시간 분석이 없으면 새로 분석 요청
      alert('실시간 AI 분석 중입니다. 잠시 후 다시 시도해주세요.');
      await triggerRealtimeAIAnalysis(selectedSymbol);
    }
  };
  
  // 매도 버튼 핸들러
  const handleSellClick = async () => {
    if (!aiInitialized) {
      alert('AI 모델이 아직 초기화되지 않았습니다. 잠시 후 다시 시도해주세요.');
      return;
    }
    
    // TODO: 실제 보유 정보를 가져와야 함
    const entryPrice = prompt('매수 가격을 입력하세요:');
    if (!entryPrice) return;
    
    const entryDate = prompt('매수 날짜를 입력하세요 (YYYY-MM-DD):');
    if (!entryDate) return;
    
    setIsAnalyzing(true);
    try {
      console.log(`매도 분석 시작: ${selectedSymbol} (${selectedStock})`);
      const result = await aiTradingService.analyzeSellSignal(
        selectedSymbol,
        parseFloat(entryPrice),
        entryDate,
        100
      );
      
      setAiAnalysisResult({
        type: 'sell',
        ...result
      });
      
      // 매도 추천인 경우 알림
      if (result.shouldSell) {
        const confirmSell = confirm(
          `AI 분석 결과: ${result.recommendation}\n` +
          `신호 점수: ${result.signalScore.toFixed(1)}/100\n` +
          `현재 수익률: ${result.currentReturn}\n` +
          `신뢰도: ${(result.confidence * 100).toFixed(1)}%\n\n` +
          `매도를 진행하시겠습니까?`
        );
        
        if (confirmSell) {
          // TODO: 실제 매도 로직 구현
          alert('매도 주문이 접수되었습니다.');
        }
      } else {
        alert(
          `AI 분석 결과: ${result.recommendation}\n` +
          `신호 점수: ${result.signalScore.toFixed(1)}/100\n` +
          `현재 수익률: ${result.currentReturn}\n` +
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

  // 주가 정보 포맷팅
  const formatStockPrice = (stockData) => {
    if (!stockData || !stockData.price) return '로딩 중...';
    return `$${stockData.price.toLocaleString()}`;
  };

  const formatStockChange = (stockData) => {
    if (!stockData) return '로딩 중...';
    
    // 달러 변화량 우선 사용
    if (stockData.dollarChange !== null && stockData.dollarChange !== undefined) {
      const sign = stockData.dollarChange > 0 ? '+' : (stockData.dollarChange < 0 ? '-' : '');
      return `${sign}$${Math.abs(stockData.dollarChange).toFixed(2)}`;
    }
    
    // 퍼센트 변화량 사용
    if (typeof stockData.change === 'number') {
      const sign = stockData.change > 0 ? '+' : (stockData.change < 0 ? '-' : '');
      return `${sign}${Math.abs(stockData.change).toFixed(2)}%`;
    }
    
    return '로딩 중...';
  };

  const getChangeClass = (stockData) => {
    if (!stockData) return '';
    
    // 달러 변화량 우선 사용
    if (stockData.dollarChange !== null && stockData.dollarChange !== undefined) {
      return stockData.dollarChange >= 0 ? 'positive' : 'negative';
    }
    
    // 퍼센트 변화량 사용
    if (typeof stockData.change === 'number') {
      return stockData.change >= 0 ? 'positive' : 'negative';
    }
    
    return '';
  };

  // 포트폴리오 상세 화면 표시
  if (showPortfolioDetail && selectedPortfolio) {
    return (
      <PortfolioDetail 
        portfolio={selectedPortfolio}
        user={user}
        onBack={() => {
          setShowPortfolioDetail(false);
          setSelectedPortfolio(null);
        }}
      />
    );
  }
  
  // 기존 포트폴리오 목록 화면
  if (!isActiveInvestment) {
    return (
      <div className="tab-content">
        <div className="content-header">
          <h1>모의투자</h1>
          <p>기존 포트폴리오와 새로운 모의투자 시작</p>
        </div>
        
        <div className="content-grid">
          {/* 기존 포트폴리오 목록 */}
          <div className="content-card">
            <h3>내 포트폴리오 목록</h3>
            
            {error && (
              <div className="error-message">
                <p>{error}</p>
              </div>
            )}
            
            {isLoading ? (
              <div className="loading-message">
                <p>포트폴리오를 불러오는 중...</p>
              </div>
            ) : portfolios.length === 0 ? (
              <div className="empty-portfolio">
                <p>아직 생성된 포트폴리오가 없습니다.</p>
                <p>새로운 모의투자를 시작해보세요!</p>
              </div>
            ) : (
            <div className="portfolio-list">
                {portfolios.map((portfolio) => (
                  <div 
                    key={portfolio.id} 
                    className="portfolio-item"
                    onClick={() => handlePortfolioClick(portfolio)}
                    style={{ cursor: 'pointer' }}
                  >
                  <div className="portfolio-header">
                    <h4>{portfolio.name}</h4>
                      <div className="portfolio-actions">
                        <button 
                          className="delete-btn"
                          onClick={(e) => handleDeletePortfolio(portfolio.id, e)}
                          title="포트폴리오 삭제"
                        >
                          삭제
                        </button>
                      </div>
                  </div>
                  <div className="portfolio-details">
                    <div className="portfolio-info">
                        <span>생성일: {new Date(portfolio.created_at).toLocaleDateString()}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            )}
            
            {/* 모의투자 시작 버튼 */}
             <div className="start-investment-container">
               <button 
                 className="start-investment-btn"
                 onClick={() => setShowPopup(true)}
                disabled={isLoading}
               >
                 새 모의투자 시작하기
               </button>
             </div>
          </div>
        </div>

                 {/* 모의투자 시작 팝업 */}
         {showPopup && (
           <div className="popup-overlay">
             <div className="investment-popup">
               <h3>새로운 모의투자 설정</h3>
               
               <div className="investment-settings">
                 <div className="setting-group">
                   <label>포트폴리오 제목</label>
                   <input
                     type="text"
                     placeholder="포트폴리오 제목을 입력하세요"
                     value={investmentSettings.portfolioTitle}
                     onChange={e => setInvestmentSettings({
                       ...investmentSettings,
                       portfolioTitle: e.target.value
                     })}
                     className="asset-input"
                    disabled={isLoading}
                   />
                 </div>
               </div>

               <div className="popup-buttons">
                <button 
                  className="cancel-btn" 
                  onClick={handleCancelInvestment}
                  disabled={isLoading}
                >
                   취소
                 </button>
                <button 
                  className="start-btn" 
                  onClick={handleCreatePortfolio}
                  disabled={isLoading}
                >
                  {isLoading ? '생성 중...' : '포트폴리오 생성'}
                 </button>
               </div>
             </div>
           </div>
         )}
      </div>
    );
  }

  // 활성 모의투자 화면
  return (
    <div className="tab-content">
      <div className="content-header">
        <div className="header-with-button">
          <div className="header-content">
            <h1>모의투자 진행중</h1>
            <p>실시간 시장 데이터를 기반으로 한 모의 투자 환경</p>
            {selectedPortfolio && (
              <p style={{ fontSize: '14px', color: '#666', marginTop: '4px' }}>
                포트폴리오: {selectedPortfolio.name}
              </p>
            )}
          </div>
          <button className="home-button" onClick={handleReturnToHome}>
            모의투자 홈
          </button>
        </div>
      </div>
      
      <div className="investment-layout">
        {/* 왼쪽: AI 분석 영역 */}
        <div className="ai-analysis-section">
          <div className="content-card">
            <h3>AI 투자 분석</h3>
            <div className="selected-stock-info">
              <h4>{selectedStock}</h4>
              <div className="stock-price-info">
                <span className="current-price">{formatStockPrice(currentStockData)}</span>
                <span className={`price-change ${getChangeClass(currentStockData)}`}>
                  {formatStockChange(currentStockData)}
                </span>
              </div>
            </div>
            
            <div className="ai-recommendation">
              <div className="recommendation-badge">
                <span className="recommendation-text">{currentAnalysis.recommendation}</span>
                <span className="confidence-score">신뢰도: {currentAnalysis.confidence}</span>
                {currentAnalysis.isRealtime && (
                  <span style={{
                    fontSize: '10px',
                    color: '#00ff00',
                    marginLeft: '10px',
                    animation: 'pulse 1s infinite'
                  }}>
                    ● LIVE
                  </span>
                )}
              </div>
              {currentAnalysis.lastUpdate && (
                <div style={{ fontSize: '11px', color: '#888', marginTop: '4px' }}>
                  마지막 업데이트: {new Date(currentAnalysis.lastUpdate).toLocaleTimeString()}
                </div>
              )}
            </div>
            
            <div className="analysis-content">
              <h5>투자 분석</h5>
              <p>{currentAnalysis.analysis}</p>
              
              <h5>주요 포인트</h5>
              <ul className="key-points">
                {currentAnalysis.keyPoints.map((point, index) => (
                  <li key={index}>{point}</li>
                ))}
              </ul>
              
              <h5>위험 요소</h5>
              <ul className="risk-factors">
                {currentAnalysis.riskFactors.map((risk, index) => (
                  <li key={index}>{risk}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
        
        {/* 오른쪽: 실시간 주식 리스트뷰 */}
        <div className="stock-list-section">
          <div className="content-card">
            <h3>실시간 주식 목록</h3>
            
            {/* 실시간 주식 리스트 */}
            <div className="stock-list-container" style={{ maxHeight: '600px', overflowY: 'auto' }}>
              <LiveQuotesList 
                onStockSelect={handleStockSelect}
                selectedStock={selectedStock}
                onStockData={handleStockData}
              />
            </div>
            
            {/* 매수/매도 버튼 */}
            <div className="trading-buttons">
              <button 
                className="buy-button"
                onClick={handleBuyClick}
                disabled={isAnalyzing}
              >
                {isAnalyzing ? 'AI 분석 중...' : '매수'}
              </button>
              <button 
                className="sell-button"
                onClick={handleSellClick}
                disabled={isAnalyzing}
              >
                {isAnalyzing ? 'AI 분석 중...' : '매도'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MockInvestment; 