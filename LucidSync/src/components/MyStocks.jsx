import React, { useState, useEffect } from 'react';
import LiveQuotesList from './LiveQuotesList';
import './TabContent.css';

const MyStocks = ({ user }) => {
  const [listViewTab, setListViewTab] = useState('myStocks');
  const [selectedStock, setSelectedStock] = useState('Apple');
  const [currentStockData, setCurrentStockData] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [newsAnalysis, setNewsAnalysis] = useState(null);
  const [aggregateNews, setAggregateNews] = useState(null);
  const [newsLoading, setNewsLoading] = useState(false);
  
  // 보유 주식 데이터 (더미 데이터)
  const myStocks = [
    { symbol: 'AAPL', name: 'Apple', quantity: 10, avgPrice: 150.00, currentPrice: 175.43, change: 2.15 },
    { symbol: 'MSFT', name: 'Microsoft', quantity: 5, avgPrice: 320.00, currentPrice: 338.11, change: -0.87 },
    { symbol: 'GOOGL', name: 'Alphabet', quantity: 8, avgPrice: 135.00, currentPrice: 142.56, change: 1.23 },
    { symbol: 'AMZN', name: 'Amazon', quantity: 12, avgPrice: 140.00, currentPrice: 145.24, change: 3.45 },
    { symbol: 'NVDA', name: 'NVIDIA', quantity: 6, avgPrice: 450.00, currentPrice: 485.09, change: 5.67 }
  ];

  // 검색 필터링 (보유 주식만)
  const filteredMyStocks = myStocks.filter(stock => {
    const matchesSearch = stock.name.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesSearch;
  });

  // 선택된 주식의 보유 데이터 찾기
  const selectedStockData = myStocks.find(stock => stock.name === selectedStock);

  const handleStockSelect = (stockName) => {
    setSelectedStock(stockName);
    // 주식 선택 시 뉴스 감정 분석도 가져오기
    fetchNewsSentiment(stockName);
  };

  // 뉴스 감정 분석 가져오기
  const fetchNewsSentiment = async (stockName) => {
    setNewsLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/api/news/sentiment/analyze-stock`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          stock_name: stockName,
          news_limit: 10,
          max_days: 14
        })
      });

      if (response.ok) {
        const result = await response.json();
        if (result.success) {
          setNewsAnalysis(result.data);
        } else {
          setNewsAnalysis({ error: '뉴스 분석 실패' });
        }
      } else {
        setNewsAnalysis({ error: '서버 응답 오류' });
      }
    } catch (error) {
      console.error('뉴스 감정 분석 오류:', error);
      setNewsAnalysis({ error: '네트워크 오류' });
    } finally {
      setNewsLoading(false);
    }
  };

  // 컴포넌트 마운트 시 초기 뉴스 감정 분석 가져오기
  useEffect(() => {
    fetchNewsSentiment(selectedStock);
  }, []);

  // LiveQuotesList에서 주가 데이터를 받아오기 위한 콜백
  const handleStockData = (stockData) => {
    console.log('선택된 주식 데이터:', stockData);
    setCurrentStockData(stockData);
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

  // 보유 주식 리스트 렌더링
  const renderMyStocksList = () => (
    <>
      {/* 검색 입력 */}
      <div className="search-container">
        <input
          type="text"
          placeholder="주식명으로 검색..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="search-input"
        />
      </div>
      
      {/* 보유 주식 리스트 */}
      <div className="stock-list-container" style={{ maxHeight: '600px', overflowY: 'auto' }}>
        {filteredMyStocks.map((stock) => (
          <div
            key={stock.symbol}
            className={`stock-list-item ${selectedStock === stock.name ? 'selected' : ''}`}
            onClick={() => handleStockSelect(stock.name)}
          >
            <div className="stock-info">
              <div className="stock-name-sector">
                <span className="stock-name">{stock.name}</span>
                <span className="stock-sector">보유: {stock.quantity}주</span>
              </div>
              <div className="stock-price-change">
                <span className="stock-price">${stock.currentPrice.toLocaleString()}</span>
                <span className={`stock-change ${stock.change >= 0 ? 'positive' : 'negative'}`}>
                  {stock.change > 0 ? '+' : ''}{stock.change.toFixed(2)}%
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </>
  );

  // 전체 주식 리스트 렌더링 (실시간 데이터)
  const renderAllStocksList = () => (
    <>
      {/* 실시간 주식 리스트 */}
      <div className="stock-list-container" style={{ maxHeight: '600px', overflowY: 'auto' }}>
        <LiveQuotesList 
          onStockSelect={handleStockSelect}
          selectedStock={selectedStock}
          onStockData={handleStockData}
        />
      </div>
    </>
  );

  // AI 분석 데이터 (미국주)
  const aiAnalysis = {
    Apple: {
      recommendation: '매수',
      confidence: '85%',
      analysis: 'Apple은 iPhone, Mac, iPad 등 프리미엄 제품 포트폴리오와 서비스 사업의 지속적인 성장으로 강력한 수익성을 보이고 있습니다. AI 기술 도입과 인도, 중국 등 신흥시장 진출이 주요 성장 동력입니다.',
      keyPoints: [
        'iPhone 15 시리즈 성공적 출시',
        '서비스 사업 매출 비중 확대',
        'AI 기술 도입 가속화',
        '신흥시장 진출 확대'
      ],
      riskFactors: [
        '중국 시장 의존도',
        '반도체 공급망 불확실성'
      ]
    },
    Microsoft: {
      recommendation: '매수',
      confidence: '90%',
      analysis: 'Microsoft는 Azure 클라우드 서비스와 AI 기술의 선도적 위치로 인해 지속적인 성장이 예상됩니다. OpenAI와의 파트너십과 Copilot 서비스가 주요 호재로 작용합니다.',
      keyPoints: [
        'Azure 클라우드 서비스 성장',
        'AI 기술 선도적 위치',
        'OpenAI 파트너십 강화',
        'Copilot 서비스 확대'
      ],
      riskFactors: [
        '클라우드 경쟁 심화',
        '규제 환경 변화'
      ]
    },
    NVIDIA: {
      recommendation: '매수',
      confidence: '95%',
      analysis: 'AI 칩 수요 급증과 데이터센터 시장 확대로 NVIDIA는 가장 강력한 성장 동력을 보이고 있습니다. H100, H200 등 차세대 AI 칩의 독점적 우위가 주요 강점입니다.',
      keyPoints: [
        'AI 칩 시장 독점적 우위',
        '데이터센터 수요 급증',
        '차세대 H200 칩 출시',
        '게임 시장 회복세'
      ],
      riskFactors: [
        '반도체 사이클 변동성',
        '경쟁사 기술 발전'
      ]
    },
    Tesla: {
      recommendation: '관망',
      confidence: '70%',
      analysis: '전기차 시장의 선도적 위치는 유지하지만, 가격 경쟁과 수요 둔화로 인한 불확실성이 존재합니다. FSD 기술과 로봇택시 사업이 장기 성장 동력입니다.',
      keyPoints: [
        '전기차 시장 선도적 위치',
        'FSD 기술 개발',
        '로봇택시 사업 확장',
        '글로벌 생산능력 확대'
      ],
      riskFactors: [
        '가격 경쟁 심화',
        '수요 둔화 우려'
      ]
    },
    Amazon: {
      recommendation: '매수',
      confidence: '80%',
      analysis: 'AWS 클라우드 서비스와 전자상거래 사업의 안정적 성장으로 Amazon은 지속적인 수익성 향상이 예상됩니다. AI 기술 투자와 광고 사업 확대가 주요 성장 동력입니다.',
      keyPoints: [
        'AWS 클라우드 서비스 성장',
        '전자상거래 시장 점유율 확대',
        'AI 기술 투자 확대',
        '광고 사업 성장'
      ],
      riskFactors: [
        '규제 환경 변화',
        '노동 비용 상승'
      ]
    }
  };

  const currentAnalysis = aiAnalysis[selectedStock] || aiAnalysis['Apple'];

  return (
    <div className="tab-content">
      <div className="content-header">
        <h1>나의 주식</h1>
        <p>보유 주식 현황과 포트폴리오 분석</p>
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
              {listViewTab === 'myStocks' && selectedStockData && (
                <div className="holding-details">
                  <div className="holding-summary">
                    <span>보유: {selectedStockData.quantity}주</span>
                    <span>평균단가: ${selectedStockData.avgPrice.toLocaleString()}</span>
                    <span className={`profit ${selectedStockData.change >= 0 ? 'positive' : 'negative'}`}>
                      수익률: {selectedStockData.change > 0 ? '+' : ''}{selectedStockData.change.toFixed(2)}%
                    </span>
                  </div>
                </div>
              )}
            </div>
            
            <div className="ai-recommendation">
              <div className="recommendation-badge">
                <span className="recommendation-text">{currentAnalysis.recommendation}</span>
                <span className="confidence-score">신뢰도: {currentAnalysis.confidence}</span>
              </div>
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
        
        {/* 오른쪽: 주식 리스트뷰 */}
        <div className="stock-list-section">
          <div className="content-card">
            <h3>주식 목록</h3>
            
            {/* 리스트뷰 탭 네비게이션 */}
            <div className="list-view-tabs">
              <button
                className={`list-tab ${listViewTab === 'myStocks' ? 'active' : ''}`}
                onClick={() => setListViewTab('myStocks')}
              >
                보유주식
              </button>
              <button
                className={`list-tab ${listViewTab === 'allStocks' ? 'active' : ''}`}
                onClick={() => setListViewTab('allStocks')}
              >
                전체주식
              </button>
            </div>
            
            {/* 탭에 따른 리스트 렌더링 */}
            {listViewTab === 'myStocks' ? renderMyStocksList() : renderAllStocksList()}
            
            {/* 매수/매도 버튼 */}
            <div className="trading-buttons">
              <button className="buy-button">
                매수
              </button>
              <button className="sell-button">
                매도
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MyStocks; 