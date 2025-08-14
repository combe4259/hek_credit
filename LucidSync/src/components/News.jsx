import React, { useState, useEffect } from 'react';
import './TabContent.css';

const News = () => {
  const [selectedStock, setSelectedStock] = useState('Apple');
  const [searchTerm, setSearchTerm] = useState('');
  const [newsData, setNewsData] = useState([]);
  const [newsLoading, setNewsLoading] = useState(false);
  const [aggregateNews, setAggregateNews] = useState(null);

  // 주식 데이터
  const stocks = [
    { name: 'Apple', price: '$175.43', change: '+2.1%', changeType: 'positive', sector: 'Technology' },
    { name: 'Microsoft', price: '$338.11', change: '-0.87%', changeType: 'negative', sector: 'Technology' },
    { name: 'NVIDIA', price: '$485.09', change: '+5.67%', changeType: 'positive', sector: 'Technology' },
    { name: 'Alphabet', price: '$142.56', change: '+1.23%', changeType: 'positive', sector: 'Technology' },
    { name: 'Amazon', price: '$145.24', change: '+3.45%', changeType: 'positive', sector: 'E-commerce' },
    { name: 'Tesla', price: '$235.87', change: '-2.34%', changeType: 'negative', sector: 'Automotive' },
    { name: 'Meta', price: '$312.44', change: '+1.87%', changeType: 'positive', sector: 'Technology' },
    { name: 'Netflix', price: '$487.23', change: '+0.92%', changeType: 'positive', sector: 'Entertainment' },
    { name: 'Intel', price: '$23.45', change: '-1.45%', changeType: 'negative', sector: 'Technology' },
    { name: 'AMD', price: '$142.78', change: '+2.67%', changeType: 'positive', sector: 'Technology' }
  ];

  // 뉴스 감정 분석 가져오기
  const fetchNewsSentiment = async (stockName) => {
    setNewsLoading(true);
    try {
      // 개별 뉴스 분석과 종합 점수를 병렬로 가져오기
      const [newsResponse, aggregateResponse] = await Promise.allSettled([
        fetch(`http://localhost:8000/api/ai/realtime/news-sentiment/${stockName}?limit=10`),
        fetch(`http://localhost:8000/api/ai/realtime/news-sentiment/aggregate/${stockName}?max_days=14`)
      ]);

      // 개별 뉴스 분석 결과 처리
      if (newsResponse.status === 'fulfilled' && newsResponse.value.ok) {
        const newsResult = await newsResponse.value.json();
        setNewsData(newsResult.status === 'success' ? newsResult.news_analysis : []);
      } else {
        setNewsData([]);
      }

      // 종합 감정 점수 처리
      if (aggregateResponse.status === 'fulfilled' && aggregateResponse.value.ok) {
        const aggregateResult = await aggregateResponse.value.json();
        setAggregateNews(aggregateResult.status === 'success' ? aggregateResult.aggregate_analysis : null);
      } else {
        setAggregateNews(null);
      }

    } catch (error) {
      console.error('뉴스 감정 분석 오류:', error);
      setNewsData([]);
      setAggregateNews(null);
    } finally {
      setNewsLoading(false);
    }
  };

  // 주식 선택 핸들러
  const handleStockSelect = (stockName) => {
    setSelectedStock(stockName);
    fetchNewsSentiment(stockName);
  };

  // 컴포넌트 마운트 시 초기 뉴스 가져오기
  useEffect(() => {
    fetchNewsSentiment(selectedStock);
  }, []);

  // 검색 필터링
  const filteredStocks = stocks.filter(stock =>
    stock.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    stock.sector.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // 뉴스 데이터를 긍정/부정으로 분류하는 함수
  const categorizeNewsByScore = (newsArray) => {
    const positive = [];
    const negative = [];
    
    newsArray.forEach(news => {
      // API 응답에서 impact_score_100 또는 sentiment_score 사용
      const score = news.impact_score_100 || news.sentiment_score || 50;
      const prediction = news.prediction || 'NEUTRAL';
      
      if (prediction === 'POSITIVE' || (prediction === 'NEUTRAL' && score >= 50)) {
        positive.push({
          category: news.category || '일반',
          title: news.title,
          url: news.url || '#',
          time: news.time || news.news_date || '시간 정보 없음',
          source: news.source || '출처 미상',
          score: score,
          prediction: prediction
        });
      } else {
        negative.push({
          category: news.category || '일반',
          title: news.title,
          url: news.url || '#',
          time: news.time || news.news_date || '시간 정보 없음',
          source: news.source || '출처 미상',
          score: score,
          prediction: prediction
        });
      }
    });
    
    return { positive, negative };
  };

  // 현재 선택된 주식의 뉴스 분류
  const categorizedNews = categorizeNewsByScore(newsData);
  const currentPositiveNews = categorizedNews.positive;
  const currentNegativeNews = categorizedNews.negative;

  return (
    <div className="tab-content">
      <div className="content-header">
        <h1>뉴스</h1>
        <p>주식 시장 동향 및 투자 관련 뉴스</p>
      </div>

      <div className="investment-layout">
        {/* 왼쪽: 뉴스 영역 */}
        <div className="news-section">
          <div className="content-card">
            <h3>{selectedStock} 관련 뉴스</h3>
            
            {/* 종합 감정 점수 표시 */}
            {aggregateNews && (
              <div className="content-card aggregate-sentiment">
                <h4>종합 감정 분석</h4>
                <div className="aggregate-info">
                  <div className="aggregate-score">
                    <span className="score-label">평균 감정 점수:</span>
                    <span className={`score-value ${aggregateNews.avg_sentiment >= 0 ? 'positive' : 'negative'}`}>
                      {aggregateNews.avg_sentiment?.toFixed(2)}
                    </span>
                  </div>
                  <div className="news-count">
                    <span>분석된 뉴스: {aggregateNews.news_count}개</span>
                  </div>
                  {aggregateNews.trend && (
                    <div className="sentiment-trend">
                      <span>감정 추세: {aggregateNews.trend}</span>
                    </div>
                  )}
                </div>
              </div>
            )}
            
            <div className="content-card news-featured">
              <h4>주요 뉴스</h4>
              <div className="featured-news">
                {newsLoading ? (
                  <div className="loading-message">뉴스 로딩 중...</div>
                ) : newsData.length > 0 ? (
                  <div className="news-item featured">
                    <div className="news-category">{newsData[0]?.category || '일반'}</div>
                    <h5>
                      <a 
                        href={newsData[0]?.url || '#'} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="news-title-link"
                      >
                        {newsData[0]?.title || `${selectedStock} 관련 뉴스`}
                      </a>
                    </h5>
                    <div className="news-meta">
                      <span className="news-time">{newsData[0]?.time || newsData[0]?.published_date || '최근'}</span>
                      <span className="news-source">{newsData[0]?.source || '뉴스 출처'}</span>
                      {(newsData[0]?.impact_score_100 !== undefined || newsData[0]?.sentiment_score !== undefined) && (
                        <span className={`sentiment-score ${(newsData[0].impact_score_100 || newsData[0].sentiment_score) >= 50 ? 'positive' : 'negative'}`}>
                          점수: {(newsData[0].impact_score_100 || newsData[0].sentiment_score || 0).toFixed(1)}/100
                        </span>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="news-item featured">
                    <div className="news-category">시장 동향</div>
                    <h5>{selectedStock} 관련 뉴스를 불러오는 중입니다.</h5>
                    <p>뉴스 데이터를 분석하고 있습니다. 잠시만 기다려 주세요.</p>
                    <div className="news-meta">
                      <span className="news-time">업데이트 중</span>
                      <span className="news-source">시스템</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
            
            <div className="content-card">
              <h4>호재 뉴스 ({currentPositiveNews.length})</h4>
              {newsLoading ? (
                <div className="loading-message">뉴스 분석 중...</div>
              ) : (
                <div className="news-list">
                  {currentPositiveNews.length > 0 ? (
                    currentPositiveNews.map((news, index) => (
                      <div key={index} className="news-item">
                        <div className="news-category">{news.category}</div>
                        <h5>
                          <a 
                            href={news.url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="news-title-link"
                          >
                            {news.title}
                          </a>
                        </h5>
                        <div className="news-meta">
                          <span className="news-time">{news.time}</span>
                          <span className="news-source">{news.source}</span>
                          {news.score !== undefined && (
                            <span className={`sentiment-score ${news.score >= 50 ? 'positive' : 'negative'}`}>
                              점수: {news.score.toFixed(1)}/100
                            </span>
                          )}
                          {news.prediction && (
                            <span className={`prediction-badge ${news.prediction.toLowerCase()}`}>
                              {news.prediction}
                            </span>
                          )}
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="no-news-message">호재 뉴스가 없습니다.</div>
                  )}
                </div>
              )}
            </div>
            
            <div className="content-card">
              <h4>악재 뉴스 ({currentNegativeNews.length})</h4>
              {newsLoading ? (
                <div className="loading-message">뉴스 분석 중...</div>
              ) : (
                <div className="news-list">
                  {currentNegativeNews.length > 0 ? (
                    currentNegativeNews.map((news, index) => (
                      <div key={index} className="news-item">
                        <div className="news-category">{news.category}</div>
                        <h5>
                          <a 
                            href={news.url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="news-title-link"
                          >
                            {news.title}
                          </a>
                        </h5>
                        <div className="news-meta">
                          <span className="news-time">{news.time}</span>
                          <span className="news-source">{news.source}</span>
                          {news.score !== undefined && (
                            <span className={`sentiment-score ${news.score >= 50 ? 'positive' : 'negative'}`}>
                              점수: {news.score.toFixed(1)}/100
                            </span>
                          )}
                          {news.prediction && (
                            <span className={`prediction-badge ${news.prediction.toLowerCase()}`}>
                              {news.prediction}
                            </span>
                          )}
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="no-news-message">악재 뉴스가 없습니다.</div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* 오른쪽: 주식 리스트뷰 */}
        <div className="stock-list-section">
          <div className="content-card">
            <h3>주식 목록</h3>

            {/* 검색바 */}
            <div className="search-container">
              <input
                type="text"
                placeholder="주식명 또는 섹터로 검색..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="stock-search"
              />
            </div>

            {/* 주식 리스트 */}
            <div className="stock-list-container">
              {filteredStocks.map((stock, index) => (
                <div
                  key={index}
                  className={`stock-list-item ${selectedStock === stock.name ? 'selected' : ''}`}
                  onClick={() => handleStockSelect(stock.name)}
                >
                  <div className="stock-info">
                    <div className="stock-name-sector">
                      <span className="stock-name">{stock.name}</span>
                      <span className="stock-sector">{stock.sector}</span>
                    </div>
                    <div className="stock-price-change">
                      <span className="stock-price">{stock.price}</span>
                      <span className={`stock-change ${stock.changeType}`}>
                        {stock.change}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default News; 