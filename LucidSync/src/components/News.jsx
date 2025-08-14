import React, { useState } from 'react';
import './TabContent.css';

const News = () => {
  const [selectedStock, setSelectedStock] = useState('삼성전자');
  const [searchTerm, setSearchTerm] = useState('');

  // 주식 데이터
  const stocks = [
    { name: '삼성전자', price: '₩75,000', change: '+2.1%', changeType: 'positive', sector: '반도체' },
    { name: 'SK하이닉스', price: '₩125,000', change: '-1.3%', changeType: 'negative', sector: '반도체' },
    { name: 'NAVER', price: '₩220,000', change: '+0.8%', changeType: 'positive', sector: 'IT' },
    { name: '카카오', price: '₩45,000', change: '+1.2%', changeType: 'positive', sector: 'IT' },
    { name: 'LG화학', price: '₩450,000', change: '-0.5%', changeType: 'negative', sector: '화학' },
    { name: '현대차', price: '₩180,000', change: '+3.2%', changeType: 'positive', sector: '자동차' },
    { name: '기아', price: '₩85,000', change: '+2.8%', changeType: 'positive', sector: '자동차' },
    { name: 'POSCO홀딩스', price: '₩380,000', change: '-0.8%', changeType: 'negative', sector: '철강' },
    { name: 'LG전자', price: '₩95,000', change: '+1.5%', changeType: 'positive', sector: '전자' },
    { name: 'KB금융', price: '₩55,000', change: '+0.3%', changeType: 'positive', sector: '금융' }
  ];

  // 검색 필터링
  const filteredStocks = stocks.filter(stock =>
    stock.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    stock.sector.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // 주식별 뉴스 데이터
  const stockNews = {
    '삼성전자': {
      positive: [
        {
          category: '기업 소식',
          title: '삼성전자, 차세대 메모리 개발 성공',
          content: '삼성전자가 차세대 메모리 반도체 개발에 성공하여 기술적 우위를 확보했다.',
          time: '2시간 전',
          source: '한국경제'
        },
        {
          category: '실적',
          title: '삼성전자 1분기 실적 예상치 상회',
          content: '삼성전자의 1분기 실적이 시장 예상치를 상회하며 반도체 업종 회복세를 확인했다.',
          time: '4시간 전',
          source: '매일경제'
        }
      ],
      negative: [
        {
          category: '경쟁',
          title: 'TSMC, 삼성전자와의 기술 격차 확대',
          content: 'TSMC가 삼성전자와의 기술 격차를 확대하고 있어 경쟁 우위에 대한 우려가 증가하고 있다.',
          time: '6시간 전',
          source: '서울경제'
        }
      ]
    },
    'SK하이닉스': {
      positive: [
        {
          category: '투자',
          title: 'SK하이닉스, AI 칩 생산 확대',
          content: 'SK하이닉스가 AI 칩 생산을 확대하여 새로운 성장 동력을 확보했다.',
          time: '3시간 전',
          source: '한국경제'
        }
      ],
      negative: [
        {
          category: '실적',
          title: 'SK하이닉스, 메모리 가격 하락으로 실적 악화',
          content: '메모리 반도체 가격 하락으로 SK하이닉스의 실적이 악화되고 있다.',
          time: '5시간 전',
          source: '매일경제'
        }
      ]
    },
    'NAVER': {
      positive: [
        {
          category: '기업 소식',
          title: 'NAVER, AI 기술 투자 확대 발표',
          content: 'NAVER가 AI 기술 개발을 위한 1조원 규모의 투자 계획을 발표했다.',
          time: '1시간 전',
          source: '매일경제'
        },
        {
          category: '사업 확장',
          title: 'NAVER, 글로벌 AI 시장 진출 가속화',
          content: 'NAVER가 글로벌 AI 시장 진출을 가속화하여 새로운 성장 기회를 모색하고 있다.',
          time: '7시간 전',
          source: '서울경제'
        }
      ],
      negative: [
        {
          category: '규제',
          title: 'NAVER, 플랫폼 규제 강화로 인한 부담 증가',
          content: '플랫폼 규제 강화로 NAVER의 사업 환경이 악화되고 있다.',
          time: '9시간 전',
          source: '한국경제'
        }
      ]
    },
    '카카오': {
      positive: [
        {
          category: '사업 확장',
          title: '카카오, 모빌리티 사업 확대',
          content: '카카오가 모빌리티 사업을 확대하여 새로운 수익원을 개발하고 있다.',
          time: '2시간 전',
          source: '매일경제'
        }
      ],
      negative: [
        {
          category: '실적',
          title: '카카오, 광고 수익 감소로 실적 부진',
          content: '광고 수익 감소로 카카오의 실적이 부진한 모습을 보이고 있다.',
          time: '4시간 전',
          source: '서울경제'
        }
      ]
    },
    'LG화학': {
      positive: [
        {
          category: '투자',
          title: 'LG화학, 배터리 사업 확대 투자',
          content: 'LG화학이 배터리 사업 확대를 위한 대규모 투자를 진행하고 있다.',
          time: '3시간 전',
          source: '한국경제'
        }
      ],
      negative: [
        {
          category: '원자재',
          title: 'LG화학, 원자재 가격 상승으로 마진 압박',
          content: '원자재 가격 상승으로 LG화학의 마진이 압박받고 있다.',
          time: '5시간 전',
          source: '매일경제'
        }
      ]
    }
  };

  // 기본 뉴스 데이터 (주식 선택되지 않았을 때)
  const defaultPositiveNews = [
    {
      category: '정책',
      title: '정부, 스타트업 투자 세제 혜택 확대',
      content: '정부가 스타트업 투자에 대한 세제 혜택을 확대하는 방안을 검토 중이다.',
      time: '6시간 전',
      source: '서울경제'
    },
    {
      category: '시장 동향',
      title: '코스피, 반도체주 상승으로 2,800선 회복',
      content: '삼성전자, SK하이닉스 등 반도체 관련주들의 상승세로 코스피가 2,800선을 회복했다.',
      time: '2시간 전',
      source: '한국경제'
    }
  ];

  const defaultNegativeNews = [
    {
      category: '글로벌',
      title: '미국 연준, 금리 인하 시점 모호',
      content: '미국 연방준비제도가 금리 인하 시점에 대해 모호한 입장을 보이고 있어 글로벌 시장의 불확실성이 증가하고 있다.',
      time: '8시간 전',
      source: '로이터'
    },
    {
      category: '경제',
      title: '원화 약세 지속, 환율 상승세',
      content: '원화 약세가 지속되면서 환율이 상승세를 보이고 있어 수출업계에 부담이 되고 있다.',
      time: '10시간 전',
      source: '매일경제'
    }
  ];

  // 현재 선택된 주식의 뉴스 또는 기본 뉴스
  const currentPositiveNews = stockNews[selectedStock]?.positive || defaultPositiveNews;
  const currentNegativeNews = stockNews[selectedStock]?.negative || defaultNegativeNews;

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
            
            <div className="content-card news-featured">
              <h4>주요 뉴스</h4>
              <div className="featured-news">
                <div className="news-item featured">
                  <div className="news-category">시장 동향</div>
                  <h5>{selectedStock}, {stocks.find(s => s.name === selectedStock)?.changeType === 'positive' ? '상승세' : '하락세'} 지속</h5>
                  <p>{selectedStock}의 주가가 {stocks.find(s => s.name === selectedStock)?.change} 변동을 보이며 {stocks.find(s => s.name === selectedStock)?.changeType === 'positive' ? '상승세' : '하락세'}를 지속하고 있다.</p>
                  <div className="news-meta">
                    <span className="news-time">1시간 전</span>
                    <span className="news-source">한국경제</span>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="content-card">
              <h4>호재 뉴스 ({currentPositiveNews.length})</h4>
              <div className="news-list">
                {currentPositiveNews.map((news, index) => (
                  <div key={index} className="news-item">
                    <div className="news-category">{news.category}</div>
                    <h5>{news.title}</h5>
                    <p>{news.content}</p>
                    <div className="news-meta">
                      <span className="news-time">{news.time}</span>
                      <span className="news-source">{news.source}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="content-card">
              <h4>악재 뉴스 ({currentNegativeNews.length})</h4>
              <div className="news-list">
                {currentNegativeNews.map((news, index) => (
                  <div key={index} className="news-item">
                    <div className="news-category">{news.category}</div>
                    <h5>{news.title}</h5>
                    <p>{news.content}</p>
                    <div className="news-meta">
                      <span className="news-time">{news.time}</span>
                      <span className="news-source">{news.source}</span>
                    </div>
                  </div>
                ))}
              </div>
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
                  onClick={() => setSelectedStock(stock.name)}
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