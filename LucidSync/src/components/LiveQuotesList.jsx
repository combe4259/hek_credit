import { useEffect, useMemo, useRef, useState } from 'react';

// S&P 500 주요 미국주 50개
const UNIVERSE = [
  'AAPL',   // Apple
  'MSFT',   // Microsoft
  'GOOGL',  // Alphabet (Google)
  'AMZN',   // Amazon
  'NVDA',   // NVIDIA
  'META',   // Meta Platforms
  'BRK.B',  // Berkshire Hathaway
  'TSLA',   // Tesla
  'UNH',    // UnitedHealth Group
  'JNJ',    // Johnson & Johnson
  'JPM',    // JPMorgan Chase
  'V',      // Visa
  'PG',     // Procter & Gamble
  'HD',     // Home Depot
  'MA',     // Mastercard
  'DIS',    // Walt Disney
  'PYPL',   // PayPal
  'BAC',    // Bank of America
  'ADBE',   // Adobe
  'CRM',    // Salesforce
  'NFLX',   // Netflix
  'ABT',    // Abbott Laboratories
  'KO',     // Coca-Cola
  'PFE',    // Pfizer
  'TMO',    // Thermo Fisher Scientific
  'AVGO',   // Broadcom
  'ACN',    // Accenture
  'WMT',    // Walmart
  'DHR',    // Danaher
  'LLY',    // Eli Lilly
  'NEE',    // NextEra Energy
  'TXN',    // Texas Instruments
  'QCOM',   // Qualcomm
  'HON',    // Honeywell
  'LOW',    // Lowe's
  'UNP',    // Union Pacific
  'UPS',    // United Parcel Service
  'IBM',    // International Business Machines
  'RTX',    // Raytheon Technologies
  'CAT',    // Caterpillar
  'SPGI',   // S&P Global
  'GS',     // Goldman Sachs
  'MS',     // Morgan Stanley
  'AXP',    // American Express
  'PLD',    // Prologis
  'SCHW',   // Charles Schwab
  'T',      // AT&T
  'VZ',     // Verizon
  'CMCSA',  // Comcast
  'COST',   // Costco
  'INTU',   // Intuit
  'AMD',    // Advanced Micro Devices
  'INTC',   // Intel
  'ORCL',   // Oracle
  'CSCO',   // Cisco Systems
];

// 심볼 정규화: Yahoo WS는 보통 '.' 대신 '-' 사용 (예: BRK.B -> BRK-B)
const toYahoo = (s) => s?.toUpperCase().replace(/\./g, '-');
const fromYahoo = (s) => s?.toUpperCase().replace(/-/g, '.');

const PAGE_SIZE = 10;

export default function LiveQuotesList({ onStockSelect, selectedStock, onStockData }) {
  const [page, setPage] = useState(1);
  const [rows, setRows] = useState(() => new Map()); // symbol -> {symbol, price, change, ...}
  const [wsConnected, setWsConnected] = useState(false);
  const [prevCloseMap, setPrevCloseMap] = useState({});
  const wsRef = useRef(null);
  const pendingSubRef = useRef(null);
  const sentinelRef = useRef(null);

  const visibleSymbols = useMemo(() => UNIVERSE.slice(0, page * PAGE_SIZE), [page]);

  // visibleSymbols가 바뀔 때마다 전일종가 스냅샷 가져오기
  useEffect(() => {
    if (!visibleSymbols.length) return;
    const yahooSyms = visibleSymbols.map(toYahoo).join(',');
    fetch(`http://localhost:8000/quotes/prevclose?symbols=${yahooSyms}`)
      .then(r => r.ok ? r.json() : {})
      .then(data => {
        // 서버는 Yahoo포맷 심볼 키로 내려줄 것 → 화면표시용으로 다시 '.' 복원
        const m = {};
        for (const [symY, pc] of Object.entries(data || {})) {
          m[fromYahoo(symY)] = pc;
        }
        setPrevCloseMap(prev => ({ ...prev, ...m }));
      })
      .catch(() => {});
  }, [visibleSymbols.join(',')]);

  // 선택된 주식의 데이터를 부모에게 전달
  useEffect(() => {
    if (selectedStock && onStockData) {
      const stockSymbol = Object.keys(getStockName).find(key => getStockName(key) === selectedStock);
      if (stockSymbol) {
        const stockData = rows.get(stockSymbol);
        if (stockData && (stockData.price || stockData.change)) {
          onStockData(stockData);
        }
      }
    }
  }, [selectedStock, rows, onStockData]);

  // WS 연결 & 구독 관리
  useEffect(() => {
    let reconnectTimeout;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;

    const connectWebSocket = () => {
      const url = `ws://localhost:8000/ws/quotes`;
      
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsConnected(true);
        reconnectAttempts = 0; // 연결 성공시 재시도 카운트 리셋
        // 초기 구독 - Yahoo 포맷으로 변환
        ws.send(JSON.stringify({ type: 'subscribe', symbols: visibleSymbols.map(toYahoo) }));
      };

      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          
          // bootstrap 묶음 or 단일 틱
          if (msg?.type === 'bootstrap' && Array.isArray(msg.data)) {
            setRows(prev => {
              const next = new Map(prev);
              for (const t of msg.data) {
                const raw = (t.id || t.symbol || t.s || '').toUpperCase();
                const sym = fromYahoo(raw);
                if (!sym) continue;
                next.set(sym, mergeTick(next.get(sym), t, prevCloseMap, sym));
              }
              return next;
            });
          } else {
            // 단일 틱
            const raw = (msg.id || msg.symbol || msg.s || '').toUpperCase();
            const sym = fromYahoo(raw);
            if (!sym) return;
            setRows(prev => {
              const next = new Map(prev);
              next.set(sym, mergeTick(next.get(sym), msg, prevCloseMap, sym));
              return next;
            });
          }
        } catch (error) {
          console.error('WebSocket 메시지 파싱 에러:', error);
        }
      };

      ws.onclose = (event) => { 
        setWsConnected(false);
        
        // 자동 재연결 (최대 5회)
        if (reconnectAttempts < maxReconnectAttempts) {
          reconnectAttempts++;
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000); // 지수 백오프
          reconnectTimeout = setTimeout(connectWebSocket, delay);
        }
      };
      
      ws.onerror = (error) => { 
        setWsConnected(false);
      };
    };

    connectWebSocket();

    return () => {
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []); // 최초 1회만 연결

  // 페이지가 늘어날 때마다 구독 심볼 갱신
  useEffect(() => {
    // 너무 자주 보내지 않도록 간단 디바운스
    clearTimeout(pendingSubRef.current);
    pendingSubRef.current = setTimeout(() => {
      if (wsRef.current?.readyState === 1) {
        // Yahoo 포맷으로 변환하여 구독
        wsRef.current.send(JSON.stringify({ type: 'subscribe', symbols: visibleSymbols.map(toYahoo) }));
      }
    }, 150);
    return () => clearTimeout(pendingSubRef.current);
  }, [visibleSymbols]);

  // 무한 스크롤(IntersectionObserver)
  useEffect(() => {
    const el = sentinelRef.current;
    if (!el) return;
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) setPage(p => {
          const next = p + 1;
          const maxPage = Math.ceil(UNIVERSE.length / PAGE_SIZE);
          return next > maxPage ? p : next;
        });
      });
    }, { rootMargin: '300px' }); // 여유를 두고 미리 로딩
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const ordered = useMemo(() => {
    // 보이는 심볼 순서대로 rows에서 꺼내기
    return visibleSymbols.map(s => {
      const row = rows.get(s);
      if (row && (row.price || row.change)) {
        return row;
      }
      return { symbol: s };
    });
  }, [rows, visibleSymbols]);

  // 심볼을 회사명으로 변환
  const getStockName = (symbol) => {
    const nameMap = {
      'AAPL': 'Apple',
      'MSFT': 'Microsoft',
      'GOOGL': 'Alphabet',
      'AMZN': 'Amazon',
      'NVDA': 'NVIDIA',
      'META': 'Meta Platforms',
      'BRK.B': 'Berkshire Hathaway',
      'TSLA': 'Tesla',
      'UNH': 'UnitedHealth Group',
      'JNJ': 'Johnson & Johnson',
      'JPM': 'JPMorgan Chase',
      'V': 'Visa',
      'PG': 'Procter & Gamble',
      'HD': 'Home Depot',
      'MA': 'Mastercard',
      'DIS': 'Walt Disney',
      'PYPL': 'PayPal',
      'BAC': 'Bank of America',
      'ADBE': 'Adobe',
      'CRM': 'Salesforce',
      'NFLX': 'Netflix',
      'ABT': 'Abbott Laboratories',
      'KO': 'Coca-Cola',
      'PFE': 'Pfizer',
      'TMO': 'Thermo Fisher Scientific',
      'AVGO': 'Broadcom',
      'ACN': 'Accenture',
      'WMT': 'Walmart',
      'DHR': 'Danaher',
      'LLY': 'Eli Lilly',
      'NEE': 'NextEra Energy',
      'TXN': 'Texas Instruments',
      'QCOM': 'Qualcomm',
      'HON': 'Honeywell',
      'LOW': 'Lowe\'s',
      'UNP': 'Union Pacific',
      'UPS': 'United Parcel Service',
      'IBM': 'IBM',
      'RTX': 'Raytheon Technologies',
      'CAT': 'Caterpillar',
      'SPGI': 'S&P Global',
      'GS': 'Goldman Sachs',
      'MS': 'Morgan Stanley',
      'AXP': 'American Express',
      'PLD': 'Prologis',
      'SCHW': 'Charles Schwab',
      'T': 'AT&T',
      'VZ': 'Verizon',
      'CMCSA': 'Comcast',
      'COST': 'Costco',
      'INTU': 'Intuit',
      'AMD': 'Advanced Micro Devices',
      'INTC': 'Intel',
      'ORCL': 'Oracle',
      'CSCO': 'Cisco Systems',
    };
    return nameMap[symbol] || symbol;
  };

  // 섹터 정보
  const getSector = (symbol) => {
    const sectorMap = {
      'AAPL': 'Technology',
      'MSFT': 'Technology',
      'GOOGL': 'Technology',
      'AMZN': 'Consumer Discretionary',
      'NVDA': 'Technology',
      'META': 'Technology',
      'BRK.B': 'Financials',
      'TSLA': 'Consumer Discretionary',
      'UNH': 'Healthcare',
      'JNJ': 'Healthcare',
      'JPM': 'Financials',
      'V': 'Financials',
      'PG': 'Consumer Staples',
      'HD': 'Consumer Discretionary',
      'MA': 'Financials',
      'DIS': 'Communication Services',
      'PYPL': 'Financials',
      'BAC': 'Financials',
      'ADBE': 'Technology',
      'CRM': 'Technology',
      'NFLX': 'Communication Services',
      'ABT': 'Healthcare',
      'KO': 'Consumer Staples',
      'PFE': 'Healthcare',
      'TMO': 'Healthcare',
      'AVGO': 'Technology',
      'ACN': 'Technology',
      'WMT': 'Consumer Staples',
      'DHR': 'Healthcare',
      'LLY': 'Healthcare',
      'NEE': 'Utilities',
      'TXN': 'Technology',
      'QCOM': 'Technology',
      'HON': 'Industrials',
      'LOW': 'Consumer Discretionary',
      'UNP': 'Industrials',
      'UPS': 'Industrials',
      'IBM': 'Technology',
      'RTX': 'Industrials',
      'CAT': 'Industrials',
      'SPGI': 'Financials',
      'GS': 'Financials',
      'MS': 'Financials',
      'AXP': 'Financials',
      'PLD': 'Real Estate',
      'SCHW': 'Financials',
      'T': 'Communication Services',
      'VZ': 'Communication Services',
      'CMCSA': 'Communication Services',
      'COST': 'Consumer Staples',
      'INTU': 'Technology',
      'AMD': 'Technology',
      'INTC': 'Technology',
      'ORCL': 'Technology',
      'CSCO': 'Technology',
    };
    return sectorMap[symbol] || 'Other';
  };

  return (
    <div className="quotes">
      {ordered.map(row => (
        <QuoteRow 
          key={row.symbol} 
          row={row} 
          stockName={getStockName(row.symbol)}
          sector={getSector(row.symbol)}
          onSelect={() => onStockSelect && onStockSelect(getStockName(row.symbol))}
          isSelected={selectedStock === getStockName(row.symbol)}
          isLive={wsConnected}
        />
      ))}
      <div ref={sentinelRef} style={{ height: 1 }} />
    </div>
  );
}

function QuoteRow({ row, stockName, sector, onSelect, isSelected, isLive }) {
  const { symbol, price, change, dollarChange } = row;
  
  // 달러 변화량 우선 표시, 없으면 퍼센트 표시
  let changeDisplay = '-';
  let changeType = '';
  
  if (dollarChange !== null && dollarChange !== undefined) {
    // 상승할 때만 +, 하락할 때는 - 표시
    const sign = dollarChange > 0 ? '+' : (dollarChange < 0 ? '-' : '');
    changeDisplay = `${sign}$${Math.abs(dollarChange).toFixed(2)}`;
    changeType = dollarChange >= 0 ? 'positive' : 'negative';
  } else if (typeof change === 'number') {
    // 상승할 때만 +, 하락할 때는 - 표시
    const sign = change > 0 ? '+' : (change < 0 ? '-' : '');
    changeDisplay = `${sign}${Math.abs(change).toFixed(2)}%`;
    changeType = change >= 0 ? 'positive' : 'negative';
  }
  
  const priceFormatted = price ? `$${price.toLocaleString()}` : '-';
  
  return (
    <div
      className={`stock-list-item ${isSelected ? 'selected' : ''}`}
      onClick={onSelect}
    >
      <div className="stock-info">
        <div className="stock-name-sector">
          <span className="stock-name">{stockName}</span>
          <span className="stock-sector">{sector}</span>
          {isLive && <span style={{ fontSize: '10px', color: '#00a651', marginLeft: '8px' }}>●</span>}
        </div>
        <div className="stock-price-change">
          <span className="stock-price">{priceFormatted}</span>
          <span className={`stock-change ${changeType}`}>
            {changeDisplay}
          </span>
        </div>
      </div>
    </div>
  );
}

/** 틱 → 우리 로우 형태로 병합 */
function mergeTick(prev, tick, prevCloseMap, symKey) {
  const rawSym = (tick.id || tick.symbol || tick.s || symKey || '').toUpperCase();
  const symbol = fromYahoo(rawSym); // 저장은 화면표시용으로

  // 가격 후보들
  const price =
    tick.price ??
    tick.regularMarketPrice ??
    tick.last ??
    tick.lp ??
    prev?.price ??
    null;

  // 달러 변화량 필드들
  const dollarChange =
    tick.regularMarketChange ??  // 정규시장 달러 변화량
    tick.change ??               // 변화량
    tick.dollarChange ??         // 달러 변화량
    prev?.dollarChange ??
    null;

  // 퍼센트 변화량 필드들 (있는 경우만 사용)
  let change = null;
  if (tick.percentChange || tick.regularMarketChangePercent || tick.changePercent || tick.percent) {
    change =
      tick.percentChange ??      // 퍼센트 변화량
      tick.regularMarketChangePercent ??  // 정규시장 퍼센트 변화량
      tick.changePercent ??      // 변화 퍼센트
      tick.percent ??            // 퍼센트
      prev?.change ??
      null;
  }

  return { ...(prev || {}), symbol, price, change, dollarChange };
}
