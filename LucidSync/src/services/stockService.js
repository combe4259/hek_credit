const API_BASE_URL = 'http://localhost:8000';

export const stockService = {
  // 주식 목록 조회
  async getStocksList(symbols) {
    try {
      const queryParams = symbols.map(symbol => `symbols=${encodeURIComponent(symbol)}`).join('&');
      const url = `${API_BASE_URL}/stocks/list?${queryParams}`;
      
      console.log('주식 API 호출 URL:', url);
      console.log('요청 심볼들:', symbols);
      
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      console.log('주식 API 응답 상태:', response.status);

      if (!response.ok) {
        const error = await response.json();
        console.error('주식 API 에러 응답:', error);
        throw new Error(error.detail || '주식 데이터 조회에 실패했습니다.');
      }

      const data = await response.json();
      console.log('주식 API 성공 응답:', data);
      return data;
    } catch (error) {
      console.error('주식 데이터 조회 에러:', error);
      throw error;
    }
  },

  // 기본 주식 심볼 목록 (한국 주요 주식들)
  getDefaultSymbols() {
    return [
      '005930', // 삼성전자
      '000660', // SK하이닉스
      '035420', // NAVER
      '035720', // 카카오
      '051910', // LG화학
      '005380', // 현대차
      '000270', // 기아
      '005490', // POSCO홀딩스
      '066570', // LG전자
      '105560', // KB금융
    ];
  },

  // 주식 심볼을 이름으로 변환하는 매핑
  getSymbolToNameMapping() {
    return {
      '005930': '삼성전자',
      '000660': 'SK하이닉스',
      '035420': 'NAVER',
      '035720': '카카오',
      '051910': 'LG화학',
      '005380': '현대차',
      '000270': '기아',
      '005490': 'POSCO홀딩스',
      '066570': 'LG전자',
      '105560': 'KB금융',
    };
  },

  // 섹터 정보
  getSectorInfo() {
    return {
      '005930': '반도체',
      '000660': '반도체',
      '035420': 'IT',
      '035720': 'IT',
      '051910': '화학',
      '005380': '자동차',
      '000270': '자동차',
      '005490': '철강',
      '066570': '전자',
      '105560': '금융',
    };
  },
};
