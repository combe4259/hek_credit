const API_BASE_URL = 'http://localhost:8000';

// 보유 종목 가져오기
const getHoldings = async (portfolioId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/portfolios/${portfolioId}/holdings`, {
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    if (!response.ok) {
      throw new Error('Failed to fetch holdings');
    }
    
    return response.json();
  } catch (error) {
    console.error('Error fetching holdings:', error);
    // 임시 데이터 반환 (백엔드 API 구현 전)
    return [
      {
        id: 1,
        symbol: 'AAPL',
        quantity: 10,
        avgPrice: 150,
        currentPrice: 233.33,
        purchaseDate: '2025-07-15'
      },
      {
        id: 2,
        symbol: 'MSFT',
        quantity: 5,
        avgPrice: 300,
        currentPrice: 520.58,
        purchaseDate: '2025-08-01'
      }
    ];
  }
};

// 거래 내역 가져오기
const getTransactions = async (portfolioId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/portfolios/${portfolioId}/transactions`, {
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    if (!response.ok) {
      throw new Error('Failed to fetch transactions');
    }
    
    return response.json();
  } catch (error) {
    console.error('Error fetching transactions:', error);
    // 임시 데이터 반환 (매수-매도 쌍 포함)
    return [
      {
        id: 1,
        symbol: 'AAPL',
        type: 'BUY',
        quantity: 10,
        price: 150,
        total: 1500,
        date: '2025-07-15',
        feedbackSent: true
      },
      {
        id: 2,
        symbol: 'AAPL',
        type: 'SELL',
        quantity: 10,
        price: 175,
        total: 1750,
        date: '2025-08-01',
        feedbackSent: true
      },
      {
        id: 3,
        symbol: 'MSFT',
        type: 'BUY',
        quantity: 5,
        price: 300,
        total: 1500,
        date: '2025-07-20',
        feedbackSent: true
      },
      {
        id: 4,
        symbol: 'MSFT',
        type: 'SELL',
        quantity: 5,
        price: 320,
        total: 1600,
        date: '2025-08-05',
        feedbackSent: true
      },
      {
        id: 5,
        symbol: 'TSLA',
        type: 'BUY',
        quantity: 8,
        price: 200,
        total: 1600,
        date: '2025-08-10',
        feedbackSent: false
      }
    ];
  }
};

// 거래 생성
const createTransaction = async (transaction) => {
  try {
    const response = await fetch(`${API_BASE_URL}/transactions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(transaction),
    });
    
    if (!response.ok) {
      throw new Error('Failed to create transaction');
    }
    
    return response.json();
  } catch (error) {
    console.error('Error creating transaction:', error);
    throw error;
  }
};

// 보유 종목 업데이트
const updateHolding = async (holdingId, updateData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/holdings/${holdingId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(updateData),
    });
    
    if (!response.ok) {
      throw new Error('Failed to update holding');
    }
    
    return response.json();
  } catch (error) {
    console.error('Error updating holding:', error);
    throw error;
  }
};

export const portfolioService = {
  // 포트폴리오 생성
  async createPortfolio(userId, portfolioName) {
    try {
      const response = await fetch(`${API_BASE_URL}/portfolios/create/${userId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: portfolioName }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || '포트폴리오 생성에 실패했습니다.');
      }

      const data = await response.json();
      return data.portfolio;
    } catch (error) {
      console.error('포트폴리오 생성 에러:', error);
      throw error;
    }
  },

  // 포트폴리오 목록 조회
  async getPortfolios(userId = null) {
    try {
      let url = `${API_BASE_URL}/portfolios`;
      if (userId) {
        url += `?user_id=${userId}`;
      }

      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || '포트폴리오 조회에 실패했습니다.');
      }

      const data = await response.json();
      return data.items;
    } catch (error) {
      console.error('포트폴리오 조회 에러:', error);
      throw error;
    }
  },

  // 포트폴리오 삭제
  async deletePortfolio(portfolioId) {
    try {
      const response = await fetch(`${API_BASE_URL}/portfolios/${portfolioId}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || '포트폴리오 삭제에 실패했습니다.');
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('포트폴리오 삭제 에러:', error);
      throw error;
    }
  },
  
  // 추가된 메소드들
  getHoldings,
  getTransactions,
  createTransaction,
  updateHolding
};
