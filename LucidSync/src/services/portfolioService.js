const API_BASE_URL = 'http://localhost:8000';

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
};
