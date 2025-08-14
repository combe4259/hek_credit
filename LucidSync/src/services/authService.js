const API_BASE_URL = 'http://localhost:8000';

/** 서버 응답을 항상 {id, user_id, ...} 형태로 정규화 */
function normalizeUser(u) {
  if (!u) return null;
  const id = u.id ?? u.user_id ?? u.userId;
  if (id == null) return null;
  return { ...u, id, user_id: u.user_id ?? String(id) };
}

export const authService = {
  // 로그인 API
  async login(phone, password) {
    try {
      // 요청 데이터 로깅
      const requestData = { phone_number: phone, password };
      console.log('로그인 요청 데이터:', requestData);

      const res = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData),
      });

      console.log('로그인 응답 상태:', res.status);

      if (!res.ok) {
        let msg = `로그인 실패 (${res.status})`;
        try {
          const err = await res.json();
          console.log('로그인 에러 응답:', err);
          // FastAPI 422일 때 detail이 배열로 옴
          if (Array.isArray(err.detail)) {
            msg = err.detail.map(d => d.msg || d.message || '필드 오류').join(', ');
          } else {
            msg = err.detail || err.message || msg;
          }
        } catch (parseError) {
          console.error('에러 응답 파싱 실패:', parseError);
        }
        throw new Error(msg);
      }

      const data = await res.json();
      console.log('로그인 성공 응답:', data);

      const user = normalizeUser(data?.user ?? data);
      if (!user) throw new Error('서버 응답 형식이 올바르지 않습니다 (user/id 누락)');

      // 사용자 정보와 토큰을 localStorage에 저장
      localStorage.setItem('user', JSON.stringify(user));
      if (data.token) {
        localStorage.setItem('token', data.token);
      }



      return user;
    } catch (e) {
      if (e instanceof TypeError) {
        // CORS/네트워크 차단 시
        throw new Error('서버에 연결할 수 없습니다. CORS/서버 상태를 확인하세요.');
      }
      throw e;
    }
  },

  // 회원가입 API
  async signup(userId, phone, password) {
    try {
      // 요청 데이터 로깅
      const requestData = { user_id: userId, phone_number: phone, password };
      console.log('회원가입 요청 데이터:', requestData);



      const res = await fetch(`${API_BASE_URL}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData),
      });

      console.log('회원가입 응답 상태:', res.status);


      if (!res.ok) {
        let msg = '회원가입에 실패했습니다';
        try {
          const err = await res.json();
          console.log('회원가입 에러 응답:', err);
          if (Array.isArray(err.detail)) {
            msg = err.detail.map(d => d.msg || d.message || '필드 오류').join(', ');
          } else {
            msg = err.detail || err.message || msg;
          }
        } catch (parseError) {
          console.error('에러 응답 파싱 실패:', parseError);
        }
        throw new Error(msg);
      }

      const data = await res.json();
      console.log('회원가입 성공 응답:', data);


      // 회원가입 응답이 user를 주면 저장
      const user = normalizeUser(data?.user ?? data);
      if (user) localStorage.setItem('user', JSON.stringify(user));
      return data;
    } catch (error) {
      throw error;
    }
  },

  // 로그아웃 API
  async logout() {
    try {
      const token = localStorage.getItem('token');
      if (token) {
        const response = await fetch(`${API_BASE_URL}/auth/logout`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
        });
      }
      // 로컬 스토리지 정리
      localStorage.removeItem('user');
      localStorage.removeItem('token');
      return true;
    } catch (error) {
      console.error('로그아웃 에러:', error);
      // 에러가 있어도 로컬 정리는 수행
      localStorage.removeItem('user');
      localStorage.removeItem('token');
      return true;
    }
  },

  // 현재 사용자 가져오기
  getCurrentUser() {
    try {
      const raw = localStorage.getItem('user');
      return raw ? JSON.parse(raw) : null;
    } catch {
      localStorage.removeItem('user');
      return null;
    }
  },

  // 인증 상태 확인
  isAuthenticated() {
    return !!this.getCurrentUser();
  },

  // 토큰 유효성 검사 (선택사항)
  async validateToken() {
    try {
      const token = localStorage.getItem('token');
      const user = localStorage.getItem('user');

      if (!token || !user) {
        return false;
      }

      const response = await fetch(`${API_BASE_URL}/auth/validate`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const data = await response.json();
        return data.user || JSON.parse(user);
      }

      // 토큰이 유효하지 않으면 로컬 스토리지 정리
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      return false;
    } catch (error) {
      console.error('토큰 검증 에러:', error);
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      return false;
    }
  }
};
