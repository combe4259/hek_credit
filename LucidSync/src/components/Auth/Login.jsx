import React, { useState } from 'react';
import { colors } from '../../theme/colors';
import './Auth.css';
import { authService } from '../../services/authService';

const Login = ({ onSwitchToSignup, onLogin, isLoading, error }) => {
  const [formData, setFormData] = useState({
    phone: '',
    password: ''
  });
  const [errors, setErrors] = useState({});

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const validateForm = () => {
    const newErrors = {};

    if (!formData.phone) {
      newErrors.phone = '전화번호를 입력해주세요';
    } else if (!/^[0-9]{10,11}$/.test(formData.phone.replace(/\s/g, ''))) {
      newErrors.phone = '올바른 전화번호 형식을 입력해주세요 (10-11자리)';
    }

    if (!formData.password) {
      newErrors.password = '비밀번호를 입력해주세요';
    } else if (formData.password.length < 6) {
      newErrors.password = '비밀번호는 최소 6자 이상이어야 합니다';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleLogin = async () => {
    if (!validateForm()) {
      return;
    }

    try {
      const resUser = await authService.login(formData.phone, formData.password);
      const user = resUser?.user ?? resUser;

      if (!user || user.id == null) {
        console.error('로그인 응답 형식 문제:', resUser);
        throw new Error('서버 응답 형식이 올바르지 않습니다 (id 누락)');
      }
      // 부모 컴포넌트의 onLogin 콜백 호출 - formData 전달
      if (onLogin) {
        onLogin(formData);
      }
      localStorage.setItem('user', JSON.stringify(user));
    } catch (e) {
      console.error('로그인 실패:', e);
      setErrors({ general: e.message || '로그인에 실패했습니다.' });
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    handleLogin();
  };

  return (
      <div className="auth-container">
        <div className="auth-card">
          <div className="auth-header">
            <h2>로그인</h2>
            <p>LucidSync에 오신 것을 환영합니다</p>
          </div>

          {(error || errors.general) && (
              <div className="auth-error">
                <p>{error || errors.general}</p>
              </div>
          )}

          <form onSubmit={handleSubmit} className="auth-form">
            <div className="form-group">
              <label htmlFor="phone">전화번호</label>
              <input
                  type="tel"
                  id="phone"
                  name="phone"
                  value={formData.phone}
                  onChange={handleChange}
                  className={errors.phone ? 'error' : ''}
                  placeholder="전화번호를 입력하세요 (예: 01012345678)"
                  disabled={isLoading}
              />
              {errors.phone && <span className="error-message">{errors.phone}</span>}
            </div>

            <div className="form-group">
              <label htmlFor="password">비밀번호</label>
              <input
                  type="password"
                  id="password"
                  name="password"
                  value={formData.password}
                  onChange={handleChange}
                  className={errors.password ? 'error' : ''}
                  placeholder="비밀번호를 입력하세요"
                  disabled={isLoading}
              />
              {errors.password && <span className="error-message">{errors.password}</span>}
            </div>

            <button
                type="button"
                className="auth-button"
                disabled={isLoading}
                onClick={handleLogin}
            >
              {isLoading ? '로그인 중...' : '로그인'}
            </button>
          </form>

          <div className="auth-footer">
            <p>
              계정이 없으신가요?{' '}
              <button
                  type="button"
                  className="link-button"
                  onClick={onSwitchToSignup}
                  disabled={isLoading}
              >
                회원가입
              </button>
            </p>
          </div>
        </div>
      </div>
  );
};

export default Login;
