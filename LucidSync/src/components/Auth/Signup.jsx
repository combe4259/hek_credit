import React, { useState } from 'react';
import { colors } from '../../theme/colors';
import './Auth.css';

const Signup = ({ onSwitchToLogin, onSignup, isLoading, error }) => {
  const [formData, setFormData] = useState({
    userId: '',
    phone: '',
    password: '',
    confirmPassword: ''
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
    
    if (!formData.userId) {
      newErrors.userId = '아이디를 입력해주세요';
    } else if (formData.userId.length < 4) {
      newErrors.userId = '아이디는 최소 4자 이상이어야 합니다';
    } else if (!/^[a-zA-Z0-9_]+$/.test(formData.userId)) {
      newErrors.userId = '아이디는 영문, 숫자, 언더스코어(_)만 사용 가능합니다';
    }
    
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
    
    if (!formData.confirmPassword) {
      newErrors.confirmPassword = '비밀번호 확인을 입력해주세요';
    } else if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = '비밀번호가 일치하지 않습니다';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSignup = () => {
    if (validateForm()) {
      if (onSignup) {
        onSignup(formData);
      }
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    handleSignup();
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <h2>회원가입</h2>
          <p>LucidSync 계정을 만들어보세요</p>
        </div>
        
        {error && (
          <div className="auth-error">
            <p>{error}</p>
          </div>
        )}
        
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-group">
            <label htmlFor="userId">아이디</label>
            <input
              type="text"
              id="userId"
              name="userId"
              value={formData.userId}
              onChange={handleChange}
              className={errors.userId ? 'error' : ''}
              placeholder="아이디를 입력하세요 (4자 이상)"
              disabled={isLoading}
            />
            {errors.userId && <span className="error-message">{errors.userId}</span>}
          </div>

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
              placeholder="비밀번호를 입력하세요(6자 이상)"
              disabled={isLoading}
            />
            {errors.password && <span className="error-message">{errors.password}</span>}
          </div>

          <div className="form-group">
            <label htmlFor="confirmPassword">비밀번호 확인</label>
            <input
              type="password"
              id="confirmPassword"
              name="confirmPassword"
              value={formData.confirmPassword}
              onChange={handleChange}
              className={errors.confirmPassword ? 'error' : ''}
              placeholder="비밀번호를 다시 입력하세요"
              disabled={isLoading}
            />
            {errors.confirmPassword && <span className="error-message">{errors.confirmPassword}</span>}
          </div>

          <button 
            type="button" 
            className="auth-button"
            disabled={isLoading}
            onClick={handleSignup}
          >
            {isLoading ? '회원가입 중...' : '회원가입'}
          </button>
        </form>

        <div className="auth-footer">
          <p>
            이미 계정이 있으신가요?{' '}
            <button 
              type="button" 
              className="link-button"
              onClick={onSwitchToLogin}
              disabled={isLoading}
            >
              로그인
            </button>
          </p>
        </div>
      </div>
    </div>
  );
};

export default Signup;
