import React from 'react';
import './Navigation.css';

const Navigation = ({ activeTab, onTabChange, user, onLogout }) => {
  return (
    <nav className="navigation">
      <div className="nav-container">
        <div className="nav-brand">
          <h2>LucidSync</h2>
        </div>
        <div className="nav-tabs">
          <button
            className={`nav-tab ${activeTab === 'mock-investment' ? 'active' : ''}`}
            onClick={() => onTabChange('mock-investment')}
          >
            모의투자
          </button>
          <button
            className={`nav-tab ${activeTab === 'my-stocks' ? 'active' : ''}`}
            onClick={() => onTabChange('my-stocks')}
          >
            나의 주식
          </button>
          <button
            className={`nav-tab ${activeTab === 'news' ? 'active' : ''}`}
            onClick={() => onTabChange('news')}
          >
            뉴스
          </button>
        </div>
        {user && (
          <div className="nav-user">
            <span className="user-name">{user.user_id}</span>
            <button className="logout-button" onClick={onLogout}>
              로그아웃
            </button>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navigation; 