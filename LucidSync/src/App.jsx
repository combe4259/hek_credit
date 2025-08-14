import { useState, useEffect } from 'react'
import Navigation from './components/Navigation'
import MockInvestment from './components/MockInvestment'
import MyStocks from './components/MyStocks'
import News from './components/News'
import Login from './components/Auth/Login'
import Signup from './components/Auth/Signup'
import { authService } from './services/authService'
import './App.css'

function App() {
  const [activeTab, setActiveTab] = useState('mock-investment')
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [authMode, setAuthMode] = useState('login') // 'login' or 'signup'
  const [user, setUser] = useState(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)

  // 앱 시작 시 사용자 정보 확인
  useEffect(() => {
    const checkAuth = () => {
      try {
        const currentUser = authService.getCurrentUser()
        if (currentUser) {
          setUser(currentUser)
          setIsAuthenticated(true)
        }
      } catch (error) {
        console.error('사용자 정보 확인 실패:', error)
      } finally {
        setIsLoading(false)
      }
    }

    checkAuth()
  }, [])

  const handleLogin = async (loginData) => {
    setIsLoading(true)
    setError(null)
    
    try {
      const userData = await authService.login(loginData.phone, loginData.password)
      
      // 사용자 정보 설정
      setUser(userData)
      setIsAuthenticated(true)
    } catch (error) {
      console.error('로그인 실패:', error)
      setError(error.message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSignup = async (signupData) => {
    setIsLoading(true)
    setError(null)
    
    try {
      await authService.signup(
        signupData.userId, 
        signupData.phone, 
        signupData.password
      )
      
      // 회원가입 성공 시 로그인 창으로 이동
      setAuthMode('login')
      setError(null)
      // 성공 메시지 표시
      alert('회원가입이 완료되었습니다. 로그인해주세요.')
    } catch (error) {
      console.error('회원가입 실패:', error)
      setError(error.message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleLogout = async () => {
    try {
      await authService.logout()
      setUser(null)
      setIsAuthenticated(false)
      setAuthMode('login')
      setError(null)
    } catch (error) {
      console.error('로그아웃 실패:', error)
    }
  }

  const switchToSignup = () => {
    setAuthMode('signup')
    setError(null)
  }

  const switchToLogin = () => {
    setAuthMode('login')
    setError(null)
  }

  const renderActiveTab = () => {
    switch (activeTab) {
      case 'mock-investment':
        return <MockInvestment user={user} />
      case 'my-stocks':
        return <MyStocks />
      case 'news':
        return <News />
      default:
        return <MockInvestment user={user} />
    }
  }

  // 로딩 중일 때
  if (isLoading) {
    return (
      <div className="app">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>로딩 중...</p>
        </div>
      </div>
    )
  }

  // 인증되지 않은 경우 로그인/회원가입 화면 표시
  if (!isAuthenticated) {
    return (
      <div className="app">
        {authMode === 'login' ? (
          <Login 
            onSwitchToSignup={switchToSignup}
            onLogin={handleLogin}
            isLoading={isLoading}
            error={error}
          />
        ) : (
          <Signup 
            onSwitchToLogin={switchToLogin}
            onSignup={handleSignup}
            isLoading={isLoading}
            error={error}
          />
        )}
      </div>
    )
  }

  // 인증된 경우 메인 앱 화면 표시
  return (
    <div className="app">
      <Navigation 
        activeTab={activeTab} 
        onTabChange={setActiveTab}
        user={user}
        onLogout={handleLogout}
      />
      <main className="main-content">
        {renderActiveTab()}
      </main>
    </div>
  )
}

export default App
