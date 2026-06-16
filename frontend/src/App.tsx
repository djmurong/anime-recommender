import { useMemo, useState } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { AppLayout } from './components/AppLayout'
import { clearAuth, clearOnboarding, readAuth, readOnboarding, saveAuth, saveOnboarding } from './lib/storage'
import { AnimeDetailPage } from './pages/AnimeDetailPage'
import { HomePage } from './pages/HomePage'
import { OnboardingPage } from './pages/OnboardingPage'
import { WelcomePage } from './pages/WelcomePage'
import { LoginPage } from './pages/LoginPage'
import { SignupPage } from './pages/SignupPage'
import { ListsPage } from './pages/ListsPage'
import { SearchPage } from './pages/SearchPage'
import { ProfilePage } from './pages/ProfilePage'
import { ImportPage } from './pages/ImportPage'
import type { UserOnboarding } from './types'

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(() => readAuth())
  const [onboarding, setOnboarding] = useState<UserOnboarding | null>(() => readOnboarding())

  const isOnboarded = useMemo(
    () => Boolean(onboarding?.favoriteGenreIds.length && onboarding.favoriteAnimeIds.length),
    [onboarding],
  )

  function handleLogin() {
    saveAuth(true)
    setIsAuthenticated(true)
  }

  function handleSignup() {
    saveAuth(true)
    setIsAuthenticated(true)
  }

  function handleCompleteOnboarding(data: UserOnboarding) {
    saveOnboarding(data)
    setOnboarding(data)
  }

  function handleLogout() {
    clearAuth()
    clearOnboarding()
    setIsAuthenticated(false)
    setOnboarding(null)
  }

  return (
    <Routes>
      <Route element={<AppLayout onLogout={handleLogout} isAuthenticated={isAuthenticated} />}>
        {/* Public Routes */}
        <Route
          path="/welcome"
          element={!isAuthenticated ? <WelcomePage /> : <Navigate to="/" replace />}
        />
        <Route
          path="/login"
          element={!isAuthenticated ? <LoginPage onLogin={handleLogin} /> : <Navigate to="/" replace />}
        />
        <Route
          path="/signup"
          element={!isAuthenticated ? <SignupPage onSignup={handleSignup} /> : <Navigate to="/" replace />}
        />

        {/* Authenticated Routes */}
        <Route
          path="/"
          element={
            !isAuthenticated ? (
              <Navigate to="/welcome" replace />
            ) : isOnboarded ? (
              <HomePage onboarding={onboarding} />
            ) : (
              <Navigate to="/onboarding" replace />
            )
          }
        />
        <Route
          path="/onboarding"
          element={
            !isAuthenticated ? (
              <Navigate to="/welcome" replace />
            ) : isOnboarded ? (
              <Navigate to="/" replace />
            ) : (
              <OnboardingPage onComplete={handleCompleteOnboarding} />
            )
          }
        />
        <Route
          path="/anime/:id"
          element={isAuthenticated ? <AnimeDetailPage /> : <Navigate to="/welcome" replace />}
        />
        <Route
          path="/lists"
          element={isAuthenticated ? <ListsPage /> : <Navigate to="/welcome" replace />}
        />
        <Route
          path="/search"
          element={isAuthenticated ? <SearchPage /> : <Navigate to="/welcome" replace />}
        />
        <Route
          path="/profile"
          element={isAuthenticated ? <ProfilePage /> : <Navigate to="/welcome" replace />}
        />
        <Route
          path="/import"
          element={isAuthenticated ? <ImportPage /> : <Navigate to="/welcome" replace />}
        />
      </Route>
    </Routes>
  )
}

export default App
