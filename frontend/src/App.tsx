import { useMemo, useState } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { AppLayout } from './components/AppLayout'
import { clearOnboarding, readOnboarding, saveOnboarding } from './lib/storage'
import { AnimeDetailPage } from './pages/AnimeDetailPage'
import { HomePage } from './pages/HomePage'
import { OnboardingPage } from './pages/OnboardingPage'
import type { UserOnboarding } from './types'

function App() {
  const [onboarding, setOnboarding] = useState<UserOnboarding | null>(() => readOnboarding())

  const isOnboarded = useMemo(
    () => Boolean(onboarding?.favoriteGenreIds.length && onboarding.favoriteAnimeIds.length),
    [onboarding],
  )

  function handleComplete(data: UserOnboarding) {
    saveOnboarding(data)
    setOnboarding(data)
  }

  function handleReset() {
    clearOnboarding()
    setOnboarding(null)
  }

  return (
    <Routes>
      <Route element={<AppLayout onResetOnboarding={handleReset} />}>
        <Route
          path="/"
          element={isOnboarded ? <HomePage onboarding={onboarding} /> : <Navigate to="/onboarding" replace />}
        />
        <Route
          path="/onboarding"
          element={isOnboarded ? <Navigate to="/" replace /> : <OnboardingPage onComplete={handleComplete} />}
        />
        <Route
          path="/anime/:id"
          element={isOnboarded ? <AnimeDetailPage /> : <Navigate to="/onboarding" replace />}
        />
      </Route>
    </Routes>
  )
}

export default App
