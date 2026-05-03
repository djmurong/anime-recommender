import { Link, Outlet } from 'react-router-dom'

interface AppLayoutProps {
  onResetOnboarding: () => void
}

export function AppLayout({ onResetOnboarding }: AppLayoutProps) {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/70 backdrop-blur">
        <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-4 py-4">
          <Link to="/" className="text-lg font-semibold tracking-tight text-indigo-300">
            Anime Recommender
          </Link>
          <nav className="flex items-center gap-4 text-sm text-slate-300">
            <Link className="hover:text-indigo-300" to="/">
              Recommendations
            </Link>
            <button
              type="button"
              onClick={onResetOnboarding}
              className="rounded-md border border-slate-700 px-3 py-1.5 hover:border-indigo-400 hover:text-indigo-300"
            >
              Redo onboarding
            </button>
          </nav>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl px-4 py-8">
        <Outlet />
      </main>

      <footer className="border-t border-slate-800 px-4 py-4 text-center text-xs text-slate-400">
        Model: placeholder · Latency: --
      </footer>
    </div>
  )
}
