import { useState, useRef, useEffect } from 'react'
import { Link, Outlet } from 'react-router-dom'

interface AppLayoutProps {
  onLogout: () => void
  isAuthenticated: boolean
}

const NAV_LINKS = [
  { label: 'Anime', to: '/' },
  { label: 'My List', to: '/lists' },
  { label: 'Import', to: '/import' },
]

export function AppLayout({ onLogout, isAuthenticated }: AppLayoutProps) {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  return (
    <div className="min-h-screen bg-[#e1e7f5] text-[#1c1c1c]">
      <header className="bg-[#2e51a2] text-white shadow">
        <div className="mx-auto flex w-full max-w-[1024px] items-center gap-6 px-4 py-3">
          <Link to="/" className="flex items-center gap-2 text-2xl font-extrabold tracking-tight text-white">
            <span className="flex h-8 w-8 items-center justify-center rounded bg-white text-xl text-[#2e51a2]">
              次
            </span>
            Tsugi
          </Link>
          {isAuthenticated && (
            <>
              <nav className="hidden items-center gap-4 text-[13px] font-bold md:flex">
                {NAV_LINKS.map((link) => (
                  <Link key={link.label} to={link.to} className="text-white/90 hover:text-white">
                    {link.label}
                  </Link>
                ))}
              </nav>
              <div className="ml-auto flex items-center gap-4">
                <Link to="/search" className="hidden items-center rounded bg-white/95 px-2 py-1 sm:flex text-[#1c1c1c] text-[12px] w-32 justify-between">
                  <span className="text-slate-500">Search</span>
                  <svg className="h-4 w-4 text-[#2e51a2]" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-4.35-4.35M17 11a6 6 0 11-12 0 6 6 0 0112 0z" />
                  </svg>
                </Link>
                <div className="relative flex items-center border-l border-white/20 pl-4" ref={dropdownRef}>
                  <button
                    type="button"
                    onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                    className="flex items-center gap-2 focus:outline-none"
                  >
                    <div className="h-8 w-8 overflow-hidden rounded bg-white/20 ring-2 ring-transparent transition-all hover:ring-white/50">
                      <img src="https://picsum.photos/seed/user/32/32" alt="User" />
                    </div>
                    <svg className={`h-4 w-4 text-white transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>

                  {isDropdownOpen && (
                    <div className="absolute right-0 top-full mt-2 w-48 rounded border border-[#bbbbbb] bg-white py-1 shadow-lg z-50">
                      <Link
                        to="/profile"
                        onClick={() => setIsDropdownOpen(false)}
                        className="block px-4 py-2 text-[13px] font-bold text-[#1c1c1c] hover:bg-[#f0f3fa] hover:text-[#2e51a2]"
                      >
                        Profile
                      </Link>
                      <button
                        type="button"
                        onClick={() => {
                          setIsDropdownOpen(false)
                          onLogout()
                        }}
                        className="block w-full text-left px-4 py-2 text-[13px] font-bold text-[#1c1c1c] hover:bg-[#f0f3fa] hover:text-[#a12f31]"
                      >
                        Logout
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
          {!isAuthenticated && (
            <div className="ml-auto flex items-center gap-4">
              <Link to="/login" className="text-[13px] font-bold text-white/90 hover:text-white">
                Login
              </Link>
              <Link to="/signup" className="rounded border border-white/40 px-3 py-1.5 text-[13px] font-bold text-white hover:bg-white/10">
                Sign Up
              </Link>
            </div>
          )}
        </div>
      </header>

      <main className="mx-auto w-full max-w-[1024px] px-2 py-4 md:px-0 md:py-6">
        <Outlet />
      </main>

      <footer className="mx-auto w-full max-w-[1024px] px-4 py-6 text-center text-[11px] text-slate-500">
        Tsugi · Model: placeholder · Latency: --
      </footer>
    </div>
  )
}
