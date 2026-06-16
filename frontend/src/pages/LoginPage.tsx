import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

interface LoginPageProps {
  onLogin: () => void
}

export function LoginPage({ onLogin }: LoginPageProps) {
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    // Mock login
    onLogin()
    navigate('/')
  }

  return (
    <div className="mx-auto max-w-md overflow-hidden rounded border border-[#bbbbbb] bg-white">
      <div className="bg-[#2e51a2] px-4 py-4 text-center text-white">
        <h1 className="text-xl font-bold">Sign In</h1>
      </div>
      <form onSubmit={handleSubmit} className="space-y-4 p-6">
        <label className="block">
          <span className="mb-1 block text-[12px] font-bold text-[#666]">Email</span>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full rounded border border-[#bbbbbb] px-3 py-2 text-[12px] focus:border-[#2e51a2] focus:outline-none"
            placeholder="user@example.com"
            required
          />
        </label>
        <label className="block">
          <span className="mb-1 block text-[12px] font-bold text-[#666]">Password</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded border border-[#bbbbbb] px-3 py-2 text-[12px] focus:border-[#2e51a2] focus:outline-none"
            placeholder="••••••••"
            required
          />
        </label>
        <button
          type="submit"
          className="w-full rounded bg-[#2e51a2] px-4 py-2.5 text-[13px] font-bold text-white hover:bg-[#27468d]"
        >
          Sign In
        </button>
      </form>
    </div>
  )
}
