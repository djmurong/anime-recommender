import { Link } from 'react-router-dom'

export function WelcomePage() {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center">
      <h1 className="flex items-center justify-center gap-4 text-4xl font-extrabold text-[#2e51a2] md:text-6xl">
        <span className="flex h-12 w-12 items-center justify-center rounded bg-[#2e51a2] text-3xl text-white md:h-16 md:w-16 md:text-4xl">
          次
        </span>
        Tsugi
      </h1>
      <p className="mt-4 text-[16px] text-[#666] max-w-lg">
        Your personalized anime recommender. Track what you've watched and discover what to watch next.
      </p>
      <div className="mt-8 flex gap-4">
        <Link
          to="/login"
          className="rounded bg-[#2e51a2] px-8 py-3 text-[14px] font-bold text-white hover:bg-[#27468d]"
        >
          Sign In
        </Link>
        <Link
          to="/signup"
          className="rounded border border-[#2e51a2] px-8 py-3 text-[14px] font-bold text-[#2e51a2] hover:bg-[#f0f3fa]"
        >
          Create Account
        </Link>
      </div>
    </div>
  )
}
