import { useMemo, useState } from 'react'
import { searchAnimeCatalog } from '../api/client'
import { genreOptions } from '../lib/mock/anime'
import type { Anime, UserOnboarding } from '../types'

interface OnboardingPageProps {
  onComplete: (data: UserOnboarding) => void
}

export function OnboardingPage({ onComplete }: OnboardingPageProps) {
  const [step, setStep] = useState(1)
  const [genres, setGenres] = useState<string[]>([])
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<Anime[]>([])
  const [favoriteAnimeIds, setFavoriteAnimeIds] = useState<number[]>([])

  const canContinueStepOne = genres.length > 0
  const canFinish = favoriteAnimeIds.length > 0

  const selectedCountLabel = useMemo(
    () => `${favoriteAnimeIds.length} anime selected`,
    [favoriteAnimeIds.length],
  )

  function toggleGenre(genre: string) {
    setGenres((current) =>
      current.includes(genre)
        ? current.filter((item) => item !== genre)
        : [...current, genre],
    )
  }

  function toggleFavoriteAnime(id: number) {
    setFavoriteAnimeIds((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id],
    )
  }

  async function runSearch(value: string) {
    setQuery(value)
    const list = await searchAnimeCatalog(value)
    setResults(list.slice(0, 8))
  }

  function handleSubmit() {
    onComplete({
      favoriteGenreIds: genres,
      favoriteAnimeIds,
      completedAt: new Date().toISOString(),
    })
  }

  return (
    <section className="mx-auto max-w-3xl space-y-6">
      <header>
        <h1 className="text-2xl font-semibold text-slate-100">Welcome to Anime Recommender</h1>
        <p className="mt-2 text-sm text-slate-300">
          Placeholder onboarding for cold-start users. This simulates collecting preference
          signals before model embeddings are available.
        </p>
      </header>

      {step === 1 ? (
        <div className="space-y-4 rounded-xl border border-slate-800 bg-slate-900 p-6">
          <h2 className="text-lg font-semibold">Step 1: Pick your favorite genres</h2>
          <div className="flex flex-wrap gap-2">
            {genreOptions.map((genre) => (
              <button
                key={genre}
                type="button"
                onClick={() => toggleGenre(genre)}
                className={`rounded-full border px-3 py-1 text-sm ${
                  genres.includes(genre)
                    ? 'border-indigo-400 bg-indigo-400/15 text-indigo-200'
                    : 'border-slate-700 text-slate-300 hover:border-indigo-400'
                }`}
              >
                {genre}
              </button>
            ))}
          </div>
          <div className="flex justify-end">
            <button
              type="button"
              disabled={!canContinueStepOne}
              onClick={() => setStep(2)}
              className="rounded-md bg-indigo-500 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
            >
              Continue
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-4 rounded-xl border border-slate-800 bg-slate-900 p-6">
          <h2 className="text-lg font-semibold">Step 2: Choose anime you like</h2>
          <label className="block">
            <span className="mb-2 block text-sm text-slate-300">Search anime titles</span>
            <input
              value={query}
              onChange={(event) => void runSearch(event.target.value)}
              className="w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 focus:border-indigo-400 focus:outline-none"
              placeholder="Try: Skyblade, Metro, Lotus..."
            />
          </label>
          <p className="text-xs text-slate-400">{selectedCountLabel}</p>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            {results.map((anime) => (
              <button
                key={anime.id}
                type="button"
                onClick={() => toggleFavoriteAnime(anime.id)}
                className={`rounded-md border p-3 text-left text-sm ${
                  favoriteAnimeIds.includes(anime.id)
                    ? 'border-emerald-400 bg-emerald-400/10'
                    : 'border-slate-700 hover:border-indigo-400'
                }`}
              >
                <p className="font-medium text-slate-100">{anime.title}</p>
                <p className="text-xs text-slate-400">{anime.genres.join(' · ')}</p>
              </button>
            ))}
          </div>
          <div className="flex items-center justify-between">
            <button
              type="button"
              onClick={() => setStep(1)}
              className="rounded-md border border-slate-700 px-4 py-2 text-sm text-slate-300 hover:border-indigo-400"
            >
              Back
            </button>
            <button
              type="button"
              disabled={!canFinish}
              onClick={handleSubmit}
              className="rounded-md bg-indigo-500 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
            >
              See recommendations
            </button>
          </div>
        </div>
      )}
    </section>
  )
}
