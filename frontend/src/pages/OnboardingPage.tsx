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
    <section className="mx-auto max-w-3xl space-y-3">
      <div className="overflow-hidden rounded border border-[#bbbbbb] bg-white">
        <div className="flex flex-col items-center justify-center bg-gradient-to-b from-[#2e51a2] to-[#3b62bd] px-4 py-8 text-center text-white">
          <h1 className="flex items-center justify-center gap-3 text-3xl font-extrabold text-white">
            <span className="flex h-10 w-10 items-center justify-center rounded bg-white text-2xl text-[#2e51a2]">
              次
            </span>
            Tsugi
          </h1>
          <p className="mt-1 text-[13px]">
            Set up your profile to get personalized anime recommendations.
          </p>
        </div>

        {step === 1 ? (
          <div className="space-y-4 p-5">
            <h2 className="border-b border-[#e5e5e5] pb-1 text-[14px] font-bold text-[#2e51a2]">
              Step 1: Pick your favorite genres
            </h2>
            <div className="flex flex-wrap gap-2">
              {genreOptions.map((genre) => (
                <button
                  key={genre}
                  type="button"
                  onClick={() => toggleGenre(genre)}
                  className={`rounded border px-3 py-1 text-[12px] font-bold ${
                    genres.includes(genre)
                      ? 'border-[#2e51a2] bg-[#2e51a2] text-white'
                      : 'border-[#bbbbbb] text-[#1f4392] hover:border-[#2e51a2]'
                  }`}
                >
                  {genre}
                </button>
              ))}
            </div>
            <div className="flex justify-end pt-4">
              <button
                type="button"
                disabled={!canContinueStepOne}
                onClick={() => setStep(2)}
                className="rounded bg-[#2e51a2] px-5 py-2 text-[12px] font-bold text-white hover:bg-[#27468d] disabled:cursor-not-allowed disabled:opacity-50"
              >
                Continue
              </button>
            </div>
          </div>
        ) : (
          <div className="space-y-4 p-5">
            <h2 className="border-b border-[#e5e5e5] pb-1 text-[14px] font-bold text-[#2e51a2]">
              Step 2: Choose anime you like
            </h2>
            <label className="block">
              <span className="mb-1 block text-[12px] font-bold text-[#666]">Search anime titles</span>
              <input
                value={query}
                onChange={(event) => void runSearch(event.target.value)}
                className="w-full rounded border border-[#bbbbbb] bg-white px-3 py-2 text-[12px] text-[#1c1c1c] focus:border-[#2e51a2] focus:outline-none"
                placeholder="Try: Skyblade, Metro, Lotus..."
              />
            </label>
            <p className="text-[11px] text-[#666]">{selectedCountLabel}</p>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {results.map((anime) => (
                <button
                  key={anime.id}
                  type="button"
                  onClick={() => toggleFavoriteAnime(anime.id)}
                  className={`flex items-center gap-2 rounded border p-2 text-left ${
                    favoriteAnimeIds.includes(anime.id)
                      ? 'border-[#2db039] bg-[#eefbf0]'
                      : 'border-[#bbbbbb] hover:border-[#2e51a2]'
                  }`}
                >
                  <img
                    src={anime.imageUrl}
                    alt={anime.title}
                    className="h-12 w-10 rounded-sm object-cover"
                  />
                  <span>
                    <span className="block text-[12px] font-bold text-[#1f4392]">{anime.title}</span>
                    <span className="block text-[11px] text-[#666]">{anime.genres.join(', ')}</span>
                  </span>
                </button>
              ))}
            </div>
            <div className="flex items-center justify-between pt-4">
              <button
                type="button"
                onClick={() => setStep(1)}
                className="rounded border border-[#bbbbbb] px-5 py-2 text-[12px] font-bold text-[#1f4392] hover:border-[#2e51a2]"
              >
                Back
              </button>
              <button
                type="button"
                disabled={!canFinish}
                onClick={handleSubmit}
                className="rounded bg-[#2e51a2] px-5 py-2 text-[12px] font-bold text-white hover:bg-[#27468d] disabled:cursor-not-allowed disabled:opacity-50"
              >
                Generate Recommendations
              </button>
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
