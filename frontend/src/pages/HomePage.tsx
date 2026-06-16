import { useEffect, useState } from 'react'
import { fetchRecommendations } from '../api/client'
import { AnimeCard } from '../components/AnimeCard'
import { EmptyState } from '../components/EmptyState'
import type { Recommendation, UserOnboarding } from '../types'

interface HomePageProps {
  onboarding: UserOnboarding | null
}

export function HomePage({ onboarding }: HomePageProps) {
  const [items, setItems] = useState<Recommendation[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function load() {
      setLoading(true)
      const list = await fetchRecommendations(onboarding)
      setItems(list)
      setLoading(false)
    }
    void load()
  }, [onboarding])

  if (loading) {
    return (
      <div className="rounded border border-[#bbbbbb] bg-white p-8 text-center text-[12px] text-slate-500">
        Loading your recommendations...
      </div>
    )
  }

  if (!items.length) {
    return (
      <EmptyState
        title="No recommendations available"
        description="Showing fallback content would happen here if model retrieval fails."
      />
    )
  }

  const topMatches = items.slice(0, 5)
  const discovery = items.filter((r) => r.isExploratory)
  const otherRecs = items.filter((r) => !r.isExploratory).slice(5)

  return (
    <div className="flex flex-col gap-4 md:flex-row">
      {/* Left Sidebar - Profile & Stats */}
      <aside className="w-full shrink-0 space-y-4 md:w-[225px]">
        <div className="overflow-hidden rounded border border-[#bbbbbb] bg-white">
          <h2 className="bg-[#2e51a2] px-3 py-1.5 text-[12px] font-bold uppercase tracking-wide text-white">
            Your Profile
          </h2>
          <div className="p-3 text-[11px] text-[#1c1c1c]">
            <div className="mb-3 flex items-center gap-3">
              <img src="https://picsum.photos/seed/user/50/50" alt="Avatar" className="rounded" />
              <div>
                <p className="font-bold text-[#1f4392]">Guest User</p>
                <p className="text-[#666]">Synced just now</p>
              </div>
            </div>
            <div className="space-y-1 border-t border-[#e5e5e5] pt-2">
              <div className="flex justify-between">
                <span className="font-bold text-[#666]">Anime Watched:</span>
                <span className="font-bold">{onboarding?.favoriteAnimeIds.length || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="font-bold text-[#666]">Top Genres:</span>
                <span className="text-right">
                  {onboarding?.favoriteGenreIds.slice(0, 2).join(', ') || 'N/A'}
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="overflow-hidden rounded border border-[#bbbbbb] bg-white">
          <h2 className="bg-[#2e51a2] px-3 py-1.5 text-[12px] font-bold uppercase tracking-wide text-white">
            Model Status
          </h2>
          <div className="p-3 text-[11px] leading-relaxed text-[#1c1c1c]">
            <p>
              <span className="font-bold text-[#666]">Strategy:</span>{' '}
              {items[0]?.fallbackReason === 'onboarding' ? 'Personalized (Two-Tower)' : 'Global Popularity'}
            </p>
            <p>
              <span className="font-bold text-[#666]">Exploration:</span> ~10% Epsilon-Greedy
            </p>
          </div>
        </div>
      </aside>

      {/* Main Content - Recommendations */}
      <main className="flex-1 space-y-6">
        <section>
          <div className="mb-2 flex items-center justify-between border-b border-[#bbbbbb] pb-1">
            <h2 className="text-[14px] font-bold text-[#1f4392]">Top Recommendations For You</h2>
            <span className="text-[11px] text-[#666]">Based on your watch history</span>
          </div>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
            {topMatches.map((rec) => (
              <AnimeCard key={rec.anime.id} recommendation={rec} />
            ))}
          </div>
        </section>

        {discovery.length > 0 && (
          <section>
            <div className="mb-2 flex items-center justify-between border-b border-[#bbbbbb] pb-1">
              <h2 className="text-[14px] font-bold text-[#1f4392]">Discovery Picks</h2>
              <span className="text-[11px] text-[#666]">Outside your usual genres</span>
            </div>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
              {discovery.map((rec) => (
                <AnimeCard key={rec.anime.id} recommendation={rec} />
              ))}
            </div>
          </section>
        )}

        {otherRecs.length > 0 && (
          <section>
            <div className="mb-2 flex items-center justify-between border-b border-[#bbbbbb] pb-1">
              <h2 className="text-[14px] font-bold text-[#1f4392]">More Like This</h2>
            </div>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
              {otherRecs.map((rec) => (
                <AnimeCard key={rec.anime.id} recommendation={rec} />
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  )
}
