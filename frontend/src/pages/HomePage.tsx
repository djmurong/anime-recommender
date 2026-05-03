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
    return <p className="text-sm text-slate-400">Loading placeholder recommendations...</p>
  }

  if (!items.length) {
    return (
      <EmptyState
        title="No recommendations available"
        description="Showing fallback content would happen here if model retrieval fails."
      />
    )
  }

  const fallbackReason = items[0]?.fallbackReason ?? 'popularity'

  return (
    <section className="space-y-6">
      <div className="rounded-xl border border-indigo-400/30 bg-indigo-400/10 p-4 text-sm text-indigo-200">
        Fallback chain active: {fallbackReason}. Placeholder serving path for frontend-only mode.
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {items.map((recommendation) => (
          <AnimeCard key={recommendation.anime.id} recommendation={recommendation} />
        ))}
      </div>
    </section>
  )
}
